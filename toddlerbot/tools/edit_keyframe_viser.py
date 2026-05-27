from __future__ import annotations

import argparse
import os
import threading
import time
import shutil
from typing import Callable, Dict, List, Optional, Tuple

import joblib
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.io_utils import find_latest_file_with_time_str
from toddlerbot.utils.math_utils import interpolate_action

from viser import GuiEvent

# The current `viser_ui_augmented` branch is a vanilla copy of upstream Viser,

try:
    import viser
except Exception as exc:  # pragma: no cover - helpful import guidance
    raise ImportError(
        "Viser is required for the visor keyframe editor.\n"
        "Make sure to install from the forked viser branch specified in pyproject.toml"
        f"Original import error: {exc}"
    ) from exc


@dataclass
class Keyframe:
    name: str
    motor_pos: np.ndarray
    joint_pos: Optional[np.ndarray] = None
    qpos: Optional[np.ndarray] = None


class SimWorker(threading.Thread):
    """Background worker to mutate MuJoCo state and generate replay arrays.

    Mirrors the logic of UpdateSimThread in edit_keyframe_old, using threading
    instead of Qt. Use the provided lock to synchronize with the UI thread.
    """

    def __init__(
        self,
        sim: MuJoCoSim,
        robot: Robot,
        lock: threading.Lock,
        *,
        on_state: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], None]] = None,
        on_traj: Optional[
            Callable[
                [
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                ],
                None,
            ]
        ] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.sim = sim
        self.robot = robot
        self.lock = lock
        self.on_state = on_state
        self.on_traj = on_traj

        self.running = True
        self.is_testing = False
        self.is_qpos_traj = False
        self.is_relative_frame = True

        self.update_joint_angles_requested = False
        self.joint_angles_to_update = robot.default_joint_angles.copy()

        self.update_qpos_requested = False
        self.qpos_to_update = sim.model.qpos0.copy()

        # Stepping state
        self.keyframe_test_counter = -1
        self.keyframe_test_dt = 0.0

        self.traj_test_counter = -1
        self.action_traj: Optional[List[np.ndarray]] = None
        self.traj_test_dt = 0.0
        self.traj_physics_enabled = False

        # Replay buffers
        self.qpos_replay: List[np.ndarray] = []
        self.body_pos_replay: List[np.ndarray] = []
        self.body_quat_replay: List[np.ndarray] = []
        self.body_lin_vel_replay: List[np.ndarray] = []
        self.body_ang_vel_replay: List[np.ndarray] = []
        self.site_pos_replay: List[np.ndarray] = []
        self.site_quat_replay: List[np.ndarray] = []

    # ----- Requests from UI -----
    def request_state_data(self):
        with self.lock:
            if self.is_testing:
                return
            motor_pos = self.sim.get_motor_angles(type="array")
            joint_pos = self.sim.get_joint_angles(type="array")
            qpos = self.sim.data.qpos.copy()
        if self.on_state:
            self.on_state(motor_pos, joint_pos, qpos)

    def update_joint_angles(self, joint_angles_to_update: Dict[str, float]):
        self.update_joint_angles_requested = True
        self.joint_angles_to_update = joint_angles_to_update.copy()

    def update_qpos(self, qpos: np.ndarray):
        self.update_qpos_requested = True
        self.qpos_to_update = qpos.copy()

    def request_on_ground(self):
        with self.lock:
            if self.is_testing:
                return
            torso_t_curr = self.sim.get_body_transform("torso")
            site_z_min = float("inf")
            min_site = "left_foot_center"
            for site_name in [
                "left_hand_center",
                "right_hand_center",
                "left_foot_center",
                "right_foot_center",
            ]:
                curr_transform = self.sim.get_site_transform(site_name)
                if curr_transform[2, 3] < site_z_min:
                    site_z_min = curr_transform[2, 3]
                    min_site = site_name

            foot_t = self.sim.get_site_transform(min_site)
            aligned_torso_t = foot_t @ np.linalg.inv(foot_t) @ torso_t_curr
            # Translate torso so the min foot touches z=0 plane (approx)
            dz = foot_t[2, 3]
            aligned_torso_t[2, 3] -= dz
            self.sim.data.qpos[:3] = aligned_torso_t[:3, 3]
            self.sim.data.qpos[3:7] = R.from_matrix(aligned_torso_t[:3, :3]).as_quat(
                scalar_first=True
            )
            self.sim.forward()
            print(
                f"[Ground] Placed {min_site} on ground (moved down {dz:.3f}m)",
                flush=True,
            )

            # Trigger UI update callback so sliders refresh
            motor_pos = self.sim.get_motor_angles(type="array")
            joint_pos = self.sim.get_joint_angles(type="array")
            qpos = self.sim.data.qpos.copy()
        if self.on_state:
            self.on_state(motor_pos, joint_pos, qpos)

    def request_keyframe_test(self, keyframe: Keyframe, dt: float):
        with self.lock:
            if self.is_testing:
                return
            self.keyframe_test_counter = -1
            self.traj_test_counter = -1
            self.sim.data.qpos = keyframe.qpos.copy()
            self.sim.forward()
            self.sim.set_motor_target(keyframe.motor_pos.copy())
            self.keyframe_test_dt = dt
            self.keyframe_test_counter = 0
            self.is_testing = True

    def request_trajectory_test(
        self,
        qpos_start: np.ndarray,
        traj: List[np.ndarray],
        dt: float,
        physics_enabled: bool,
        *,
        is_qpos_traj: bool = False,
        is_relative_frame: bool = True,
    ):
        with self.lock:
            if self.is_testing:
                print(
                    "[Viser] Worker: request_trajectory_test ignored (already testing)",
                    flush=True,
                )
                return
            self.keyframe_test_counter = -1
            self.traj_test_counter = -1

            self.sim.data.qpos = qpos_start.copy()
            self.sim.data.qvel[:] = 0
            self.sim.data.ctrl[:] = 0
            self.sim.forward()

            self.action_traj = traj
            self.traj_test_dt = dt
            self.traj_physics_enabled = physics_enabled
            self.traj_test_counter = 0
            self.is_testing = True
            self.is_qpos_traj = is_qpos_traj
            self.is_relative_frame = is_relative_frame

            try:
                print(
                    f"[Viser] Worker: start trajectory test: len={len(traj)}, dt={dt}, physics={physics_enabled}, qpos={is_qpos_traj}, rel={is_relative_frame}",
                    flush=True,
                )
            except Exception:
                pass

            # Clear replay
            self.qpos_replay.clear()
            self.body_pos_replay.clear()
            self.body_quat_replay.clear()
            self.body_lin_vel_replay.clear()
            self.body_ang_vel_replay.clear()
            self.site_pos_replay.clear()
            self.site_quat_replay.clear()

    def stop(self):
        self.running = False

    # ----- Main loop -----
    def run(self) -> None:
        while self.running:
            if self.update_qpos_requested:
                with self.lock:
                    self.is_testing = False
                    self.keyframe_test_counter = -1
                    self.traj_test_counter = -1
                    self.sim.data.qpos = self.qpos_to_update.copy()
                    self.sim.forward()
                    self.update_qpos_requested = False
                time.sleep(0)  # yield
                continue

            if self.update_joint_angles_requested:
                with self.lock:
                    self.is_testing = False
                    self.keyframe_test_counter = -1
                    self.traj_test_counter = -1
                    joint_angles = self.sim.get_joint_angles()
                    joint_angles.update(self.joint_angles_to_update)
                    self.sim.set_joint_angles(joint_angles)
                    self.sim.forward()
                    self.update_joint_angles_requested = False
                time.sleep(0)  # yield
                continue

            if 0 <= self.keyframe_test_counter <= 100:
                with self.lock:
                    if self.keyframe_test_counter == 100:
                        self.keyframe_test_counter = -1
                        self.is_testing = False
                    else:
                        self.sim.step()
                        self.keyframe_test_counter += 1
                time.sleep(self.keyframe_test_dt)
                continue

            # Trajectory test
            if self.traj_test_counter >= 0 and self.action_traj is not None:
                # Check stop
                with self.lock:
                    trajectory_running = self.is_testing
                    current_counter = self.traj_test_counter
                    traj_len = len(self.action_traj)
                if current_counter == 0:
                    try:
                        print(
                            f"[Viser] Worker: stepping trajectory... len={traj_len}, dt={self.traj_test_dt}, physics={self.traj_physics_enabled}",
                            flush=True,
                        )
                    except Exception:
                        pass
                if not trajectory_running:
                    # Emit and clear
                    if self.on_traj:
                        self.on_traj(
                            self.qpos_replay.copy(),
                            self.body_pos_replay.copy(),
                            self.body_quat_replay.copy(),
                            self.body_lin_vel_replay.copy(),
                            self.body_ang_vel_replay.copy(),
                            self.site_pos_replay.copy(),
                            self.site_quat_replay.copy(),
                        )
                    with self.lock:
                        self.traj_test_counter = -1
                        self.keyframe_test_counter = -1
                        self.action_traj = None
                        self.qpos_replay.clear()
                        self.body_pos_replay.clear()
                        self.body_quat_replay.clear()
                        self.body_lin_vel_replay.clear()
                        self.body_ang_vel_replay.clear()
                        self.site_pos_replay.clear()
                        self.site_quat_replay.clear()
                    time.sleep(0)
                    continue

                # If trajectory is exhausted or empty, finalize and emit once
                if current_counter >= traj_len:
                    try:
                        print(
                            f"[Viser] Worker: trajectory complete. frames={len(self.qpos_replay)}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    if self.on_traj:
                        self.on_traj(
                            self.qpos_replay.copy(),
                            self.body_pos_replay.copy(),
                            self.body_quat_replay.copy(),
                            self.body_lin_vel_replay.copy(),
                            self.body_ang_vel_replay.copy(),
                            self.site_pos_replay.copy(),
                            self.site_quat_replay.copy(),
                        )
                    with self.lock:
                        self.traj_test_counter = -1
                        self.keyframe_test_counter = -1
                        self.action_traj = None
                        self.is_testing = False
                        self.qpos_replay.clear()
                        self.body_pos_replay.clear()
                        self.body_quat_replay.clear()
                        self.body_lin_vel_replay.clear()
                        self.body_ang_vel_replay.clear()
                        self.site_pos_replay.clear()
                        self.site_quat_replay.clear()
                    time.sleep(0)
                    continue

                # Step one action
                t1 = time.monotonic()
                with self.lock:
                    if self.is_qpos_traj:
                        qpos_goal = self.action_traj[current_counter]
                        self.sim.set_qpos(qpos_goal)
                        self.sim.forward()
                    else:
                        target = self.action_traj[current_counter]
                        if self.traj_physics_enabled:
                            # With physics enabled, command targets and step dynamics
                            self.sim.set_motor_target(target)
                            self.sim.step()
                        else:
                            # Without physics, directly apply joint angles for visible motion
                            self.sim.set_motor_angles(target)
                            self.sim.forward()

                    # Record
                    qpos_data = self.sim.data.qpos.copy()
                    torso_rot = R.from_quat(
                        self.sim.data.body("torso").xquat.copy(), scalar_first=True
                    )
                    r_inv = torso_rot.inv()

                    if self.is_relative_frame:
                        body_pos_world = np.array(self.sim.data.xpos, dtype=np.float32)
                        body_quat_world = np.array(
                            self.sim.data.xquat, dtype=np.float32
                        )
                        body_pos = []
                        body_quat = []
                        for i in range(self.sim.model.nbody):
                            p = body_pos_world[i]
                            q = body_quat_world[i]
                            body_pos.append(
                                r_inv.apply(p - self.sim.data.body("torso").xpos)
                            )
                            # Convert world quat to torso-relative by q_rel = q_inv(torso)*q_body
                            q_rel = (r_inv * R.from_quat(q, scalar_first=True)).as_quat(
                                scalar_first=True
                            )
                            body_quat.append(q_rel)
                        body_pos = np.array(body_pos, dtype=np.float32)
                        body_quat = np.array(body_quat, dtype=np.float32)
                        body_lin_vel_world = np.array(
                            self.sim.data.cvel[:, 3:], dtype=np.float32
                        )
                        body_ang_vel_world = np.array(
                            self.sim.data.cvel[:, :3], dtype=np.float32
                        )
                        body_lin_vel = r_inv.apply(body_lin_vel_world)
                        body_ang_vel = r_inv.apply(body_ang_vel_world)
                        site_pos = []
                        site_quat = []
                        for side in ["left", "right"]:
                            for ee in ["hand", "foot"]:
                                sname = f"{side}_{ee}_center"
                                ee_pos_world = self.sim.data.site(sname).xpos.copy()
                                ee_mat = self.sim.data.site(sname).xmat.reshape(3, 3)
                                ee_quat_world = R.from_matrix(ee_mat).as_quat(
                                    scalar_first=True
                                )
                                site_pos.append(ee_pos_world)
                                site_quat.append(ee_quat_world)
                    else:
                        body_pos = np.array(self.sim.data.xpos, dtype=np.float32)
                        body_quat = np.array(self.sim.data.xquat, dtype=np.float32)
                        body_lin_vel = np.array(
                            self.sim.data.cvel[:, 3:], dtype=np.float32
                        )
                        body_ang_vel = np.array(
                            self.sim.data.cvel[:, :3], dtype=np.float32
                        )
                        site_pos = []
                        site_quat = []
                        for side in ["left", "right"]:
                            for ee in ["hand", "foot"]:
                                sname = f"{side}_{ee}_center"
                                ee_pos_world = self.sim.data.site(sname).xpos.copy()
                                ee_mat = self.sim.data.site(sname).xmat.reshape(3, 3)
                                ee_quat_world = R.from_matrix(ee_mat).as_quat(
                                    scalar_first=True
                                )
                                site_pos.append(ee_pos_world)
                                site_quat.append(ee_quat_world)

                # Append outside of lock
                self.qpos_replay.append(qpos_data)
                self.body_pos_replay.append(body_pos)
                self.body_quat_replay.append(body_quat)
                self.body_lin_vel_replay.append(body_lin_vel)
                self.body_ang_vel_replay.append(body_ang_vel)
                self.site_pos_replay.append(np.array(site_pos, dtype=np.float32))
                self.site_quat_replay.append(np.array(site_quat, dtype=np.float32))

                self.traj_test_counter += 1
                try:
                    if (
                        self.traj_test_counter
                        % max(1, int(0.5 / max(self.traj_test_dt, 1e-6)))
                        == 0
                    ):
                        print(
                            f"[Viser] Worker: progressed to step {self.traj_test_counter}/{traj_len}",
                            flush=True,
                        )
                except Exception:
                    pass
                t2 = time.monotonic()
                dt_left = self.traj_test_dt - (t2 - t1)
                if dt_left > 0:
                    time.sleep(dt_left)
                else:
                    time.sleep(0.001)
                continue

            time.sleep(0.005)


class ViserKeyframeEditor:
    """Controller that manages the Viser server and MuJoCo geometry.

    Geometry-specific helpers are ported from the legacy Viser editor. The class
    is intentionally minimal for now; additional UI and editing features will be
    layered on in subsequent milestones.
    """

    def __init__(
        self,
        sim: MuJoCoSim,
        robot: Robot,
        *,
        task_name: str,
        run_name: str = "",
        xml_path: str = "",
        dt: float = 0.02,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.dt = dt
        self.xml_path = xml_path
        self.task_name = task_name
        self.run_name = run_name

        if run_name == task_name:
            time_str = time.strftime("%Y%m%d_%H%M%S")
            self.result_dir = os.path.join(
                "results", f"{robot.name}_keyframe_{sim.name}_{time_str}"
            )
            os.makedirs(self.result_dir, exist_ok=True)
            self.data_path = os.path.join(self.result_dir, f"{task_name}.lz4")
            motion_src = os.path.join("motion", f"{task_name}.lz4")
            if os.path.exists(motion_src):
                shutil.copy2(motion_src, self.data_path)
        elif len(run_name) > 0:
            self.data_path = find_latest_file_with_time_str(
                os.path.join("results", run_name), task_name
            )
            if self.data_path is None:
                self.data_path = os.path.join("results", run_name, f"{task_name}.lz4")
            self.result_dir = os.path.dirname(self.data_path)
        else:
            self.data_path = ""
            time_str = time.strftime("%Y%m%d_%H%M%S")
            self.result_dir = os.path.join(
                "results", f"{robot.name}_keyframe_{sim.name}_{time_str}"
            )
            os.makedirs(self.result_dir, exist_ok=True)

        self.mirror_joint_signs = {
            "left_hip_pitch": -1,
            "left_hip_roll": -1,
            "left_hip_yaw_driven": -1,
            "left_knee": -1,
            "left_ankle_pitch": -1,
            "left_ankle_roll": -1,
            "left_shoulder_pitch": -1,
            "left_shoulder_roll": 1,
            "left_shoulder_yaw_driven": -1,
            "left_elbow_roll": 1,
            "left_elbow_yaw_driven": -1,
            "left_wrist_pitch_driven": -1,
            "left_wrist_roll": 1,
            "left_gripper_pinion": 1,
        }
        if self.robot.name == "toddlerbot_2xc":
            self.mirror_joint_signs["left_hip_roll"] = 1

        # State
        self.keyframes: List[Keyframe] = []
        self.sequence_list: List[Tuple[str, float]] = []
        self.selected_keyframe: Optional[int] = None
        self.selected_sequence: Optional[int] = None
        self.traj_times: List[float] = []
        self.action_traj: Optional[List[np.ndarray]] = None
        self.is_qpos_traj = False
        self.is_relative_frame = True
        self.qpos_replay: List[np.ndarray] = []
        self.body_pos_replay: List[np.ndarray] = []
        self.body_quat_replay: List[np.ndarray] = []
        self.body_lin_vel_replay: List[np.ndarray] = []
        self.body_ang_vel_replay: List[np.ndarray] = []
        self.site_pos_replay: List[np.ndarray] = []
        self.site_quat_replay: List[np.ndarray] = []

        self.saved_left_foot_pose = None
        self.saved_right_foot_pose = None

        # Scene bookkeeping (populated by `_build_robot_meshes`).
        self._geom_handles: Dict[int, object] = {}
        self._geom_groups: Dict[int, int] = {}
        self._geom_base_rgba: Dict[int, Tuple[float, float, float, float]] = {}
        self._scene_handles: Dict[str, object] = {}
        self._mesh_file_map: Dict[str, str] = {}
        self._mesh_scale_map: Dict[str, Tuple[float, float, float]] = {}
        self._mesh_quat_map: Dict[str, Tuple[float, float, float, float]] = {}
        self._com_sphere: Optional[object] = None
        self._scene_updater: Optional[threading.Thread] = None

        # GUI bookkeeping.
        self.slider_widgets: Dict[str, viser.GuiSliderHandle] = {}
        self.collision_geom_checked: Optional[viser.GuiCheckboxHandle] = None
        self.show_all_geoms: Optional[viser.GuiCheckboxHandle] = None
        self.motion_name_input: Optional[viser.GuiTextHandle] = None
        self.keyframes_summary: Optional[viser.GuiHtmlHandle] = None
        self.keyframe_index_input: Optional[viser.GuiTextHandle] = None
        self.keyframe_name_input: Optional[viser.GuiTextHandle] = None
        self.sequence_summary: Optional[viser.GuiHtmlHandle] = None
        self.sequence_index_input: Optional[viser.GuiTextHandle] = None
        self.sequence_time_input: Optional[viser.GuiTextHandle] = None
        self.mirror_checked: Optional[viser.GuiCheckboxHandle] = None
        self.rev_mirror_checked: Optional[viser.GuiCheckboxHandle] = None
        self.physics_enabled: Optional[viser.GuiCheckboxHandle] = None
        self.relative_frame_checked: Optional[viser.GuiCheckboxHandle] = None

        self._updating_handles: set[int] = set()
        self.normalized_range = (-2000.0, 2000.0)

        # Lock shared between geometry updates and worker callbacks.
        self.worker_lock = threading.Lock()

        # The Viser server hosts the UI + WebGL viewport.
        self.server = viser.ViserServer(label="Keyframe Editor (New)")
        try:
            self.server.gui.configure_theme(
                control_layout="fixed",
                control_width="large",
            )
        except Exception as exc:  # pragma: no cover - best effort configuration
            print(f"[Viser] configure_theme failed: {exc}", flush=True)
        self.server.scene.add_grid("/grid", width=20, height=20, infinite_grid=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            """Apply a sensible default orbit camera when a client connects."""
            camera_pos = (1.5, -0.5, 0.55)
            look_at_pos = (0.0, 0.0, 0.25)
            try:
                client.camera.position = camera_pos
                client.camera.look_at = look_at_pos
                if hasattr(client.camera, "up"):
                    client.camera.up = (0.0, 0.0, 1.0)
                if hasattr(client.camera, "vertical_fov"):
                    client.camera.vertical_fov = 55.0
            except Exception:
                # The camera API is version-dependent; swallow attribute errors.
                pass

        try:
            self.sim.forward()
        except Exception:
            pass

        self._build_ui()

        # Background worker for physics interactions.
        self.worker = SimWorker(
            self.sim,
            self.robot,
            self.worker_lock,
            on_state=self._on_state,
            on_traj=self._on_traj,
        )
        self.worker.start()

        self._build_robot_meshes()

        if not self._geom_handles:
            try:
                self._scene_handles["torso_frame"] = self.server.scene.add_frame(
                    "/robot/torso",
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.0, 0.0, 0.0),
                )
                for body_index in range(self.sim.model.nbody):
                    name = (
                        mujoco.mj_id2name(
                            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, body_index
                        )
                        or f"body_{body_index}"
                    )
                    try:
                        self._scene_handles[f"body_{body_index}"] = (
                            self.server.scene.add_icosphere(
                                f"/robot/bodies/{body_index:04d}_{name}",
                                radius=0.02,
                                position=(0.0, 0.0, 0.0),
                                color=(0.7, 0.7, 0.7),
                            )
                        )
                    except Exception:
                        continue
            except Exception:
                self._scene_handles.clear()

        self._apply_geom_visibility()

        try:
            self._com_sphere = self.server.scene.add_icosphere(
                "/robot/com",
                radius=0.03,
                position=(0.0, 0.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            print("[Viser] Center of mass sphere added (red ball)", flush=True)
        except Exception as exc:
            print(f"[Viser] Failed to add CoM sphere: {exc}", flush=True)
            self._com_sphere = None

        self._start_scene_updater()
        self._load_data()

    # --------------------------------------------------------------------- init
    # -- GUI helpers ------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the three-column GUI layout (controls + joint sliders)."""

        left_joints, right_joints = self._split_joint_lists()
        columns_handle: Optional[viser.GuiColumnsHandle]
        try:
            columns_handle = self.server.gui.add_columns(
                3,
                widths=(0.25, 0.25, 0.25),
            )
            print(
                "[Viser] Using gui.add_columns(3) for controls + joint sliders.",
                flush=True,
            )
        except Exception as exc:
            columns_handle = None
            print(
                f"[Viser] add_columns(3) unavailable, falling back to single column: {exc}",
                flush=True,
            )

        if columns_handle is not None:
            controls_col, left_col, right_col = columns_handle
            with controls_col:
                self._build_controls_panel()
                self._build_keyframe_sequence_panels()
            with left_col:
                self._build_joint_slider_column(left_joints, "Joint Sliders (L)")
                self._build_foot_hand_panel()
            with right_col:
                self._build_joint_slider_column(right_joints, "Joint Sliders (R)")
                self._build_settings_panel()
            return

        # Fallback layout if multi-column support is unavailable.
        self._build_controls_panel()
        self._build_keyframe_sequence_panels()
        self._build_joint_slider_column(left_joints, "Joint Sliders (L)")
        self._build_joint_slider_column(right_joints, "Joint Sliders (R)")
        self._build_foot_hand_panel()
        self._build_settings_panel()

    def _build_controls_panel(self) -> None:
        """Populate the controls column with visualization toggles."""

        if self.motion_name_input is not None:
            # Controls already constructed.
            return

        self.motion_name_input = self.server.gui.add_text(
            "Motion Name",
            self.task_name,
        )
        save_btn = self.server.gui.add_button("💾 Save Motion")

        @save_btn.on_click
        def _(_e: GuiEvent) -> None:
            self._save_data()

        with self.server.gui.add_folder("🔑 Keyframe Operations"):
            keyframe_ops_row1 = self.server.gui.add_button_group(
                "Actions",
                ["Add", "Remove", "Update"],
            )

            @keyframe_ops_row1.on_click
            def _(_e: GuiEvent) -> None:
                val = keyframe_ops_row1.value
                if val == "Add":
                    self._add_keyframe()
                elif val == "Remove":
                    self._remove_keyframe()
                elif val == "Update":
                    self.worker.request_state_data()

            keyframe_ops_row2 = self.server.gui.add_button_group(
                "Actions",
                ["Test", "Ground"],
            )

            @keyframe_ops_row2.on_click
            def _(_e: GuiEvent) -> None:
                val = keyframe_ops_row2.value
                if val == "Test":
                    self._test_keyframe()
                elif val == "Ground":
                    self.worker.request_on_ground()

        with self.server.gui.add_folder("🎬 Sequence Operations"):
            seq_ops_row1 = self.server.gui.add_button_group(
                "Sequence",
                ["Add to Seq", "Remove from Seq"],
            )

            @seq_ops_row1.on_click
            def _(_e: GuiEvent) -> None:
                val = seq_ops_row1.value
                if val == "Add to Seq":
                    self._add_to_sequence()
                elif val == "Remove from Seq":
                    self._remove_from_sequence()

            seq_ops_row2 = self.server.gui.add_button_group(
                "Sequence",
                ["Play Traj", "Play Qpos", "Stop"],
            )

            @seq_ops_row2.on_click
            def _(_e: GuiEvent) -> None:
                val = seq_ops_row2.value
                if val == "Play Traj":
                    self._test_trajectory()
                elif val == "Play Qpos":
                    self._test_qpos_trajectory()
                elif val == "Stop":
                    with self.worker.lock:
                        self.worker.is_testing = False

    def _build_keyframe_sequence_panels(self) -> None:
        if self.keyframes_summary is None:
            with self.server.gui.add_folder("Keyframes List"):
                self.keyframes_summary = self.server.gui.add_html("No keyframes")
                self.keyframes_summary.scroll = {"enable": True, "max_height": 200}

            with self.server.gui.add_folder("Keyframe Controls"):
                self.keyframe_index_input = self.server.gui.add_text(
                    "Selected Keyframe Index",
                    "-1",
                )
                self.keyframe_name_input = self.server.gui.add_text(
                    "Keyframe Name",
                    self.task_name if self.task_name else "",
                )
                keyframe_actions = self.server.gui.add_button_group(
                    "Actions",
                    ["Copy", "Move Up", "Move Down"],
                )

                @keyframe_actions.on_click
                def _(_e: GuiEvent) -> None:
                    val = keyframe_actions.value
                    if val == "Duplicate":
                        self._duplicate_selected_keyframe()
                    elif val == "Move Up":
                        self._move_keyframe(-1)
                    elif val == "Move Down":
                        self._move_keyframe(1)

                assert self.keyframe_index_input is not None
                assert self.keyframe_name_input is not None

                @self.keyframe_index_input.on_update
                def _(ev: GuiEvent) -> None:
                    if id(ev.target) in self._updating_handles:
                        return
                    try:
                        idx = int(str(self.keyframe_index_input.value))
                    except Exception:
                        return
                    if 0 <= idx < len(self.keyframes):
                        self.selected_keyframe = idx
                        self.keyframe_name_input.value = self.keyframes[idx].name
                        self._load_keyframe_to_ui(idx)

                @self.keyframe_name_input.on_update
                def _(ev: GuiEvent) -> None:
                    if (
                        id(ev.target) in self._updating_handles
                        or self.selected_keyframe is None
                    ):
                        return
                    new_name = str(self.keyframe_name_input.value).strip()
                    if not new_name:
                        return
                    for i, kf in enumerate(self.keyframes):
                        if i != self.selected_keyframe and kf.name == new_name:
                            return
                    old_name = self.keyframes[self.selected_keyframe].name
                    self.keyframes[self.selected_keyframe].name = new_name
                    for i, (n, t) in enumerate(self.sequence_list):
                        if n == old_name:
                            self.sequence_list[i] = (new_name, t)
                    self._refresh_keyframes_summary()
                    self._refresh_sequence_summary()

        if self.sequence_summary is None:
            with self.server.gui.add_folder("Sequence List"):
                self.sequence_summary = self.server.gui.add_html("No sequence")
                self.sequence_summary.scroll = {"enable": True, "max_height": 200}

            with self.server.gui.add_folder("Sequence Controls"):
                self.sequence_index_input = self.server.gui.add_text(
                    "Selected Sequence Index",
                    "-1",
                )
                self.sequence_time_input = self.server.gui.add_text(
                    "Arrival Time (t)",
                    "0.0",
                )
                sequence_actions = self.server.gui.add_button_group(
                    "Actions",
                    ["Move Up", "Move Down"],
                )

                @sequence_actions.on_click
                def _(_e: GuiEvent) -> None:
                    val = sequence_actions.value
                    if val == "Move Up":
                        self._move_sequence(-1)
                    elif val == "Move Down":
                        self._move_sequence(1)

                assert self.sequence_index_input is not None
                assert self.sequence_time_input is not None

                @self.sequence_index_input.on_update
                def _(ev: GuiEvent) -> None:
                    if id(ev.target) in self._updating_handles:
                        return
                    try:
                        idx = int(str(self.sequence_index_input.value))
                    except Exception:
                        return
                    if 0 <= idx < len(self.sequence_list):
                        self.selected_sequence = idx
                        name, t = self.sequence_list[idx]
                        self._set_handle_value(self.sequence_time_input, f"{float(t)}")

                @self.sequence_time_input.on_update
                def _(ev: GuiEvent) -> None:
                    if (
                        id(ev.target) in self._updating_handles
                        or self.selected_sequence is None
                    ):
                        return
                    try:
                        new_t = float(str(self.sequence_time_input.value))
                    except Exception:
                        return
                    self._edit_sequence_time(self.selected_sequence, new_t)

    def _split_joint_lists(self) -> Tuple[List[str], List[str]]:
        """Split joints into two buckets for left/right slider columns."""

        left_column: List[str] = []
        right_column: List[str] = []
        processed: set[str] = set()

        for joint in self.robot.joint_ordering:
            if "left" in joint and joint not in processed:
                partner = joint.replace("left", "right", 1)
                if partner in self.robot.joint_ordering:
                    left_column.append(joint)
                    right_column.append(partner)
                    processed.add(joint)
                    processed.add(partner)

        unpaired = [
            joint for joint in self.robot.joint_ordering if joint not in processed
        ]
        midpoint = (len(unpaired) + 1) // 2
        left_column.extend(unpaired[:midpoint])
        right_column.extend(unpaired[midpoint:])

        return left_column, right_column

    def _build_joint_slider_column(self, joints: List[str], title: str) -> None:
        """Create a folder populated with sliders for the provided joints."""

        if not joints:
            with self.server.gui.add_folder(title):
                self.server.gui.add_markdown("_No joints available._")
            return

        with self.server.gui.add_folder(title):
            for joint_name in joints:
                self._create_joint_slider(joint_name)

    def _create_joint_slider(self, joint_name: str) -> Optional[viser.GuiSliderHandle]:
        """Create a GUI slider for a single joint."""

        if joint_name in self.slider_widgets:
            return self.slider_widgets[joint_name]

        limits = self.robot.joint_limits.get(joint_name)
        if limits is None:
            print(
                f"[Viser] Missing joint limits for '{joint_name}', skipping slider.",
                flush=True,
            )
            return None

        jmin, jmax = float(limits[0]), float(limits[1])
        rounded_min = round(jmin, 2)
        rounded_max = round(jmax, 2)
        span = max(rounded_max - rounded_min, 1e-6)
        step = max(span / 4000.0, 1e-4)
        default_val = float(
            self.robot.default_joint_angles.get(joint_name, (jmin + jmax) * 0.5)
        )
        default_val = min(max(default_val, rounded_min), rounded_max)

        label = self._format_joint_label(joint_name)
        slider = self.server.gui.add_slider(
            label,
            min=rounded_min,
            max=rounded_max,
            step=step,
            initial_value=default_val,
        )
        self.slider_widgets[joint_name] = slider
        slider.precision = 4
        slider.value = round(float(slider.value), 4)

        @slider.on_update  # type: ignore[misc]
        def _(_event: GuiEvent, jname=joint_name, sld=slider) -> None:
            if id(sld) in self._updating_handles:
                return
            try:
                value = round(float(sld.value), 4)
                if sld.value != value:
                    self._set_handle_value(sld, value)
            except Exception:
                return
            updates = self._update_joint_pos(jname, value)
            for name, angle in updates.items():
                if name == jname:
                    continue
                other = self.slider_widgets.get(name)
                if other is not None:
                    self._set_handle_value(other, float(angle))

        return slider

    # -- Geometry helpers (to be populated with the full implementations) ------

    def _build_robot_meshes(self) -> None:
        """Populate `self._geom_handles` with MuJoCo meshes."""
        import xml.etree.ElementTree as ET

        import trimesh

        self._geom_handles.clear()
        self._geom_groups.clear()
        self._geom_base_rgba.clear()
        self._mesh_file_map.clear()
        self._mesh_scale_map.clear()
        self._mesh_quat_map.clear()

        m = self.sim.model

        # Parse materials from XML (MuJoCo doesn't populate geom_rgba when asset comes after worldbody)
        material_colors: Dict[str, Tuple[float, float, float, float]] = {}
        geom_materials: Dict[str, str] = {}

        def parse_materials_from_xml(xml_file: str, visited: set[str]) -> None:
            """Parse material definitions and geom material references."""
            try:
                xml_file = os.path.abspath(xml_file)
                if xml_file in visited:
                    return
                visited.add(xml_file)
                tree = ET.parse(xml_file)
                root = tree.getroot()
                basedir = os.path.dirname(xml_file)

                # Parse material definitions
                for material in root.findall(".//material[@rgba]"):
                    mat_name = material.attrib.get("name")
                    rgba_str = material.attrib.get("rgba", "0.5 0.5 0.5 1")
                    rgba = tuple(map(float, rgba_str.split()))
                    if mat_name:
                        material_colors[mat_name] = rgba

                # Parse geom material references
                for geom in root.findall(".//geom[@material]"):
                    geom_name = geom.attrib.get("name")
                    mat_ref = geom.attrib.get("material")
                    if geom_name and mat_ref:
                        geom_materials[geom_name] = mat_ref

                # Follow includes
                for inc in root.findall(".//include"):
                    inc_file = inc.attrib.get("file")
                    if inc_file:
                        inc_path = (
                            inc_file
                            if os.path.isabs(inc_file)
                            else os.path.join(basedir, inc_file)
                        )
                        if os.path.exists(inc_path):
                            parse_materials_from_xml(inc_path, visited)
            except Exception as exc:
                print(
                    f"[Viser] Warning: Failed to parse materials from {xml_file}: {exc}",
                    flush=True,
                )

        # Parse materials from the scene XML
        if self.xml_path and os.path.exists(self.xml_path):
            parse_materials_from_xml(self.xml_path, set())

        print(
            f"[Viser] Parsed {len(material_colors)} materials, {len(geom_materials)} geom references",
            flush=True,
        )

        # Build maps from mesh name -> file path, asset-scale, and asset-orientation by parsing XML (handles <include>)
        def parse_xml_meshes(xml_file: str, visited: set[str]) -> None:
            try:
                xml_file = os.path.abspath(xml_file)
                if xml_file in visited:
                    return
                visited.add(xml_file)
                tree = ET.parse(xml_file)
                root = tree.getroot()
                basedir = os.path.dirname(xml_file)
                meshdir = root.attrib.get("meshdir", "")
                comp = root.find("compiler")
                if comp is not None and comp.attrib.get("meshdir"):
                    meshdir = comp.attrib.get("meshdir")
                asset_base = os.path.join(basedir, meshdir) if meshdir else basedir

                for mesh in root.findall(".//asset/mesh"):
                    name = mesh.attrib.get("name")
                    file = mesh.attrib.get("file")
                    if not name and file:
                        name = os.path.splitext(os.path.basename(file))[0]
                    if name and file:
                        fpath = (
                            file
                            if os.path.isabs(file)
                            else os.path.join(asset_base, file)
                        )
                        if os.path.exists(fpath):
                            self._mesh_file_map.setdefault(name, fpath)
                        scale_txt = mesh.attrib.get("scale")
                        if scale_txt:
                            try:
                                vals = [float(x) for x in scale_txt.strip().split()]
                                if len(vals) == 1:
                                    s = (vals[0], vals[0], vals[0])
                                elif len(vals) == 3:
                                    s = (vals[0], vals[1], vals[2])
                                else:
                                    s = (1.0, 1.0, 1.0)
                                self._mesh_scale_map[name] = s
                            except Exception:
                                pass
                        quat_txt = mesh.attrib.get("quat")
                        if quat_txt:
                            try:
                                vals = [float(x) for x in quat_txt.strip().split()]
                                if len(vals) == 4:
                                    self._mesh_quat_map[name] = (
                                        vals[0],
                                        vals[1],
                                        vals[2],
                                        vals[3],
                                    )
                            except Exception:
                                pass

                for inc in root.findall(".//include"):
                    inc_file = inc.attrib.get("file")
                    if inc_file:
                        inc_path = (
                            inc_file
                            if os.path.isabs(inc_file)
                            else os.path.join(basedir, inc_file)
                        )
                        if os.path.exists(inc_path):
                            parse_xml_meshes(inc_path, visited)
            except Exception:
                return

        if self.xml_path and os.path.exists(self.xml_path):
            parse_xml_meshes(self.xml_path, set())
        else:
            desc_dir = os.path.join("toddlerbot", "descriptions", self.robot.name)
            for fname in (
                "scene.xml",
                "scene_pos.xml",
                "scene_fixed.xml",
                "scene_pos_fixed.xml",
            ):
                cand = os.path.join(desc_dir, fname)
                if os.path.exists(cand):
                    parse_xml_meshes(cand, set())
                    break

        try:
            geom_type = np.array(m.geom_type, dtype=np.int32)
            geom_size = np.array(m.geom_size, dtype=np.float32)
            geom_rgba = (
                np.array(m.geom_rgba, dtype=np.float32)
                if hasattr(m, "geom_rgba")
                else np.tile(
                    np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32), (m.ngeom, 1)
                )
            )
            geom_group = (
                np.array(m.geom_group, dtype=np.int32)
                if hasattr(m, "geom_group")
                else np.zeros(m.ngeom, dtype=np.int32)
            )
            geom_dataid = (
                np.array(m.geom_dataid, dtype=np.int32)
                if hasattr(m, "geom_dataid")
                else np.full(m.ngeom, -1, dtype=np.int32)
            )
            # geom_contype not used in this implementation
        except Exception:
            geom_type = np.array(
                [m.geom(i).type for i in range(m.ngeom)], dtype=np.int32
            )
            geom_size = np.array(
                [m.geom(i).size for i in range(m.ngeom)], dtype=np.float32
            )
            try:
                geom_rgba = np.array(
                    [m.geom(i).rgba for i in range(m.ngeom)], dtype=np.float32
                )
            except Exception:
                geom_rgba = np.tile(
                    np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32), (m.ngeom, 1)
                )
            try:
                geom_group = np.array(
                    [m.geom(i).group for i in range(m.ngeom)], dtype=np.int32
                )
            except Exception:
                geom_group = np.zeros(m.ngeom, dtype=np.int32)
            try:
                geom_dataid = np.array(
                    [m.geom(i).dataid for i in range(m.ngeom)], dtype=np.int32
                )
            except Exception:
                geom_dataid = np.full(m.ngeom, -1, dtype=np.int32)
            # geom_contype not used in this implementation

        mesh_vert = getattr(m, "mesh_vert", None)
        mesh_face = getattr(m, "mesh_face", None)
        mesh_vertadr = getattr(m, "mesh_vertadr", None)
        mesh_vertnum = getattr(m, "mesh_vertnum", None)
        mesh_faceadr = getattr(m, "mesh_faceadr", None)
        mesh_facenum = getattr(m, "mesh_facenum", None)

        for i in range(m.ngeom):
            try:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            except Exception:
                name = ""
            gtype = int(geom_type[i])
            size = np.array(geom_size[i])
            rgba = tuple(map(float, geom_rgba[i]))
            group = int(geom_group[i])

            if name and name in geom_materials:
                mat_name = geom_materials[name]
                if mat_name in material_colors:
                    rgba = material_colors[mat_name]

            if not name:
                mesh_name = ""
                try:
                    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                        dataid = int(geom_dataid[i]) if geom_dataid is not None else -1
                        if dataid >= 0:
                            mesh_name = (
                                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, dataid)
                                or ""
                            )
                except Exception:
                    mesh_name = ""

                if mesh_name:
                    inferred_mat = f"{mesh_name}_material"
                    if inferred_mat in material_colors:
                        rgba = material_colors[inferred_mat]
                    else:
                        for prefix in ("left_", "right_"):
                            if mesh_name.startswith(prefix):
                                stripped_name = mesh_name[len(prefix) :]
                                inferred_mat = f"{stripped_name}_material"
                                if inferred_mat in material_colors:
                                    rgba = material_colors[inferred_mat]
                                    break

            self._geom_groups[i] = group
            self._geom_base_rgba[i] = rgba

            try:
                if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE) or "floor" in name:
                    continue
            except Exception:
                pass

            mesh = None
            try:
                if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    mesh = trimesh.creation.icosphere(
                        radius=float(size[0]), subdivisions=3
                    )
                    color_rgba_255 = [
                        int(rgba[0] * 255),
                        int(rgba[1] * 255),
                        int(rgba[2] * 255),
                        int(rgba[3] * 255),
                    ]
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh, face_colors=color_rgba_255
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                    height = float(2.0 * size[1])
                    mesh = trimesh.creation.capsule(
                        radius=float(size[0]), height=height, count=[16, 16]
                    )
                    color_rgba_255 = [
                        int(rgba[0] * 255),
                        int(rgba[1] * 255),
                        int(rgba[2] * 255),
                        int(rgba[3] * 255),
                    ]
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh, face_colors=color_rgba_255
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    height = float(2.0 * size[1])
                    mesh = trimesh.creation.cylinder(
                        radius=float(size[0]), height=height, sections=24
                    )
                    color_rgba_255 = [
                        int(rgba[0] * 255),
                        int(rgba[1] * 255),
                        int(rgba[2] * 255),
                        int(rgba[3] * 255),
                    ]
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh, face_colors=color_rgba_255
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                    extents = 2.0 * size[:3]
                    mesh = trimesh.creation.box(extents=extents)
                    color_rgba_255 = [
                        int(rgba[0] * 255),
                        int(rgba[1] * 255),
                        int(rgba[2] * 255),
                        int(rgba[3] * 255),
                    ]
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh, face_colors=color_rgba_255
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                    sph = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
                    sph.apply_scale(2.0 * size[:3])
                    mesh = sph
                    color_rgba_255 = [
                        int(rgba[0] * 255),
                        int(rgba[1] * 255),
                        int(rgba[2] * 255),
                        int(rgba[3] * 255),
                    ]
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh, face_colors=color_rgba_255
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mid = int(geom_dataid[i]) if geom_dataid is not None else -1
                    built = False
                    if (
                        mid >= 0
                        and mesh_vert is not None
                        and mesh_vertadr is not None
                        and mesh_vertnum is not None
                        and mesh_face is not None
                        and mesh_faceadr is not None
                        and mesh_facenum is not None
                    ):
                        try:
                            vadr = int(mesh_vertadr[mid])
                            vnum = int(mesh_vertnum[mid])
                            fadr = int(mesh_faceadr[mid])
                            fnum = int(mesh_facenum[mid])
                            vert_arr = np.asarray(mesh_vert, dtype=np.float32)
                            face_arr = np.asarray(mesh_face, dtype=np.int32)
                            verts = vert_arr[vadr : vadr + vnum]
                            faces = face_arr[fadr : fadr + fnum]
                            mesh = trimesh.Trimesh(
                                vertices=verts, faces=faces, process=False
                            )
                            color_rgba_255 = [
                                int(rgba[0] * 255),
                                int(rgba[1] * 255),
                                int(rgba[2] * 255),
                                int(rgba[3] * 255),
                            ]
                            mesh.visual = trimesh.visual.ColorVisuals(
                                mesh, face_colors=color_rgba_255
                            )
                            built = True
                        except Exception:
                            built = False
                    if not built and mid >= 0:
                        try:
                            mm = m.mesh(mid)
                            v = getattr(mm, "vertex", None)
                            f = getattr(mm, "face", None)
                            if v is not None and f is not None:
                                verts = np.asarray(v, dtype=np.float32)
                                faces = np.asarray(f, dtype=np.int32)
                                if verts.ndim == 1:
                                    verts = verts.reshape(-1, 3)
                                if faces.ndim == 1:
                                    faces = faces.reshape(-1, 3)
                                mesh = trimesh.Trimesh(
                                    vertices=verts, faces=faces, process=False
                                )
                                color_rgba_255 = [
                                    int(rgba[0] * 255),
                                    int(rgba[1] * 255),
                                    int(rgba[2] * 255),
                                    int(rgba[3] * 255),
                                ]
                                mesh.visual = trimesh.visual.ColorVisuals(
                                    mesh, face_colors=color_rgba_255
                                )
                                built = True
                        except Exception:
                            built = False
                    if not built and mid >= 0:
                        try:
                            mesh_name = mujoco.mj_id2name(
                                m, mujoco.mjtObj.mjOBJ_MESH, mid
                            )
                        except Exception:
                            mesh_name = None
                        fpath = (
                            self._mesh_file_map.get(mesh_name) if mesh_name else None
                        )
                        if fpath and os.path.exists(fpath):
                            try:
                                mesh = trimesh.load(fpath, force="mesh")
                                try:
                                    sc = self._mesh_scale_map.get(mesh_name)
                                    if sc is not None:
                                        mesh.apply_scale(sc)
                                except Exception:
                                    pass
                                try:
                                    q = self._mesh_quat_map.get(mesh_name)
                                    if q is not None:
                                        rot = R.from_quat([q[1], q[2], q[3], q[0]])
                                        T = np.eye(4)
                                        T[:3, :3] = rot.as_matrix()
                                        mesh.apply_transform(T)
                                except Exception:
                                    pass
                                color_rgba_255 = [
                                    int(rgba[0] * 255),
                                    int(rgba[1] * 255),
                                    int(rgba[2] * 255),
                                    int(rgba[3] * 255),
                                ]
                                mesh.visual = trimesh.visual.ColorVisuals(
                                    mesh, face_colors=color_rgba_255
                                )
                                built = True
                            except Exception:
                                built = False
                        if not built and mesh_name:
                            try:
                                desc_dir = os.path.join(
                                    "toddlerbot", "descriptions", self.robot.name
                                )
                                alt = os.path.join(
                                    desc_dir, "assets", f"{mesh_name}.stl"
                                )
                                if os.path.exists(alt):
                                    mesh = trimesh.load(alt, force="mesh")
                                    try:
                                        sc = self._mesh_scale_map.get(mesh_name)
                                        if sc is not None:
                                            mesh.apply_scale(sc)
                                    except Exception:
                                        pass
                                    try:
                                        q = self._mesh_quat_map.get(mesh_name)
                                        if q is not None:
                                            rot = R.from_quat([q[1], q[2], q[3], q[0]])
                                            T = np.eye(4)
                                            T[:3, :3] = rot.as_matrix()
                                            mesh.apply_transform(T)
                                    except Exception:
                                        pass
                                    built = True
                            except Exception:
                                built = False
            except Exception:
                mesh = None

            if mesh is None:
                continue

            try:
                color_rgba_255 = [
                    int(rgba[0] * 255),
                    int(rgba[1] * 255),
                    int(rgba[2] * 255),
                    int(rgba[3] * 255),
                ]
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh, face_colors=color_rgba_255
                )
            except Exception as exc:
                print(
                    f"[Viser] Warning: Failed to apply color to geom {i} ({name}): {exc}",
                    flush=True,
                )

            path = f"/robot/{'collision' if group == 3 else 'visual'}/{i:04d}_{name}"
            handle = None
            try:
                handle = self.server.scene.add_mesh_trimesh(path, mesh)
            except Exception:
                try:
                    handle = self.server.scene.add_mesh_simple(
                        path,
                        vertices=np.asarray(mesh.vertices, dtype=float),
                        faces=np.asarray(mesh.faces, dtype=int),
                        color=(rgba[0], rgba[1], rgba[2]),
                    )
                except Exception as exc2:
                    print(
                        f"[Viser] Failed to add mesh for geom {i} ({name}): {exc2}",
                        flush=True,
                    )
                    handle = None

            if handle is not None:
                self._geom_handles[i] = handle
                try:
                    if hasattr(handle, "color"):
                        handle.color = (
                            float(rgba[0]),
                            float(rgba[1]),
                            float(rgba[2]),
                        )
                except Exception:
                    pass
                try:
                    if hasattr(handle, "rgba"):
                        handle.rgba = (
                            float(rgba[0]),
                            float(rgba[1]),
                            float(rgba[2]),
                            float(rgba[3]),
                        )
                except Exception:
                    pass

        try:
            gcounts: Dict[int, int] = {}
            for gi, grp in self._geom_groups.items():
                gcounts[grp] = gcounts.get(grp, 0) + 1
            built_by_group: Dict[int, int] = {}
            for gi, _h in self._geom_handles.items():
                grp = self._geom_groups.get(gi, -1)
                built_by_group[grp] = built_by_group.get(grp, 0) + 1
            color_samples: List[str] = []
            for gi in list(self._geom_handles.keys())[:5]:
                rgba = self._geom_base_rgba.get(gi, (0.5, 0.5, 0.5, 1.0))
                color_samples.append(
                    f"geom{gi}=({rgba[0]:.2f},{rgba[1]:.2f},{rgba[2]:.2f})"
                )
            print(
                f"[Viser] Built {len(self._geom_handles)} geom meshes; groups={gcounts}; built_by_group={built_by_group}",
                flush=True,
            )
            if color_samples:
                print(f"[Viser] Color samples: {', '.join(color_samples)}", flush=True)
        except Exception:
            pass

    def _apply_geom_visibility(self) -> None:
        """Update visibility of robot geometry handles."""
        show_collision = (
            bool(self.collision_geom_checked.value)
            if self.collision_geom_checked
            else False
        )
        try:
            show_all = bool(self.show_all_geoms.value) if self.show_all_geoms else False
        except Exception:
            show_all = True

        for i, handle in self._geom_handles.items():
            group = self._geom_groups.get(i, 0)
            base = self._geom_base_rgba.get(i, (0.7, 0.7, 0.7, 1.0))
            if show_all:
                visible = True
            else:
                visible = (group == 3) if show_collision else (group != 3)
            try:
                if hasattr(handle, "visible"):
                    handle.visible = visible
                elif hasattr(handle, "rgba"):
                    handle.rgba = (
                        float(base[0]),
                        float(base[1]),
                        float(base[2]),
                        1.0 if visible else 0.05,
                    )
                elif hasattr(handle, "color"):
                    handle.color = (
                        float(base[0]),
                        float(base[1]),
                        float(base[2]),
                    )
            except Exception:
                continue

    def _start_scene_updater(self) -> None:
        """Launch (or relaunch) the background scene updater thread."""
        if not (self._geom_handles or self._scene_handles or self._com_sphere):
            return
        if self._scene_updater and self._scene_updater.is_alive():
            return

        self._scene_updater = threading.Thread(
            target=self._update_scene_loop,
            name="ViserSceneUpdater",
            daemon=True,
        )
        self._scene_updater.start()

    def _update_scene_loop(self) -> None:
        """Periodically push scene poses and apply visibility toggles."""
        while True:
            try:
                if self._com_sphere is not None:
                    with self.worker_lock:
                        com_pos = self.sim.data.subtree_com[0].copy()
                    try:
                        self._com_sphere.position = tuple(map(float, com_pos))
                    except Exception:
                        pass

                if self._geom_handles:
                    with self.worker_lock:
                        xpos = np.array(self.sim.data.geom_xpos, dtype=np.float32)
                        use_quat = True
                        try:
                            xquat_wxyz = np.array(
                                self.sim.data.geom_xquat, dtype=np.float32
                            )
                        except Exception:
                            use_quat = False
                            xmat = np.array(
                                self.sim.data.geom_xmat, dtype=np.float32
                            ).reshape(-1, 3, 3)
                    for index, handle in list(self._geom_handles.items()):
                        try:
                            position = tuple(map(float, xpos[index]))
                            if use_quat:
                                qw, qx, qy, qz = map(float, xquat_wxyz[index])
                            else:
                                quat_xyzw = R.from_matrix(xmat[index]).as_quat()
                                qx, qy, qz, qw = map(float, quat_xyzw)
                            handle.position = position
                            if hasattr(handle, "wxyz"):
                                handle.wxyz = (
                                    float(qw),
                                    float(qx),
                                    float(qy),
                                    float(qz),
                                )
                            elif hasattr(handle, "xyzw"):
                                handle.xyzw = (
                                    float(qx),
                                    float(qy),
                                    float(qz),
                                    float(qw),
                                )
                        except Exception:
                            continue
                    self._apply_geom_visibility()
                elif self._scene_handles:
                    with self.worker_lock:
                        body_pos = np.array(self.sim.data.body_xpos, dtype=np.float32)
                        body_mat = np.array(
                            self.sim.data.body_xmat, dtype=np.float32
                        ).reshape(-1, 3, 3)
                    for body_index in range(
                        min(len(body_pos), len(self._scene_handles))
                    ):
                        handle = self._scene_handles.get(f"body_{body_index}")
                        if handle is None:
                            continue
                        try:
                            handle.position = tuple(map(float, body_pos[body_index]))
                            if hasattr(handle, "wxyz"):
                                rot = R.from_matrix(body_mat[body_index])
                                qw, qx, qy, qz = rot.as_quat(scalar_first=True)
                                handle.wxyz = (
                                    float(qw),
                                    float(qx),
                                    float(qy),
                                    float(qz),
                                )
                        except Exception:
                            continue
            except Exception:
                pass
            time.sleep(0.05)

    def _test_trajectory(self) -> None:
        if len(self.sequence_list) < 2:
            print("[Viser] Action traj: need at least 2 sequence entries.", flush=True)
            return
        start_idx = self.selected_sequence or 0
        action_list: List[np.ndarray] = []
        qpos_list: List[np.ndarray] = []
        times: List[float] = []
        for name, t in self.sequence_list:
            for kf in self.keyframes:
                if kf.name == name and kf.qpos is not None:
                    action_list.append(kf.motor_pos)
                    qpos_list.append(kf.qpos)
                    times.append(t)
                    break
        if len(times) < 2:
            print(
                f"[Viser] Action traj: collected fewer than 2 timestamps. times={times}",
                flush=True,
            )
            return
        times_arr = np.array(times)
        if np.any(np.diff(times_arr) <= 0):
            print(
                f"[Viser] Action traj: times not strictly increasing: {times}",
                flush=True,
            )
            return
        qpos_start = qpos_list[start_idx]
        enabled = bool(self.physics_enabled.value) if self.physics_enabled else True
        action_arr = np.array(action_list)
        times_arr = times_arr - times_arr[0]
        self.traj_times = list(np.arange(0, times_arr[-1], self.dt))
        self.action_traj = []
        print(
            f"[Viser] Action traj: start_idx={start_idx}, dt={self.dt}, steps={len(self.traj_times)}, physics={enabled}",
            flush=True,
        )
        for t in self.traj_times:
            if t < times_arr[-1]:
                motor_pos = interpolate_action(t, times_arr, action_arr)
            else:
                motor_pos = action_arr[-1]
            self.action_traj.append(motor_pos)
        traj_start = int(np.searchsorted(self.traj_times, times_arr[start_idx]))
        self.is_qpos_traj = False
        self.is_relative_frame = (
            bool(self.relative_frame_checked.value)
            if self.relative_frame_checked
            else True
        )
        self.worker.request_trajectory_test(
            qpos_start,
            self.action_traj[traj_start:],
            self.dt,
            enabled,
            is_qpos_traj=False,
            is_relative_frame=self.is_relative_frame,
        )

    def _test_qpos_trajectory(self) -> None:
        if len(self.sequence_list) < 2:
            print("[Viser] Qpos traj: need at least 2 sequence entries.", flush=True)
            return
        start_idx = self.selected_sequence or 0
        qpos_list: List[np.ndarray] = []
        times: List[float] = []
        for name, t in self.sequence_list:
            for kf in self.keyframes:
                if kf.name == name and kf.qpos is not None:
                    qpos_list.append(kf.qpos)
                    times.append(t)
                    break
        if len(times) < 2:
            print(
                f"[Viser] Qpos traj: collected fewer than 2 timestamps. times={times}",
                flush=True,
            )
            return
        times_arr = np.array(times)
        if np.any(np.diff(times_arr) <= 0):
            print(
                f"[Viser] Qpos traj: times not strictly increasing: {times}",
                flush=True,
            )
            return
        times_arr = times_arr - times_arr[0]
        self.traj_times = list(np.arange(0, times_arr[-1], self.dt))
        qpos_arr = np.array(qpos_list)
        qpos_traj: List[np.ndarray] = []
        traj_start = int(np.searchsorted(self.traj_times, times_arr[start_idx]))
        print(
            f"[Viser] Qpos traj: start_idx={start_idx}, traj_start={traj_start}, steps={len(self.traj_times)}",
            flush=True,
        )
        for t in self.traj_times:
            if t < times_arr[-1]:
                qpos_t = interpolate_action(t, times_arr, qpos_arr)
            else:
                qpos_t = qpos_arr[-1]
            qpos_traj.append(qpos_t)
        self.is_qpos_traj = True
        self.is_relative_frame = (
            bool(self.relative_frame_checked.value)
            if self.relative_frame_checked
            else True
        )
        print(
            f"[Viser] Qpos traj: request worker run. physics=False, is_rel={self.is_relative_frame}",
            flush=True,
        )
        self.worker.request_trajectory_test(
            qpos_list[start_idx],
            qpos_traj[traj_start:],
            self.dt,
            physics_enabled=False,
            is_qpos_traj=True,
            is_relative_frame=self.is_relative_frame,
        )

    def _save_data(self) -> None:
        result_dict: Dict[str, object] = {}
        saved_keyframes = [asdict(kf) for kf in self.keyframes]

        result_dict["time"] = np.array(self.traj_times)
        result_dict["qpos"] = np.array(self.qpos_replay)
        result_dict["body_pos"] = np.array(self.body_pos_replay)
        result_dict["body_quat"] = np.array(self.body_quat_replay)
        result_dict["body_lin_vel"] = np.array(self.body_lin_vel_replay)
        result_dict["body_ang_vel"] = np.array(self.body_ang_vel_replay)
        result_dict["site_pos"] = np.array(self.site_pos_replay)
        result_dict["site_quat"] = np.array(self.site_quat_replay)
        result_dict["action"] = (
            None if self.is_qpos_traj else np.array(self.action_traj)
        )
        result_dict["keyframes"] = saved_keyframes
        result_dict["timed_sequence"] = self.sequence_list
        result_dict["is_robot_relative_frame"] = self.is_relative_frame

        time_str = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.result_dir, f"{self.task_name}_{time_str}.lz4")
        joblib.dump(result_dict, result_path, compress="lz4")

        motion_name = (
            self.motion_name_input.value if self.motion_name_input else self.task_name
        )
        motion_file_path = os.path.join("motion", f"{motion_name}.lz4")
        if os.path.exists(motion_file_path):
            print(f"⚠️  Overwriting existing file: motion/{motion_name}.lz4")
        joblib.dump(result_dict, motion_file_path, compress="lz4")
        print(f"✓ Saved: motion/{motion_name}.lz4")

    def _load_data(self) -> None:
        self.keyframes.clear()
        self.sequence_list.clear()
        self._refresh_keyframes_table()
        self._refresh_sequence_table()

        if not self.data_path or not os.path.exists(self.data_path):
            self.keyframes.append(
                Keyframe(
                    name="default",
                    motor_pos=np.array(
                        list(self.robot.default_motor_angles.values()), dtype=np.float32
                    ),
                    joint_pos=np.array(
                        list(self.robot.default_joint_angles.values()), dtype=np.float32
                    ),
                    qpos=self.sim.home_qpos.copy(),
                )
            )
            self._refresh_keyframes_table()
            self._load_keyframe_to_ui(0)
            return

        try:
            data = joblib.load(self.data_path)
        except Exception:
            data = None

        if data is None:
            self.keyframes.append(
                Keyframe(
                    name="default",
                    motor_pos=np.array(
                        list(self.robot.default_motor_angles.values()), dtype=np.float32
                    ),
                    joint_pos=np.array(
                        list(self.robot.default_joint_angles.values()), dtype=np.float32
                    ),
                    qpos=self.sim.home_qpos.copy(),
                )
            )
            self._refresh_keyframes_table()
            self._load_keyframe_to_ui(0)
            return

        keyframes_data = data.get("keyframes") if isinstance(data, dict) else None
        if keyframes_data is not None:
            loaded_keyframes: List[Keyframe] = []
            for k in keyframes_data:
                loaded_keyframes.append(
                    Keyframe(
                        name=k["name"],
                        motor_pos=np.array(k["motor_pos"], dtype=np.float32),
                        joint_pos=np.array(k["joint_pos"], dtype=np.float32)
                        if k.get("joint_pos") is not None
                        else None,
                        qpos=np.array(k["qpos"], dtype=np.float32)
                        if k.get("qpos") is not None
                        else None,
                    )
                )
            if loaded_keyframes and len(loaded_keyframes[0].motor_pos) != self.robot.nu:
                raise ValueError("Loaded data is incompatible with the current robot.")
            self.keyframes.extend(loaded_keyframes)
            sequence_entries = data.get("timed_sequence", [])
            self.sequence_list = [
                (n.replace(" ", "_"), float(t)) for (n, t) in sequence_entries
            ]
            self.traj_times = list(map(float, data.get("time", [])))
            self.action_traj = data.get("action", [])
            self.qpos_replay = list(data.get("qpos", []))
            self._refresh_keyframes_table()
            self._refresh_sequence_table()
            if self.keyframes:
                self._load_keyframe_to_ui(0)
            return

        # Legacy format (list of Keyframe objects)
        legacy_keyframes = data
        if legacy_keyframes and len(legacy_keyframes[0].motor_pos) != self.robot.nu:
            raise ValueError("Loaded data is incompatible with the current robot.")
        self.keyframes.extend(legacy_keyframes)
        self._refresh_keyframes_table()
        if self.keyframes:
            self._load_keyframe_to_ui(0)
        self._refresh_sequence_table()

    def _add_keyframe(self) -> None:
        if self.selected_keyframe is None:
            base_name = "keyframe"
            new_name = self._generate_unique_name(base_name)
            new_kf = Keyframe(
                name=new_name,
                motor_pos=np.array(
                    list(self.robot.default_motor_angles.values()), dtype=np.float32
                ),
                joint_pos=np.array(
                    list(self.robot.default_joint_angles.values()), dtype=np.float32
                ),
                qpos=self.sim.home_qpos.copy(),
            )
        else:
            kf = self.keyframes[self.selected_keyframe]
            base_name = self._base_keyframe_name(kf.name)
            new_name = self._generate_unique_name(base_name)
            new_kf = Keyframe(
                name=new_name,
                motor_pos=kf.motor_pos.copy(),
                joint_pos=kf.joint_pos.copy(),
                qpos=kf.qpos.copy(),
            )
        self.keyframes.append(new_kf)
        self.selected_keyframe = len(self.keyframes) - 1
        self._refresh_keyframes_table()
        self._load_keyframe_to_ui(self.selected_keyframe)

    def _remove_keyframe(self) -> None:
        if self.selected_keyframe is None:
            return
        name_to_remove = self.keyframes[self.selected_keyframe].name
        self.keyframes.pop(self.selected_keyframe)
        self.sequence_list = [
            (n, t) for (n, t) in self.sequence_list if n != name_to_remove
        ]
        self.selected_keyframe = None
        self._refresh_keyframes_table()
        self._refresh_sequence_table()

    def _duplicate_selected_keyframe(self) -> None:
        if self.selected_keyframe is None:
            return
        kf = self.keyframes[self.selected_keyframe]
        base_name = self._base_keyframe_name(kf.name)
        new_name = self._generate_unique_name(base_name)
        new_kf = Keyframe(
            name=new_name,
            motor_pos=kf.motor_pos.copy(),
            joint_pos=kf.joint_pos.copy(),
            qpos=kf.qpos.copy(),
        )
        self.keyframes.append(new_kf)
        self._refresh_keyframes_table()

    def _move_keyframe(self, direction: int) -> None:
        if self.selected_keyframe is None:
            return
        current_idx = self.selected_keyframe
        new_idx = current_idx + direction
        if 0 <= new_idx < len(self.keyframes):
            self.keyframes[current_idx], self.keyframes[new_idx] = (
                self.keyframes[new_idx],
                self.keyframes[current_idx],
            )
            self.selected_keyframe = new_idx
            self._refresh_keyframes_table()

    def _load_keyframe_to_ui(self, idx: int) -> None:
        kf = self.keyframes[idx]
        self.selected_keyframe = idx
        if kf.qpos is not None:
            self.worker.update_qpos(kf.qpos)
        if kf.joint_pos is not None:
            for jname, val in zip(self.robot.joint_ordering, kf.joint_pos):
                slider = self.slider_widgets.get(jname)
                if slider is not None:
                    self._set_handle_value(slider, float(val))
        if self.keyframe_index_input is not None:
            self._set_handle_value(self.keyframe_index_input, str(idx))
        if self.keyframe_name_input is not None:
            self._set_handle_value(self.keyframe_name_input, kf.name)
        self._refresh_sequence_table()

    def _add_to_sequence(self) -> None:
        if self.selected_keyframe is None:
            return
        kf = self.keyframes[self.selected_keyframe]
        last_t = self.sequence_list[-1][1] if self.sequence_list else 0.0
        self.sequence_list.append((kf.name, last_t + 1.0))
        self._refresh_sequence_table()

    def _remove_from_sequence(self) -> None:
        if self.selected_sequence is None:
            return
        self.sequence_list.pop(self.selected_sequence)
        self.selected_sequence = None
        self._refresh_sequence_table()

    def _edit_sequence_time(self, row: int, new_time: float) -> None:
        if not (0 <= row < len(self.sequence_list)):
            return
        name, old_time = self.sequence_list[row]
        delta = float(new_time) - float(old_time)
        for i in range(row, len(self.sequence_list)):
            n, t = self.sequence_list[i]
            self.sequence_list[i] = (n, float(t) + delta)
        self._refresh_sequence_table()

    def _refresh_keyframes_table(self) -> None:
        self._refresh_keyframes_summary()

    def _refresh_keyframes_summary(self) -> None:
        if self.keyframes_summary is None:
            return
        if not self.keyframes:
            self.keyframes_summary.content = (
                '<div style="font-size:0.875em; margin-left:0.75em">No keyframes</div>'
            )
            return
        lines = [f"{i}: {kf.name}" for i, kf in enumerate(self.keyframes)]
        content = "<br/>".join(lines)
        visible_rows = min(len(lines), 3) or 1
        line_height_em = 1.3
        max_height_em = line_height_em * visible_rows
        wrapped = (
            (
                f'<div style="line-height:{line_height_em}em; '
                f"max-height:{max_height_em}em; overflow-y:auto; "
                f'font-size:0.875em; margin-left:0.75em">'
                f"{content}</div>"
            )
            if content
            else "No keyframes"
        )
        self.keyframes_summary.content = wrapped

    def _refresh_sequence_table(self) -> None:
        self._refresh_sequence_summary()

    def _refresh_sequence_summary(self) -> None:
        if self.sequence_summary is None:
            return
        if not self.sequence_list:
            self.sequence_summary.content = (
                '<div style="font-size:0.875em; margin-left:0.75em">No sequence</div>'
            )
            return
        lines = [
            f"{i}: {n.replace(' ', '_')} &nbsp;&nbsp; t={t}"
            for i, (n, t) in enumerate(self.sequence_list)
        ]
        content = "<br/>".join(lines)
        visible_rows = min(len(lines), 3) or 1
        line_height_em = 1.3
        max_height_em = line_height_em * visible_rows
        wrapped = (
            (
                f'<div style="line-height:{line_height_em}em; '
                f"max-height:{max_height_em}em; overflow-y:auto; "
                f'font-size:0.875em; margin-left:0.75em">'
                f"{content}</div>"
            )
            if content
            else "No sequence"
        )
        self.sequence_summary.content = wrapped

    def _test_keyframe(self) -> None:
        if self.selected_keyframe is None:
            return
        kf = self.keyframes[self.selected_keyframe]
        if kf.qpos is None:
            return
        self.worker.request_keyframe_test(kf, self.dt)

    def _set_handle_value(self, handle, value: object) -> None:
        hid = id(handle)
        self._updating_handles.add(hid)
        try:
            rounded_value = value
            if hasattr(handle, "precision"):
                try:
                    prec = int(getattr(handle, "precision"))
                except Exception:
                    prec = None
                if prec is not None and isinstance(value, (int, float)):
                    rounded_value = round(float(value), prec)
            handle.value = rounded_value
        finally:
            self._updating_handles.discard(hid)

    def _update_joint_pos(self, joint_name: str, value: float) -> Dict[str, float]:
        updates: Dict[str, float] = {joint_name: float(value)}
        mirror = bool(self.mirror_checked.value) if self.mirror_checked else True
        rev_mirror = (
            bool(self.rev_mirror_checked.value) if self.rev_mirror_checked else False
        )
        if mirror or rev_mirror:
            if ("left" in joint_name) or ("right" in joint_name):
                mirrored_joint_name = (
                    joint_name.replace("left", "right")
                    if "left" in joint_name
                    else joint_name.replace("right", "left")
                )
                mirror_sign = (
                    self.mirror_joint_signs.get(joint_name, 1.0)
                    if "left" in joint_name
                    else self.mirror_joint_signs.get(mirrored_joint_name, 1.0)
                )
                updates[mirrored_joint_name] = (
                    1 if mirror else 0
                ) * value * mirror_sign - (1 if rev_mirror else 0) * value * mirror_sign
        self.worker.update_joint_angles(updates)
        return updates

    def _on_state(
        self,
        motor_pos: np.ndarray,
        joint_pos: np.ndarray,
        qpos: np.ndarray,
    ) -> None:
        if self.selected_keyframe is None:
            return
        idx = self.selected_keyframe
        self.keyframes[idx].motor_pos = motor_pos.copy()
        self.keyframes[idx].joint_pos = joint_pos.copy()
        self.keyframes[idx].qpos = qpos.copy()
        for jname, val in zip(self.robot.joint_ordering, joint_pos):
            slider = self.slider_widgets.get(jname)
            if slider is not None:
                self._set_handle_value(slider, float(val))

    def _on_traj(
        self,
        qpos_replay: List[np.ndarray],
        body_pos_replay: List[np.ndarray],
        body_quat_replay: List[np.ndarray],
        body_lin_vel_replay: List[np.ndarray],
        body_ang_vel_replay: List[np.ndarray],
        site_pos_replay: List[np.ndarray],
        site_quat_replay: List[np.ndarray],
    ) -> None:
        self.qpos_replay = qpos_replay
        self.body_pos_replay = body_pos_replay
        self.body_quat_replay = body_quat_replay
        self.body_lin_vel_replay = body_lin_vel_replay
        self.body_ang_vel_replay = body_ang_vel_replay
        self.site_pos_replay = site_pos_replay
        self.site_quat_replay = site_quat_replay

    def _format_joint_label(self, joint_name: str) -> str:
        tokens = joint_name.split("_")
        if not tokens:
            return joint_name
        prefix_map = {
            "left": "L",
            "right": "R",
        }
        formatted_tokens: List[str] = []
        for idx, tok in enumerate(tokens):
            lower_tok = tok.lower()
            if idx == 0 and lower_tok in prefix_map:
                formatted_tokens.append(prefix_map[lower_tok])
            else:
                formatted_tokens.append(lower_tok.capitalize())
        return " ".join(formatted_tokens)

    def _build_foot_hand_panel(self) -> None:
        with self.server.gui.add_folder("👣 Foot & Hand Poses"):
            save_l_btn = self.server.gui.add_button("Save L Foot")

            @save_l_btn.on_click
            def _(_e: GuiEvent) -> None:
                with self.worker.lock:
                    self.saved_left_foot_pose = self.sim.get_site_transform(
                        "left_foot_center"
                    )

            apply_l_btn = self.server.gui.add_button("Apply L Foot")

            @apply_l_btn.on_click
            def _(_e: GuiEvent) -> None:
                if self.saved_left_foot_pose is not None:
                    self._align_foot_to_pose(
                        "left_foot_center", self.saved_left_foot_pose
                    )
                else:
                    print("[Viser] No saved left foot pose to apply.", flush=True)

            save_r_btn = self.server.gui.add_button("Save R Foot")

            @save_r_btn.on_click
            def _(_e: GuiEvent) -> None:
                with self.worker.lock:
                    self.saved_right_foot_pose = self.sim.get_site_transform(
                        "right_foot_center"
                    )

            apply_r_btn = self.server.gui.add_button("Apply R Foot")

            @apply_r_btn.on_click
            def _(_e: GuiEvent) -> None:
                if self.saved_right_foot_pose is not None:
                    self._align_foot_to_pose(
                        "right_foot_center", self.saved_right_foot_pose
                    )
                else:
                    print("[Viser] No saved right foot pose to apply.", flush=True)

            mark_hands_btn = self.server.gui.add_button("Mark Hands")

            @mark_hands_btn.on_click
            def _(_e: GuiEvent) -> None:
                with self.worker.lock:
                    left_pos = self.sim.get_site_transform("left_hand_center")[:3, 3]
                    right_pos = self.sim.get_site_transform("right_hand_center")[:3, 3]
                try:
                    self.server.scene.add_icosphere(
                        "/markers/left_hand",
                        radius=0.01,
                        position=tuple(map(float, left_pos)),
                        color=(0.9, 0.1, 0.1),
                    )
                    self.server.scene.add_icosphere(
                        "/markers/right_hand",
                        radius=0.01,
                        position=tuple(map(float, right_pos)),
                        color=(0.9, 0.1, 0.1),
                    )
                except Exception:
                    pass

    def _align_foot_to_pose(
        self, foot_site_name: str, target_pose: Optional[np.ndarray]
    ) -> None:
        """Align the torso so that `foot_site_name` matches `target_pose`."""
        if target_pose is None:
            print(f"[Viser] No saved pose for {foot_site_name}.", flush=True)
            return
        with self.worker_lock:
            torso_t_curr = self.sim.get_body_transform("torso")
            foot_t_curr = self.sim.get_site_transform(foot_site_name)
            aligned_torso_t = target_pose @ np.linalg.inv(foot_t_curr) @ torso_t_curr
            self.sim.data.qpos[:3] = aligned_torso_t[:3, 3]
            self.sim.data.qpos[3:7] = R.from_matrix(aligned_torso_t[:3, :3]).as_quat(
                scalar_first=True
            )
            self.sim.forward()
        print(f"[Viser] {foot_site_name} aligned to saved pose.", flush=True)

    def _build_settings_panel(self) -> None:
        with self.server.gui.add_folder("⚙️ Settings"):
            self.mirror_checked = self.server.gui.add_checkbox("Mirror", True)
            self.rev_mirror_checked = self.server.gui.add_checkbox("Rev. Mirror", False)
            self.physics_enabled = self.server.gui.add_checkbox("Enable Physics", True)
            self.relative_frame_checked = self.server.gui.add_checkbox(
                "Save in Robot Frame",
                True,
            )
            self.collision_geom_checked = self.server.gui.add_checkbox(
                "Show Collision Geoms",
                False,
            )
            self.show_all_geoms = self.server.gui.add_checkbox(
                "Show All Geoms",
                False,
            )

        assert self.mirror_checked is not None
        assert self.rev_mirror_checked is not None
        assert self.collision_geom_checked is not None
        assert self.show_all_geoms is not None

        @self.mirror_checked.on_update
        def _(ev: GuiEvent) -> None:
            if id(ev.target) in self._updating_handles:
                return
            if bool(self.mirror_checked.value) and bool(self.rev_mirror_checked.value):
                self._set_handle_value(self.rev_mirror_checked, False)

        @self.rev_mirror_checked.on_update
        def _(ev: GuiEvent) -> None:
            if id(ev.target) in self._updating_handles:
                return
            if bool(self.rev_mirror_checked.value) and bool(self.mirror_checked.value):
                self._set_handle_value(self.mirror_checked, False)

        @self.collision_geom_checked.on_update
        def _(_event: GuiEvent) -> None:
            self._apply_geom_visibility()

        @self.show_all_geoms.on_update
        def _(_event: GuiEvent) -> None:
            self._apply_geom_visibility()

    def _base_keyframe_name(self, name: str) -> str:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return name

    def _generate_unique_name(self, base: str) -> str:
        existing_names = {kf.name for kf in self.keyframes}
        if base not in existing_names:
            return base
        suffix = 1
        while f"{base}_{suffix}" in existing_names:
            suffix += 1
        return f"{base}_{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo Keyframe Editor (Viser)")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Must match descriptions dir.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="push_up",
        help="The name of the task (unused for geometry-only milestone).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Scene to load (e.g., 'scene', 'scene_climb_up_box').",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "If same as task, a copy from motion/<task>.lz4 is created under results/. "
            "Otherwise treated as a subfolder inside results/."
        ),
    )
    args = parser.parse_args()

    robot = Robot(args.robot)
    if len(args.scene) > 0:
        xml_path = os.path.join(
            "toddlerbot", "descriptions", robot.name, args.scene + ".xml"
        )
    else:
        xml_path = ""
    sim = MuJoCoSim(robot, xml_path=xml_path, vis_type="none")

    actual_xml_path = (
        xml_path
        if xml_path
        else os.path.join("toddlerbot", "descriptions", robot.name, "scene.xml")
    )

    ViserKeyframeEditor(
        sim,
        robot,
        task_name=args.task,
        run_name=args.run_name,
        xml_path=actual_xml_path,
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
