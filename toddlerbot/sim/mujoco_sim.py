import os
import time
from typing import Dict

import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.motor_control import MotorController, PositionController
from toddlerbot.sim.mujoco_utils import MuJoCoRenderer, MuJoCoViewer
from toddlerbot.sim.robot import Robot
from toddlerbot.sim.terrain.get_elevation import (
    get_elevation_map,
    update_elevation_visualization,
)


class MuJoCoSim(BaseSim):
    """MuJoCo physics simulation environment for robot control and testing.

    This class provides a MuJoCo-based simulation environment for robotics
    applications, supporting both torque and position control modes. It handles
    robot model loading, physics stepping, visualization, and observation collection.

    The simulation supports various robot configurations including fixed-base
    and free-floating robots, with customizable control frequencies and
    visualization options.
    """

    def __init__(
        self,
        robot: Robot,
        n_frames: int = 20,
        dt: float = 0.001,
        fixed_base: bool = False,
        xml_path: str = "",
        model: mujoco.MjModel = None,
        vis_type: str = "",
        controller_type: str = "torque",
    ):
        """Initialize the MuJoCo simulation environment for robot control.

        Args:
            robot: The robot configuration object containing joint and motor information.
            n_frames: Number of simulation frames per control step. Higher values provide
                more stable control but slower execution. Defaults to 20.
            dt: Physics simulation timestep in seconds. Defaults to 0.001.
            fixed_base: If True, the robot base is fixed in space. If False, the robot
                can move freely. Defaults to False.
            xml_path: Path to the MuJoCo XML model file. If empty, uses the robot's
                default model. Defaults to "".
            vis_type: Visualization mode - 'render' for off-screen rendering,
                'view' for interactive viewer, or '' for no visualization. Defaults to "".
            controller_type: Motor control type, either 'torque' for torque control
                or 'position' for position control. Defaults to "torque".
        """
        super().__init__("mujoco")

        self.robot = robot
        self.n_frames = n_frames
        self.dt = dt
        self.control_dt = n_frames * dt
        self.fixed_base = fixed_base

        if len(xml_path) == 0:
            description_dir = os.path.join("toddlerbot", "descriptions")
            xml_suffix = ""
            if controller_type != "torque":
                xml_suffix += "_pos"
            if fixed_base:
                xml_suffix += "_fixed"
            xml_path = os.path.join(
                description_dir, robot.name, f"scene{xml_suffix}.xml"
            )

        if model is not None:
            self.model = model
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = self.dt
        # self.model.pair_friction[:, :2] = 0.5
        # self.model.opt.gravity[2] = -1.0

        self.motor_indices = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.joint_indices = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        if not self.fixed_base:
            self.motor_indices -= 1
            self.joint_indices -= 1

        self.actuator_indices = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.q_start_idx = 0 if self.fixed_base else 7
        self.qd_start_idx = 0 if self.fixed_base else 6

        self.target_motor_pos = np.zeros(self.model.nu, dtype=np.float32)

        try:
            self.home_qpos = np.array(
                self.model.keyframe("home").qpos, dtype=np.float32
            )
        except KeyError:
            self.home_qpos = None
            print("No keyframe named 'home' found in the model.")

        if controller_type == "torque":
            self.controller = MotorController(robot)
        else:
            self.controller = PositionController()

        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model)
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(robot, self.model, self.data)
        else:
            self.visualizer = None

        self.ee_markers = []  # List of (pos, color) tuples

        # Elevation visualization attributes (set by external code like run_multiple_policy.py)
        self.elevation_info = None
        self.visualize_elevation = False
        self.elevation_map_size = 0.5
        self.elevation_map_resolution = 0.02
        self.elevation_map_forward_offset = 0.25

        self.depth_renderer = None  # Depth renderer for capturing depth images
        self.rgb_renderer = None  # RGB renderer for capturing RGB images

    def add_ee_marker(self, pos, color):
        if len(self.ee_markers) > 1:
            self.ee_markers.clear()

        self.ee_markers.append((pos, color))

    def get_body_transform(self, body_name: str):
        """Computes the transformation matrix for a specified body.

        Args:
            body_name (str): The name of the body for which to compute the transformation matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the position and orientation of the body.
        """
        transformation = np.eye(4)
        body_pos = self.data.body(body_name).xpos.copy()
        body_mat = self.data.body(body_name).xmat.reshape(3, 3).copy()
        transformation[:3, :3] = body_mat
        transformation[:3, 3] = body_pos
        return transformation

    def get_site_transform(self, site_name: str):
        """Retrieves the transformation matrix for a specified site.

        This method constructs a 4x4 transformation matrix for the given site name,
        using the site's position and orientation matrix. The transformation matrix
        is composed of a 3x3 rotation matrix and a 3x1 translation vector.

        Args:
            site_name (str): The name of the site for which to retrieve the transformation matrix.

        Returns:
            numpy.ndarray: A 4x4 transformation matrix representing the site's position and orientation.
        """
        transformation = np.eye(4)
        site_pos = self.data.site(site_name).xpos.copy()
        site_mat = self.data.site(site_name).xmat.reshape(3, 3).copy()
        transformation[:3, :3] = site_mat
        transformation[:3, 3] = site_pos
        return transformation

    def get_motor_angles(
        self, type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current angles of the robot's motors.

        Args:
            type (str): The format in which to return the motor angles.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The motor angles in the specified format.
            If "dict", returns a dictionary with motor names as keys and angles as values.
            If "array", returns a NumPy array of motor angles.
        """
        motor_angles: Dict[str, float] = {}
        for name in self.robot.motor_ordering:
            motor_angles[name] = self.data.joint(name).qpos.item()

        if type == "array":
            motor_pos_arr = np.array(list(motor_angles.values()), dtype=np.float32)
            return motor_pos_arr
        else:
            return motor_angles

    def get_motor_velocities(
        self, type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current velocities of the robot's motors.

        Args:
            type (str): The format in which to return the motor velocities.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The motor velocities in the specified format.
            If "dict", returns a dictionary with motor names as keys and velocities as values.
            If "array", returns a NumPy array of motor velocities.
        """
        motor_velocities: Dict[str, float] = {}
        for name in self.robot.motor_ordering:
            motor_velocities[name] = self.data.joint(name).qvel.item()

        if type == "array":
            motor_vel_arr = np.array(list(motor_velocities.values()), dtype=np.float32)
            return motor_vel_arr
        else:
            return motor_velocities

    def get_joint_velocities(
        self, type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current velocities of the robot's passive joints.

        Args:
            type (str): The format in which to return the joint velocities.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The joint velocities in the specified format.
            If "dict", returns a dictionary with joint names as keys and velocities as values.
            If "array", returns a NumPy array of joint velocities.
        """
        joint_velocities: Dict[str, float] = {}
        for name in self.robot.joint_ordering:
            joint_velocities[name] = self.data.joint(name).qvel.item()

        if type == "array":
            joint_vel_arr = np.array(list(joint_velocities.values()), dtype=np.float32)
            return joint_vel_arr
        else:
            return joint_velocities

    def get_joint_angles(
        self, type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current joint angles of the robot.

        Args:
            type (str): The format in which to return the joint angles.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The joint angles of the robot.
                Returns a dictionary with joint names as keys and angles as values if
                `type` is "dict". Returns a NumPy array of joint angles if `type` is "array".
        """
        joint_angles: Dict[str, float] = {}
        for name in self.robot.joint_ordering:
            joint_angles[name] = self.data.joint(name).qpos.item()

        if type == "array":
            joint_pos_arr = np.array(list(joint_angles.values()), dtype=np.float32)
            return joint_pos_arr
        else:
            return joint_angles

    def get_observation(self) -> Obs:
        """Retrieves the current observation of the robot's state, including motor and joint states, and torso dynamics.

        Returns:
            Obs: An observation object.
        """
        motor_pos_arr = self.data.qpos[self.q_start_idx + self.motor_indices].copy()
        motor_vel_arr = self.data.qvel[self.qd_start_idx + self.motor_indices].copy()
        motor_acc_arr = self.data.qacc[self.qd_start_idx + self.motor_indices].copy()
        motor_tor_arr = self.data.actuator_force[self.actuator_indices].copy()
        joint_pos_arr = self.data.qpos[self.q_start_idx + self.joint_indices].copy()
        joint_vel_arr = self.data.qvel[self.qd_start_idx + self.joint_indices].copy()

        if self.fixed_base:
            torso_lin_vel = np.zeros(3, dtype=np.float32)
            torso_ang_vel = np.zeros(3, dtype=np.float32)
            torso_pos = np.zeros(3, dtype=np.float32)
            torso_rot = R.identity()
        else:
            lin_vel_global = np.array(
                self.data.body("torso").cvel[3:],
                dtype=np.float32,
                copy=True,
            )
            ang_vel_global = np.array(
                self.data.body("torso").cvel[:3],
                dtype=np.float32,
                copy=True,
            )
            torso_pos = np.array(
                self.data.body("torso").xpos,
                dtype=np.float32,
                copy=True,
            )
            torso_quat = np.array(
                self.data.body("torso").xquat,
                dtype=np.float32,
                copy=True,
            )
            if np.linalg.norm(torso_quat) == 0:
                torso_quat = np.array([1, 0, 0, 0], dtype=np.float32)

            # Create rotation object from torso quaternion
            torso_rot = R.from_quat(torso_quat, scalar_first=True)
            torso_rot_inv = torso_rot.inv()

            # Rotate global velocities into torso frame
            torso_lin_vel = torso_rot_inv.apply(lin_vel_global)
            torso_ang_vel = torso_rot_inv.apply(ang_vel_global)

        obs = Obs(
            time=time.monotonic(),
            motor_pos=motor_pos_arr,
            motor_vel=motor_vel_arr,
            motor_acc=motor_acc_arr,
            motor_tor=motor_tor_arr,
            lin_vel=torso_lin_vel,
            ang_vel=torso_ang_vel,
            pos=torso_pos,
            rot=torso_rot,
            joint_pos=joint_pos_arr,
            joint_vel=joint_vel_arr,
        )
        return obs

    def check_self_collisions(self):
        contact_pairs = []
        if self.data.ncon > 0:
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
                g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
                if "floor" not in g1 and "floor" not in g2:
                    contact_pairs.append((g1, g2))

        return contact_pairs

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """Sets the proportional gain (Kp) values for the motors.

        This method updates the Kp values for each motor specified in the `motor_kps` dictionary. If the controller is an instance of `MotorController`, it sets the Kp value directly in the controller. Otherwise, it adjusts the gain and bias parameters of the actuator model.

        Args:
            motor_kps (Dict[str, float]): A dictionary where keys are motor names and values are the Kp values to be set.
        """
        kp_ratio = self.robot.config["actuators"]["kp_ratio"]
        for name, kp in motor_kps.items():
            if isinstance(self.controller, MotorController):
                self.controller.kp[self.model.actuator(name).id] = kp / kp_ratio
            else:
                self.model.actuator(name).gainprm[0] = kp / kp_ratio
                self.model.actuator(name).biasprm[1] = -kp / kp_ratio

    def set_motor_target(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the target angles for the motors.

        Args:
            motor_angles (Dict[str, float] | npt.NDArray[np.float32]): A dictionary mapping motor names to their target angles or a NumPy array of target angles. If a dictionary is provided, the values are converted to a NumPy array of type float32.
        """
        if isinstance(motor_angles, dict):
            self.target_motor_pos = np.array(
                list(motor_angles.values()), dtype=np.float32
            )
        else:
            self.target_motor_pos = motor_angles

    def set_motor_angles(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the motor angles for the robot and updates the corresponding joint and passive angles.

        Args:
            motor_angles (Dict[str, float] | npt.NDArray[np.float32]): A dictionary mapping motor names to angles or an array of motor angles in the order specified by the robot's motor ordering.
        """
        if not isinstance(motor_angles, dict):
            motor_angles = dict(zip(self.robot.motor_ordering, motor_angles))

        dof_angles = self.robot.motor_to_dof_angles(motor_angles)
        for name in dof_angles:
            self.data.joint(name).qpos = dof_angles[name]

    def set_joint_angles(
        self, joint_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the joint angles of the robot.

        This method updates the joint positions of the robot based on the provided joint angles. It converts the input joint angles to motor and passive angles and updates the robot's data structure accordingly.

        Args:
            joint_angles (Dict[str, float] | npt.NDArray[np.float32]): A dictionary mapping joint names to their respective angles, or a NumPy array of joint angles in the order specified by the robot's joint ordering.
        """
        if not isinstance(joint_angles, dict):
            joint_angles = dict(zip(self.robot.joint_ordering, joint_angles))

        dof_angles = self.robot.joint_to_dof_angles(joint_angles)
        for name in dof_angles:
            self.data.joint(name).qpos = dof_angles[name]

    def set_qpos(self, qpos: npt.NDArray[np.float32]):
        """Set the position of the system's generalized coordinates.

        Args:
            qpos (npt.NDArray[np.float32]): An array representing the desired positions of the system's generalized coordinates.
        """
        self.data.qpos = qpos

    def set_motor_dynamics(self, motor_dyn: Dict[str, float]):
        """Sets the motor dynamics by updating the controller's attributes.

        Args:
            motor_dyn (Dict[str, float]): A dictionary where keys are attribute names and values are the corresponding dynamics values to be set on the controller.
        """
        for key, value in motor_dyn.items():
            setattr(self.controller, key, value)

    def set_joint_dynamics(self, joint_dyn: Dict[str, Dict[str, float]]):
        """Sets the dynamics parameters for specified joints in the model.

        Args:
            joint_dyn (Dict[str, Dict[str, float]]): A dictionary where each key is a joint name and the value is another dictionary containing dynamics parameters and their corresponding values to be set for that joint.
        """
        for joint_name, dyn in joint_dyn.items():
            for key, value in dyn.items():
                setattr(self.model.joint(joint_name), key, value)

    def forward(self):
        """Advances the simulation forward by a specified number of frames and visualizes the result if a visualizer is available.

        Iterates through the simulation for the number of frames specified by `self.n_frames`, updating the model state at each step. If a visualizer is provided, it visualizes the current state of the simulation data.
        """
        mujoco.mj_forward(self.model, self.data)

        if self.visualizer is not None:
            self.visualizer.visualize(self.data)

    def reset(self, key_name: str = "home"):
        """Reset state and optional keyframe (required between Optuna trials so runs are independent)."""
        mujoco.mj_resetData(self.model, self.data)
        applied_keyframe = False
        try:
            kid = self.model.key(key_name).id
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
            applied_keyframe = True
        except (AttributeError, KeyError, ValueError):
            pass
        if not applied_keyframe and self.home_qpos is not None:
            n = min(len(self.home_qpos), int(self.model.nq) - self.q_start_idx)
            self.data.qpos[self.q_start_idx : self.q_start_idx + n] = self.home_qpos[:n]
        self.target_motor_pos = np.zeros(self.model.nu, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """Advances the simulation by a specified number of frames and updates the visualizer.

        This method iterates over the number of frames defined by `n_frames`, updating the control inputs using the controller's step method based on the current position and velocity of the motors. It then advances the simulation state using Mujoco's `mj_step` function. If a visualizer is provided, it updates the visualization with the current simulation data.
        """
        for _ in range(self.n_frames):
            self.data.ctrl = self.controller.step(
                self.data.qpos[self.q_start_idx + self.motor_indices],
                self.data.qvel[self.qd_start_idx + self.motor_indices],
                self.data.qacc[self.qd_start_idx + self.motor_indices],
                self.target_motor_pos,
            )
            mujoco.mj_step(self.model, self.data)

        if self.visualize_elevation:
            global_map, _, x_coords, y_coords = get_elevation_map(
                self.data,
                self.elevation_info,
                self.elevation_map_size,
                self.elevation_map_resolution,
                self.elevation_map_forward_offset,
            )
            update_elevation_visualization(
                self.data, self.model, global_map, x_coords, y_coords
            )

        if self.visualizer is not None:
            self.visualizer.visualize(self.data)

    def save_recording(
        self,
        exp_folder_path: str,
        dt: float,
        render_every: int,
        name: str = "eval.mp4",
    ):
        """Saves a recording of the current MuJoCo simulation.

        Args:
            exp_folder_path (str): The path to the folder where the recording will be saved.
            dt (float): The time step interval for the recording.
            render_every (int): The frequency at which frames are rendered.
            name (str, optional): The name of the output video file. Defaults to "eval.mp4".
        """
        if isinstance(self.visualizer, MuJoCoRenderer):
            self.visualizer.save_recording(exp_folder_path, dt, render_every, name)

    def close(self):
        """Closes the visualizer if it is currently open."""
        if self.visualizer is not None:
            self.visualizer.close()
