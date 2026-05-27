"""GUI application for editing and creating keyframes in MuJoCo simulations.

This module provides a comprehensive GUI interface for creating, editing, and testing
keyframe sequences for robot tasks in MuJoCo simulations. It includes real-time
visualization, joint control sliders, and keyframe management capabilities.
"""

import argparse
import copy
import os
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import partial
from typing import List

import joblib
import mujoco
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation as R

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.io_utils import find_latest_file_with_time_str
from toddlerbot.utils.math_utils import interpolate_action

# This script is a GUI application that allows the user to interact with MuJoCo simulations in real-time
# and create keyframes for a given task. The keyframes can be tested and saved as a sequence
# for later use. The user can also visualize the keyframes and the sequence in MuJoCo.
# This script is highly inspired by the following code snippets: https://gist.github.com/JeanElsner/755d0feb49864ecadab4ef00fd49a22b

format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(2)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2, 0)
# Deprecated
# format.setColorSpace(format.sRGBColorSpace)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


@dataclass
class Keyframe:
    """Dataclass for storing keyframe information."""

    name: str
    motor_pos: np.ndarray
    joint_pos: np.ndarray | None = None
    qpos: np.ndarray | None = None


class Viewport(QOpenGLWindow):
    """Class for rendering the MuJoCo simulation in a Qt window."""

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn, mutex) -> None:
        """Initializes an instance of the class with the given parameters and sets up a timer for periodic updates.

        Args:
            model: The model object to be used within the class.
            data: The data object associated with the model.
            cam: The camera object for capturing or processing images.
            opt: Options or configurations for the model or process.
            scn: The scene object related to the model or visualization.
            mutex: A threading lock to ensure thread-safe operations.

        Attributes:
            width (int): The width of the processing area, initialized to 0.
            height (int): The height of the processing area, initialized to 0.
            __last_pos: The last known position, initially set to None.
            runtime (collections.deque): A deque to store runtime data with a maximum length of 1000.
            timer (QTimer): A QTimer object set to trigger updates at approximately 60 frames per second.
        """
        super().__init__()

        self.model = model
        self.data = data

        self.cam = cam
        self.opt = opt
        self.scn = scn

        self.width = 0
        self.height = 0
        self.__last_pos = None

        self.mutex = mutex

        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1 / 60 * 1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        """Handles the mouse press event by updating the last known position of the mouse.

        Args:
            event (QMouseEvent): The mouse event containing information about the mouse press, including the position.
        """
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        """Handles mouse movement events to perform camera actions in a MuJoCo simulation.

        This method interprets mouse movements and translates them into camera actions
        such as move, rotate, or zoom based on the mouse button pressed. It updates the
        camera view accordingly using the MuJoCo API.

        Args:
            event (QMouseEvent): The mouse event containing information about the
                mouse position and button states.

        Returns:
            None
        """
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(
            self.model, action, dx / self.height, dy / self.height, self.scn, self.cam
        )
        self.__last_pos = pos

    def wheelEvent(self, event):
        """Handles the mouse wheel event to zoom the camera in or out in the MuJoCo simulation.

        This method is triggered when the mouse wheel is scrolled. It adjusts the camera's zoom level based on the scroll direction and magnitude.

        Args:
            event: The QWheelEvent object containing information about the wheel event, such as the scroll delta.
        """
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.0005 * event.angleDelta().y(),
            self.scn,
            self.cam,
        )

    def initializeGL(self):
        """Initializes the OpenGL context for rendering.

        This method sets up the OpenGL context using the MuJoCo library, which is necessary for rendering the simulation. It creates a new `MjrContext` object associated with the model and specifies the font scale to be used in the rendering context.

        Attributes:
            con (mujoco.MjrContext): The OpenGL context for rendering, initialized with the specified model and font scale.
        """
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        """Resize the OpenGL viewport to the specified width and height.

        This method updates the internal width and height attributes of the
        OpenGL context to match the new dimensions provided.

        Args:
            w (int): The new width of the OpenGL viewport.
            h (int): The new height of the OpenGL viewport.
        """
        self.width = w
        self.height = h

    def paintGL(self) -> None:
        """Renders the current scene using the MuJoCo physics engine and updates the runtime.

        This method updates the scene with the current model and data, renders it to a viewport,
        and calculates the time taken for rendering. The average runtime is then emitted through
        a signal for further processing or display.

        """
        t = time.monotonic()

        with QMutexLocker(self.mutex):
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scn,
            )

        for pos, color in sim.ee_markers:
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.005, 0.0, 0.0], dtype=np.float64),
                np.array(pos, dtype=np.float64),
                np.eye(3, dtype=np.float64).flatten(),
                np.array(color, dtype=np.float32),
            )

            self.scn.ngeom += 1

        viewport = mujoco.MjrRect(0, 0, self.width * 2, self.height * 2)
        mujoco.mjr_render(viewport, self.scn, self.con)

        self.runtime.append(time.monotonic() - t)
        self.updateRuntime.emit(np.average(self.runtime))

    def keyPressEvent(self, event):
        """Handles keyboard events in the viewport window.

        Args:
            event (QKeyEvent): The keyboard event containing key information.
        """
        from PySide6.QtCore import Qt

        if event.key() == Qt.Key_Space:
            # Print qpos xyz and quaternion when spacebar is pressed
            with QMutexLocker(self.mutex):
                qpos = self.data.qpos
                print(
                    f"qpos xyz: [{qpos[0]:.6f}, {qpos[1]:.6f}, {qpos[2]:.6f}], "
                    f"quat (wxyz): [{qpos[3]:.6f}, {qpos[4]:.6f}, {qpos[5]:.6f}, {qpos[6]:.6f}]"
                )

        super().keyPressEvent(event)


class UpdateSimThread(QThread):
    """Thread for updating the simulation in real-time and handling keyframe and trajectory testing."""

    updated = Signal()
    state_data_curr = Signal(np.ndarray, np.ndarray, np.ndarray)
    traj_data_curr = Signal(list, list, list, list, list, list, list, list, list)

    def __init__(
        self, sim: MuJoCoSim, robot: Robot, mutex: QMutex, parent=None
    ) -> None:
        """Initializes the simulation control class with the given simulation, robot, and mutex.

        Args:
            sim (MuJoCoSim): The MuJoCo simulation instance to be controlled.
            robot (Robot): The robot instance containing default joint angles.
            mutex (QMutex): A mutex for synchronizing access to shared resources.
            parent (optional): The parent object, if any, for the class instance.

        Attributes:
            sim (MuJoCoSim): Stores the simulation instance.
            mutex (QMutex): Stores the mutex for synchronization.
            running (bool): Indicates if the simulation control is active.
            is_testing (bool): Indicates if the system is in testing mode.
            update_joint_angles_requested (bool): Flag to request joint angle updates.
            joint_angles_to_update (list): List of joint angles to update.
            update_qpos_requested (bool): Flag to request qpos updates.
            qpos_to_update (list): List of qpos values to update.
            keyframe_test_counter (int): Counter for keyframe testing.
            keyframe_test_dt (int): Time delta for keyframe testing.
            traj_test_counter (int): Counter for trajectory testing.
            action_traj (optional): Stores the action trajectory for testing.
            traj_test_dt (int): Time delta for trajectory testing.
            traj_physics_enabled (bool): Indicates if trajectory physics is enabled.

        """
        super().__init__(parent)
        self.sim = sim
        self.mutex = mutex
        self.running = True
        self.is_testing = False
        self.is_qpos_traj = False
        self.is_relative_frame = True

        self.update_joint_angles_requested = False
        self.joint_angles_to_update = robot.default_joint_angles.copy()

        self.update_qpos_requested = False
        self.qpos_to_update = sim.model.qpos0.copy()

        self.sim.forward()

        self.keyframe_test_counter = -1
        self.keyframe_test_dt = 0

        self.traj_test_counter = -1
        self.action_traj = None
        self.traj_test_dt = 0
        self.traj_physics_enabled = False

        self.qpos_replay = []
        self.motor_vel_replay = []
        self.joint_vel_replay = []
        self.body_pos_replay = []
        self.body_quat_replay = []
        self.body_lin_vel_replay = []
        self.body_ang_vel_replay = []
        self.site_pos_replay = []
        self.site_quat_replay = []

    @Slot()
    def update_joint_angles(self, joint_angles_to_update):
        """Updates the joint angles for the robotic system.

        This method sets a flag indicating that a joint angle update has been requested and stores a copy of the new joint angles to be updated. It also prints a confirmation message to the console.

        Args:
            joint_angles_to_update (list or array-like): A list or array containing the new joint angles to be updated.
        """
        self.update_joint_angles_requested = True
        self.joint_angles_to_update = joint_angles_to_update.copy()

        # print("Joint angles update requested!")

    @Slot()
    def update_qpos(self, qpos):
        """Request an update to the current position configuration.

        This method sets a flag indicating that an update to the position
        configuration (`qpos`) is requested. It stores a copy of the new
        position configuration to be updated later.

        Args:
            qpos (list or array-like): The new position configuration to be
                updated. It should be a list or array-like structure containing
                the desired position values.
        """
        self.update_qpos_requested = True
        self.qpos_to_update = qpos.copy()

        print("Qpos update requested!")

    @Slot()
    def request_on_ground(self):
        """Aligns the torso of the simulated model with the ground based on foot positions.

        This method checks the z-coordinates of the left and right foot to determine which foot is closer to the ground. It then adjusts the torso's position and orientation in the simulation to align with the selected foot, ensuring the model's feet are on the ground. The simulation is updated with the new torso transformation.

        If the `is_testing` attribute is set to True, the function does nothing.

        Attributes:
            is_testing (bool): A flag indicating whether the function should perform its operations.
            sim (object): The simulation object containing methods and data for body transformations.
            mutex (QMutex): A mutex for thread-safe operations on the simulation data.

        """
        # Thread-safe check of is_testing flag
        with QMutexLocker(self.mutex):
            testing_in_progress = self.is_testing

        if not testing_in_progress:
            torso_t_curr = self.sim.get_body_transform("torso")

            site_z_min = float("inf")
            for site_name in ["left_hand", "right_hand", "left_foot", "right_foot"]:
                curr_transform = self.sim.get_site_transform(f"{site_name}_center")

                if curr_transform[2, 3] < site_z_min:
                    site_z_min = curr_transform[2, 3]

            # aligned_torso_t = (
            #     self.right_foot_t_init
            #     @ np.linalg.inv(right_foot_t_curr)
            #     @ torso_t_curr
            # )
            aligned_torso_height = torso_t_curr[2, 3] - site_z_min

            with QMutexLocker(self.mutex):
                # Update the simulation with the new torso position and orientation
                # self.sim.data.qpos[:3] = aligned_torso_t[:3, 3]
                # self.sim.data.qpos[3:7] = R.from_matrix(
                #     aligned_torso_t[:3, :3]
                # ).as_quat(scalar_first=True)
                self.sim.data.qpos[2] = aligned_torso_height
                self.sim.forward()

            print("Aligned with the ground!")

    @Slot()
    def request_state_data(self):
        """Retrieve and emit current state data from the simulation.

        This method fetches the current motor and joint angles from the simulation
        environment, along with the position data (`qpos`). It then emits these
        values using the `stateDataCurr` signal. This function is only executed
        when the system is not in testing mode.

        """
        # Thread-safe check and data retrieval
        with QMutexLocker(self.mutex):
            if not self.is_testing:
                motor_pos = self.sim.get_motor_angles(type="array")
                joint_pos = self.sim.get_joint_angles(type="array")
                qpos = self.sim.data.qpos.copy()
                self.state_data_curr.emit(motor_pos, joint_pos, qpos)  # Emit data
                print("State data requested!")

    @Slot()
    def request_keyframe_test(self, keyframe: Keyframe, dt: float):
        """Initiates a keyframe test by setting the simulation state and motor targets.

        This method sets the simulation's position and motor targets to those specified
        in the given keyframe and begins a test sequence if not already in progress.

        Args:
            keyframe (Keyframe): The keyframe containing the desired positions and motor targets.
            dt (float): The time duration for which the keyframe test should run.

        """
        with QMutexLocker(self.mutex):
            if not self.is_testing:
                # Reset any previous test state
                self.keyframe_test_counter = -1
                self.traj_test_counter = -1

                # Set up simulation state
                self.sim.data.qpos = keyframe.qpos.copy()
                self.sim.forward()
                self.sim.set_motor_target(keyframe.motor_pos.copy())

                # Start keyframe test
                self.keyframe_test_dt = dt
                self.keyframe_test_counter = 0
                self.is_testing = True
                print("Keyframe test started!")

    @Slot()
    def request_trajectory_test(
        self,
        qpos_start: np.ndarray,
        traj: List[np.ndarray],
        dt: float,
        physics_enabled: bool,
        is_qpos_traj: bool = False,
        is_relative_frame: bool = True,
    ):
        """Initiates a trajectory test by setting the initial conditions and parameters.

        This function sets up a trajectory test by initializing the simulation state
        with a given starting position, action or qpos trajectory, time step, and physics settings.
        It prepares the system for testing by resetting relevant counters and flags.

        Args:
            qpos_start (np.ndarray): The starting joint positions for the simulation.
            traj (List[np.ndarray]): A list of action or qpos vectors representing the trajectory to be tested.
            dt (float): The time step duration for each action in the trajectory.
            physics_enabled (bool): A flag indicating whether physics should be enabled during the trajectory test.
            is_qpos_traj (bool): Flag to indicate if the trajectory is composed of full qpos vectors.
            is_relative_frame (bool): Flag to indicate if the trajectory should be stored in the robot's relative frame.
        """
        with QMutexLocker(self.mutex):
            if not self.is_testing:
                # Reset any previous test state
                self.keyframe_test_counter = -1
                self.traj_test_counter = -1

                # Set up simulation state
                self.sim.data.qpos = qpos_start.copy()
                self.sim.data.qvel[:] = 0
                self.sim.data.ctrl[:] = 0
                self.sim.forward()

                # Set up trajectory test parameters
                if is_qpos_traj:
                    # TODO: Sharing self.action_traj for qpos trajectory tests for now not to break existing code
                    self.action_traj = traj
                    print("Running qpos trajectory test!")
                    print(f"Trajectory length: {len(self.action_traj)}")
                else:
                    self.action_traj = traj
                    print("Running action trajectory test!")
                    print(f"Trajectory length: {len(self.action_traj)}")
                self.traj_test_dt = dt
                self.traj_physics_enabled = physics_enabled
                self.traj_test_counter = 0

                # Clear replay data
                self.qpos_replay.clear()
                self.motor_vel_replay.clear()
                self.joint_vel_replay.clear()
                self.body_pos_replay.clear()
                self.body_quat_replay.clear()
                self.body_lin_vel_replay.clear()
                self.body_ang_vel_replay.clear()
                self.site_pos_replay.clear()
                self.site_quat_replay.clear()

                # Start trajectory test
                self.is_testing = True

                self.is_qpos_traj = is_qpos_traj
                self.is_relative_frame = is_relative_frame
                if self.is_relative_frame:
                    print("Saving poses in robot-relative frame!")
                else:
                    print("saving poses in global frame!")

    def run(self) -> None:
        """Executes the main loop for updating simulation states and handling various test scenarios.

        This method continuously runs while `self.running` is True, performing updates to the simulation
        based on requested state changes or test conditions. It handles updating the simulation's
        position (`qpos`), joint angles, keyframe tests, and trajectory tests. The method ensures
        thread safety using `QMutexLocker` and emits signals to notify when updates are complete.

        Attributes:
            self.running (bool): Flag to control the loop execution.
            self.update_qpos_requested (bool): Indicates if a position update is requested.
            self.update_joint_angles_requested (bool): Indicates if a joint angles update is requested.
            self.keyframe_test_counter (int): Counter for keyframe testing.
            self.traj_test_counter (int): Counter for trajectory testing.
            self.traj_physics_enabled (bool): Flag to enable physics during trajectory testing.
            self.action_traj (list): List of actions for trajectory testing.
            self.keyframe_test_dt (float): Time delta for keyframe testing.
            self.traj_test_dt (float): Time delta for trajectory testing.
        """
        while self.running:
            if self.update_qpos_requested:
                with QMutexLocker(self.mutex):
                    # Stop any ongoing tests before updating qpos
                    self.is_testing = False
                    self.keyframe_test_counter = -1
                    self.traj_test_counter = -1

                    self.sim.data.qpos = self.qpos_to_update.copy()
                    self.sim.forward()
                    self.update_qpos_requested = False

                self.updated.emit()

            elif self.update_joint_angles_requested:
                with QMutexLocker(self.mutex):
                    # Stop any ongoing tests before updating joint angles
                    self.is_testing = False
                    self.keyframe_test_counter = -1
                    self.traj_test_counter = -1

                    joint_angles = self.sim.get_joint_angles()
                    joint_angles.update(self.joint_angles_to_update)
                    self.sim.set_joint_angles(joint_angles)
                    self.sim.forward()
                    self.update_joint_angles_requested = False

                self.updated.emit()  # Notify UI that update is complete

            elif self.keyframe_test_counter >= 0 and self.keyframe_test_counter <= 100:
                # Check for completion with thread-safe counter access
                with QMutexLocker(self.mutex):
                    current_test_counter = self.keyframe_test_counter
                    if current_test_counter == 100:
                        self.keyframe_test_counter = -1
                        self.is_testing = False
                        print("Keyframe test completed.")
                        continue
                    else:
                        # Perform simulation step
                        self.sim.step()
                        self.keyframe_test_counter += 1

                # Sleep outside mutex to reduce contention
                self.msleep(int(self.keyframe_test_dt * 1000))

            elif self.traj_test_counter >= 0 and self.traj_test_counter <= len(
                self.action_traj
            ):
                # Check if test trajectory is stopped (with proper synchronization)
                with QMutexLocker(self.mutex):
                    trajectory_stopped = not self.is_testing
                    current_counter = self.traj_test_counter

                if trajectory_stopped:
                    # Copy data for emission outside mutex
                    qpos_copy = self.qpos_replay.copy()
                    motor_vel_copy = self.motor_vel_replay.copy()
                    joint_vel_copy = self.joint_vel_replay.copy()
                    body_pos_copy = self.body_pos_replay.copy()
                    body_quat_copy = self.body_quat_replay.copy()
                    body_lin_vel_copy = self.body_lin_vel_replay.copy()
                    body_ang_vel_copy = self.body_ang_vel_replay.copy()
                    site_pos_copy = self.site_pos_replay.copy()
                    site_quat_copy = self.site_quat_replay.copy()

                    # Reset state
                    self.traj_test_counter = -1
                    self.keyframe_test_counter = -1
                    self.action_traj = None
                    self.qpos_replay.clear()
                    self.motor_vel_replay.clear()
                    self.joint_vel_replay.clear()
                    self.body_pos_replay.clear()
                    self.body_quat_replay.clear()
                    self.body_lin_vel_replay.clear()
                    self.body_ang_vel_replay.clear()
                    self.site_pos_replay.clear()
                    self.site_quat_replay.clear()

                    # Emit signal outside critical section
                    self.traj_data_curr.emit(
                        qpos_copy,
                        motor_vel_copy,
                        joint_vel_copy,
                        body_pos_copy,
                        body_quat_copy,
                        body_lin_vel_copy,
                        body_ang_vel_copy,
                        site_pos_copy,
                        site_quat_copy,
                    )
                    print("Trajectory stopped and state reset.")
                    continue

                if current_counter == len(self.action_traj):
                    # Copy data for emission outside mutex
                    qpos_copy = self.qpos_replay.copy()
                    motor_vel_copy = self.motor_vel_replay.copy()
                    joint_vel_copy = self.joint_vel_replay.copy()
                    body_pos_copy = self.body_pos_replay.copy()
                    body_quat_copy = self.body_quat_replay.copy()
                    body_lin_vel_copy = self.body_lin_vel_replay.copy()
                    body_ang_vel_copy = self.body_ang_vel_replay.copy()
                    site_pos_copy = self.site_pos_replay.copy()
                    site_quat_copy = self.site_quat_replay.copy()

                    # Reset state with mutex protection
                    with QMutexLocker(self.mutex):
                        self.traj_test_counter = -1
                        self.keyframe_test_counter = -1
                        self.action_traj = None
                        self.qpos_replay.clear()
                        self.motor_vel_replay.clear()
                        self.joint_vel_replay.clear()
                        self.body_pos_replay.clear()
                        self.body_quat_replay.clear()
                        self.body_lin_vel_replay.clear()
                        self.body_ang_vel_replay.clear()
                        self.site_pos_replay.clear()
                        self.site_quat_replay.clear()
                        self.is_testing = False

                    # Emit signal outside critical section
                    self.traj_data_curr.emit(
                        qpos_copy,
                        motor_vel_copy,
                        joint_vel_copy,
                        body_pos_copy,
                        body_quat_copy,
                        body_lin_vel_copy,
                        body_ang_vel_copy,
                        site_pos_copy,
                        site_quat_copy,
                    )
                    print("Trajectory test finished and state reset.")
                    continue

                t1 = time.monotonic()

                with QMutexLocker(self.mutex):
                    if self.traj_physics_enabled:
                        self.sim.set_motor_target(
                            self.action_traj[self.traj_test_counter]
                        )
                        self.sim.step()
                    elif self.is_qpos_traj:
                        self.sim.set_qpos(self.action_traj[self.traj_test_counter])
                        self.sim.forward()
                    else:
                        self.sim.set_motor_angles(
                            self.action_traj[self.traj_test_counter]
                        )
                        self.sim.forward()

                # Record simulation data (thread-safe access)
                with QMutexLocker(self.mutex):
                    # self.sim.check_self_collisions()
                    qpos_data = np.array(self.sim.data.qpos, dtype=np.float32)

                    # Compute motor_vel and joint_vel: finite differences for qpos traj, qvel for action traj
                    if (
                        self.is_qpos_traj
                        and self.traj_test_counter < len(self.action_traj) - 1
                    ):
                        # qpos trajectory: use MuJoCo's mj_differentiatePos for proper qvel computation
                        current_qpos = self.action_traj[self.traj_test_counter]
                        next_qpos = self.action_traj[self.traj_test_counter + 1]
                        qvel = np.zeros_like(self.sim.data.qvel)
                        mujoco.mj_differentiatePos(
                            self.sim.model,
                            qvel,
                            self.traj_test_dt,
                            current_qpos,
                            next_qpos,
                        )
                        # Extract motor and joint velocities from computed qvel
                        motor_vel_data = qvel[
                            self.sim.qd_start_idx + self.sim.motor_indices
                        ]
                        joint_vel_data = qvel[
                            self.sim.qd_start_idx + self.sim.joint_indices
                        ]
                    else:
                        # action trajectory: use qvel from sim.step() dynamics
                        motor_vel_data = self.sim.get_motor_velocities(type="array")
                        joint_vel_data = self.sim.get_joint_velocities(type="array")

                    body_pos = []
                    body_quat = []
                    body_lin_vel = []
                    body_ang_vel = []
                    site_pos = []
                    site_quat = []

                    if self.is_relative_frame:
                        # Get raw body data from simulation
                        body_pos_world = np.array(self.sim.data.xpos, dtype=np.float32)
                        body_quat_world = np.array(
                            self.sim.data.xquat, dtype=np.float32
                        )
                        body_lin_vel_world = np.array(
                            self.sim.data.cvel[:, 3:], dtype=np.float32
                        )
                        body_ang_vel_world = np.array(
                            self.sim.data.cvel[:, :3], dtype=np.float32
                        )

                        # Transform to robot-relative frame (relative to torso)
                        torso_pos = body_pos_world[1]  # Torso is typically body index 1
                        torso_quat = body_quat_world[1]  # Torso quaternion

                        # Convert torso quaternion to rotation matrix for transformation
                        torso_rot = R.from_quat(
                            np.array(
                                [
                                    torso_quat[1],
                                    torso_quat[2],
                                    torso_quat[3],
                                    torso_quat[0],
                                ]
                            )
                        )  # xyzw format
                        torso_rot_inv = torso_rot.inv()

                        # Transform all body positions to robot-relative frame
                        body_pos_rel = np.zeros_like(body_pos_world)
                        body_quat_rel = np.zeros_like(body_quat_world)
                        body_lin_vel_rel = np.zeros_like(body_lin_vel_world)
                        body_ang_vel_rel = np.zeros_like(body_ang_vel_world)

                        for i in range(len(body_pos_world)):
                            # Transform position: subtract torso position and rotate to robot frame
                            pos_diff = body_pos_world[i] - torso_pos
                            body_pos_rel[i] = torso_rot_inv.apply(pos_diff)

                            # Transform quaternion: multiply by inverse torso rotation
                            body_quat_i = R.from_quat(
                                np.array(
                                    [
                                        body_quat_world[i][1],
                                        body_quat_world[i][2],
                                        body_quat_world[i][3],
                                        body_quat_world[i][0],
                                    ]
                                )
                            )  # xyzw
                            body_quat_rel_i = body_quat_i * torso_rot_inv
                            quat_rel_wxyz = body_quat_rel_i.as_quat(
                                scalar_first=True
                            )  # wxyz format
                            body_quat_rel[i] = quat_rel_wxyz

                            body_lin_vel_rel[i] = torso_rot_inv.apply(
                                body_lin_vel_world[i]
                            )
                            body_ang_vel_rel[i] = torso_rot_inv.apply(
                                body_ang_vel_world[i]
                            )

                        body_pos = body_pos_rel.copy()
                        body_quat = body_quat_rel.copy()
                        body_lin_vel = body_lin_vel_rel.copy()
                        body_ang_vel = body_ang_vel_rel.copy()

                        # Record site poses
                        for side in ["left", "right"]:
                            for ee_name in ["hand", "foot"]:
                                ee_pos_world = self.sim.data.site(
                                    f"{side}_{ee_name}_center"
                                ).xpos.copy()
                                ee_mat = self.sim.data.site(
                                    f"{side}_{ee_name}_center"
                                ).xmat.reshape(3, 3)
                                ee_quat_world = R.from_matrix(ee_mat).as_quat(
                                    scalar_first=True
                                )

                                # Transform site position to robot-relative frame
                                ee_pos_diff = ee_pos_world - torso_pos
                                ee_pos_rel = torso_rot_inv.apply(ee_pos_diff)

                                # Transform site quaternion to robot-relative frame
                                ee_rot_world = R.from_quat(
                                    np.array(
                                        [
                                            ee_quat_world[1],
                                            ee_quat_world[2],
                                            ee_quat_world[3],
                                            ee_quat_world[0],
                                        ]
                                    )
                                )  # xyzw
                                ee_rot_rel = ee_rot_world * torso_rot_inv
                                ee_quat_rel = ee_rot_rel.as_quat(
                                    scalar_first=True
                                )  # wxyz format

                                site_pos.append(ee_pos_rel)
                                site_quat.append(ee_quat_rel)
                    else:
                        body_pos_world = np.array(self.sim.data.xpos, dtype=np.float32)
                        body_quat_world = np.array(
                            self.sim.data.xquat, dtype=np.float32
                        )
                        body_lin_vel_world = np.array(
                            self.sim.data.cvel[:, 3:], dtype=np.float32
                        )
                        body_ang_vel_world = np.array(
                            self.sim.data.cvel[:, :3], dtype=np.float32
                        )
                        body_pos = body_pos_world.copy()
                        body_quat = body_quat_world.copy()
                        body_lin_vel = body_lin_vel_world.copy()
                        body_ang_vel = body_ang_vel_world.copy()

                        # Record site poses
                        for side in ["left", "right"]:
                            for ee_name in ["hand", "foot"]:
                                ee_pos_world = self.sim.data.site(
                                    f"{side}_{ee_name}_center"
                                ).xpos.copy()
                                ee_mat = self.sim.data.site(
                                    f"{side}_{ee_name}_center"
                                ).xmat.reshape(3, 3)
                                ee_quat_world = R.from_matrix(ee_mat).as_quat(
                                    scalar_first=True
                                )

                                site_pos.append(ee_pos_world)
                                site_quat.append(ee_quat_world)

                # Store data outside mutex to minimize lock time
                self.qpos_replay.append(qpos_data)
                self.motor_vel_replay.append(motor_vel_data)
                self.joint_vel_replay.append(joint_vel_data)
                self.body_pos_replay.append(body_pos)
                self.body_quat_replay.append(body_quat)
                self.body_lin_vel_replay.append(body_lin_vel)
                self.body_ang_vel_replay.append(body_ang_vel)

                self.site_pos_replay.append(np.array(site_pos, dtype=np.float32))
                self.site_quat_replay.append(np.array(site_quat, dtype=np.float32))
                t2 = time.monotonic()
                self.traj_test_counter += 1
                time_left = self.traj_test_dt - (t2 - t1)
                if time_left > 0:
                    self.msleep(int(time_left * 1000))
                else:
                    # Yield briefly to prevent busy waiting
                    self.msleep(1)
            else:
                # No operations pending, yield briefly to prevent busy waiting
                self.msleep(5)

    def stop(self):
        """Stops the execution of the current process.

        Sets the running state to False and waits for the process to terminate.
        """
        self.running = False
        self.wait()


class MuJoCoApp(QMainWindow):
    """Main application window for interacting with MuJoCo simulations and creating keyframes."""

    def __init__(
        self,
        sim: MuJoCoSim,
        robot: Robot,
        task_name: str,
        run_name: str,
        dt: float = 0.02,
    ):
        """Initializes the class with simulation, robot, and task details, setting up directories and loading data.

        Args:
            sim (MuJoCoSim): The simulation environment instance.
            robot (Robot): The robot instance to be used in the simulation.
            task_name (str): The name of the task to be performed.
            run_name (str): The name of the run, used for directory and file management.

        Attributes:
            sim (MuJoCoSim): Stores the simulation environment instance.
            robot (Robot): Stores the robot instance.
            task_name (str): Stores the task name.
            result_dir (str): Directory path for storing results.
            data_path (str): Path to the data file for the task.
            mirror_joint_signs (dict): Dictionary mapping joint names to their mirror signs.
            paused (bool): Indicates if the simulation is paused.
            slider_columns (int): Number of columns for sliders in the UI.
            qpos_offset (int): Offset for the position of the robot.
            model: The model of the simulation.
            data: The data of the simulation.
            cam: The camera instance for the simulation.
            opt: Options for the MuJoCo visualization.
            scn: The scene for the MuJoCo visualization.
            mutex (QMutex): Mutex for thread safety.
            viewport (Viewport): The viewport for rendering the simulation.
            sim_thread (UpdateSimThread): Thread for updating the simulation.

        """
        super().__init__()

        self.sim = sim
        self.robot = robot
        self.task_name = task_name
        self.dt = dt

        if run_name == task_name:
            time_str = time.strftime("%Y%m%d_%H%M%S")
            self.result_dir = os.path.join(
                "results", f"{robot.name}_keyframe_{sim.name}_{time_str}"
            )
            os.makedirs(self.result_dir, exist_ok=True)
            self.data_path = os.path.join(self.result_dir, f"{task_name}.lz4")
            shutil.copy2(os.path.join("motion", f"{task_name}.lz4"), self.data_path)

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

        self.paused = True
        self.slider_columns = 4
        self.qpos_offset = 7

        self.saved_left_foot_pose = None
        self.saved_right_foot_pose = None
        self.saved_left_hand_pose = None

        self.model = sim.model
        self.data = sim.data
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)

        self.mutex = QMutex()

        self.viewport = Viewport(
            sim.model, sim.data, self.cam, self.opt, self.scn, self.mutex
        )
        self.viewport.updateRuntime.connect(self.show_runtime)

        self.sim_thread = UpdateSimThread(sim, robot, self.mutex, self)
        self.sim_thread.start()

        self.sim_thread.state_data_curr.connect(self.update_keyframe_with_signal)
        self.sim_thread.traj_data_curr.connect(self.update_traj_with_signal)

        self.setup_ui()
        self.load_data()

    def create_free_camera(self):
        """Creates and configures a free camera for the MuJoCo simulation environment.

        This function initializes a free camera, sets its type, and configures its
        position to focus on the median position of all geometries in the simulation.
        The camera's distance and elevation are also set to provide a broad view of
        the environment.

        Returns:
            mujoco.MjvCamera: A configured free camera object.
        """
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        for i in range(3):
            cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        cam.distance = self.model.stat.extent * 2
        cam.elevation = -45
        return cam

    @Slot(float)
    def show_runtime(self, fps: float):
        """Displays the average runtime and simulation time in the status bar.

        This method updates the status bar with the average runtime in milliseconds
        calculated from the frames per second (fps) and the simulation time in milliseconds
        from the data attribute.

        Args:
            fps (float): The frames per second value used to calculate the average runtime.
        """
        self.statusBar().showMessage(
            f"Average runtime: {fps * 1000:.0f}ms\t\
                                        Simulation time: {self.data.time * 1000:.0f}ms"
        )

    def setup_ui(self):
        """Sets up the user interface for the application, including buttons, checkboxes, entry fields, and sliders for managing keyframes, sequences, and joint configurations.

        The UI is organized into a vertical layout containing a grid of buttons for keyframe and sequence operations, checkboxes for toggling options, entry fields for inputting parameters, and a horizontal layout for displaying keyframe and sequence lists alongside joint sliders.

        The function connects various UI elements to their respective event handlers to facilitate user interaction with the application.
        """
        layout = QVBoxLayout()
        # Top button grid
        grid_layout = QGridLayout()
        # Buttons
        add_button = QPushButton("Add Keyframe")
        add_button.clicked.connect(self.add_keyframe)
        grid_layout.addWidget(add_button, 0, 0)

        remove_button = QPushButton("Remove Keyframe")
        remove_button.clicked.connect(self.remove_keyframe)
        grid_layout.addWidget(remove_button, 0, 1)

        # load_button = QPushButton("Load Keyframe")
        # load_button.clicked.connect(self.load_keyframe)
        # grid_layout.addWidget(load_button, 0, 2)

        update_button = QPushButton("Update Keyframe")
        update_button.clicked.connect(self.update_keyframe)
        grid_layout.addWidget(update_button, 0, 2)

        test_button = QPushButton("Test Keyframe")
        test_button.clicked.connect(self.test_keyframe)
        grid_layout.addWidget(test_button, 0, 3)

        ground_button = QPushButton("Put on Ground")
        ground_button.clicked.connect(self.put_on_ground)
        grid_layout.addWidget(ground_button, 0, 4)

        # Mirror & Reverse Mirror Checkboxes
        self.mirror_checked = QCheckBox("Mirror")
        self.mirror_checked.setChecked(True)
        self.mirror_checked.toggled.connect(self.on_mirror_checked)
        grid_layout.addWidget(self.mirror_checked, 1, 5)

        self.rev_mirror_checked = QCheckBox("Rev. Mirror")
        self.rev_mirror_checked.toggled.connect(self.on_rev_mirror_checked)
        grid_layout.addWidget(self.rev_mirror_checked, 1, 6)

        self.collision_geom_checked = QCheckBox("Collsion Geom")
        self.collision_geom_checked.toggled.connect(self.on_collision_geom_checked)
        grid_layout.addWidget(self.collision_geom_checked, 1, 7)

        # Sequence Buttons
        add_to_sequence_button = QPushButton("Add to Sequence")
        add_to_sequence_button.clicked.connect(self.add_to_sequence)
        grid_layout.addWidget(add_to_sequence_button, 1, 0)

        remove_from_sequence_button = QPushButton("Remove from Sequence")
        remove_from_sequence_button.clicked.connect(self.remove_from_sequence)
        grid_layout.addWidget(remove_from_sequence_button, 1, 1)

        test_trajectory_button = QPushButton("Display Trajectory")
        test_trajectory_button.clicked.connect(self.test_trajectory)
        grid_layout.addWidget(test_trajectory_button, 1, 2)

        test_qpos_trajectory_button = QPushButton("Display Qpos Trajectory")
        test_qpos_trajectory_button.clicked.connect(self.test_qpos_trajectory)
        grid_layout.addWidget(test_qpos_trajectory_button, 1, 3)
        self.is_qpos_traj = False

        stop_trajectory_button = QPushButton("Stop Trajectory")
        stop_trajectory_button.clicked.connect(self.stop_trajectory)
        grid_layout.addWidget(stop_trajectory_button, 1, 4)

        # Physics Toggle
        self.physics_enabled = QCheckBox("Enable Physics")
        self.physics_enabled.setChecked(True)
        grid_layout.addWidget(self.physics_enabled, 2, 7)

        motion_name_label = QLabel("Motion Name:")
        grid_layout.addWidget(motion_name_label, 0, 5)
        self.motion_name_entry = QLineEdit(self.task_name)
        grid_layout.addWidget(self.motion_name_entry, 0, 6)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_data)
        grid_layout.addWidget(save_button, 0, 7)

        save_l_foot_btn = QPushButton("Save Left Foot Pose")
        save_l_foot_btn.clicked.connect(self.save_left_foot_pose)
        grid_layout.addWidget(save_l_foot_btn, 2, 0)

        apply_l_foot_btn = QPushButton("Apply Left Foot Pose")
        apply_l_foot_btn.clicked.connect(self.apply_left_foot_pose)
        grid_layout.addWidget(apply_l_foot_btn, 2, 1)

        save_r_foot_btn = QPushButton("Save Right Foot Pose")
        save_r_foot_btn.clicked.connect(self.save_right_foot_pose)
        grid_layout.addWidget(save_r_foot_btn, 2, 2)

        apply_r_foot_btn = QPushButton("Apply Right Foot Pose")
        apply_r_foot_btn.clicked.connect(self.apply_right_foot_pose)
        grid_layout.addWidget(apply_r_foot_btn, 2, 3)

        save_hand_btn = QPushButton("Save Left Hand Pose")
        save_hand_btn.clicked.connect(self.save_left_hand_pose)
        grid_layout.addWidget(save_hand_btn, 2, 4)

        apply_hand_btn = QPushButton("Apply Left Hand Pose")
        apply_hand_btn.clicked.connect(self.apply_left_hand_pose)
        grid_layout.addWidget(apply_hand_btn, 2, 5)

        self.relative_frame_checked = QCheckBox("Save in Robot Frame")
        self.relative_frame_checked.setChecked(
            True
        )  # default to saving in robot frame behavior
        grid_layout.addWidget(self.relative_frame_checked, 2, 6)
        self.is_relative_frame = self.relative_frame_checked.isChecked()

        button_frame = QWidget()
        button_frame.setLayout(grid_layout)
        layout.addWidget(button_frame)

        # Horizontal Layout for Keyframe & Sequence Lists
        hbox_layout = QHBoxLayout()

        viewport_container = QWidget.createWindowContainer(self.viewport)
        hbox_layout.addWidget(viewport_container, stretch=3)

        vbox_layout = QVBoxLayout()

        keyframe_label = QLabel("Keyframes:")
        vbox_layout.addWidget(keyframe_label)

        self.keyframe_listbox = QListWidget()
        self.keyframe_listbox.setSelectionMode(
            QListWidget.ExtendedSelection
        )  # Allow multi-selection using cmd/ctrl and shift
        self.keyframe_listbox.itemSelectionChanged.connect(self.on_keyframe_select)
        # Allow double click to rename
        self.keyframe_listbox.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self.keyframe_listbox.itemChanged.connect(self.on_keyframe_name_changed)
        # Allow drag and drop (works for multi-selection)
        self.keyframe_listbox.setDragEnabled(True)
        self.keyframe_listbox.setAcceptDrops(True)
        self.keyframe_listbox.setDropIndicatorShown(True)
        self.keyframe_listbox.setDragDropMode(QListWidget.InternalMove)
        self.keyframe_listbox.model().rowsMoved.connect(self.sync_keyframe_order)
        vbox_layout.addWidget(self.keyframe_listbox, stretch=1)

        sequence_label = QLabel("Sequence:")
        vbox_layout.addWidget(sequence_label)

        self.sequence_listbox = QListWidget()
        self.sequence_listbox.setSelectionMode(
            QListWidget.ExtendedSelection
        )  # Allow multi-selection using cmd/ctrl and shift
        self.sequence_listbox.itemSelectionChanged.connect(self.on_sequence_select)
        # Allow double click to edit arrival time
        self.sequence_listbox.setEditTriggers(QListWidget.EditTrigger.NoEditTriggers)
        self.sequence_listbox.itemDoubleClicked.connect(self.edit_sequence_arrival_time)
        # Allow drag and drop (works for multi-selection)
        self.sequence_listbox.setDragEnabled(True)
        self.sequence_listbox.setAcceptDrops(True)
        self.sequence_listbox.setDropIndicatorShown(True)
        self.sequence_listbox.setDragDropMode(QListWidget.InternalMove)
        self.sequence_listbox.model().rowsMoved.connect(self.sync_sequence_order)
        vbox_layout.addWidget(self.sequence_listbox, stretch=1)

        list_frame = QWidget()
        list_frame.setLayout(vbox_layout)
        hbox_layout.addWidget(list_frame, stretch=1)

        # Joint Sliders
        joint_sliders_layout = QGridLayout()
        slider_columns = 2  # Columns for sliders
        for i in range(slider_columns):
            joint_sliders_layout.setColumnStretch(i * 3, 1)
            joint_sliders_layout.setColumnStretch(i * 3 + 1, 2)
            joint_sliders_layout.setColumnStretch(i * 3 + 2, 1)

        self.joint_sliders = {}
        self.joint_labels = {}
        self.normalized_range = (-2000, 2000)  # Fixed range for all sliders

        reordered_list = []
        # Separate left and right joints
        for joint in robot.joint_ordering:
            if "left" in joint:
                right_joint = joint.replace("left", "right")
                assert right_joint in robot.joint_ordering, f"{right_joint} not found!"
                reordered_list.append(joint)
                reordered_list.append(right_joint)
            elif "right" not in joint:
                reordered_list.append(joint)

        num_sliders = 0
        for joint_name in reordered_list:
            joint_range = robot.joint_limits[joint_name]

            row = num_sliders // slider_columns
            col = num_sliders % slider_columns

            label = QLabel(joint_name)
            joint_sliders_layout.addWidget(label, row, col * 3)

            # Scale value label (to display the current value)
            # value_label = QLabel(text="0.00")
            # QLineEdit is suprisingly faster than QLabel
            value_label = QLineEdit("0.00")

            slider = QSlider(Qt.Horizontal)

            slider.setMinimum(self.normalized_range[0])
            slider.setMaximum(self.normalized_range[1])
            slider.setValue(
                int(
                    np.interp(
                        robot.default_joint_angles[joint_name],
                        joint_range,
                        self.normalized_range,
                    )
                )
            )

            slider.setTickPosition(QSlider.TicksBelow)
            slider.setSingleStep(1)

            value_label.returnPressed.connect(
                partial(self.on_joint_label_change, joint_name)
            )
            joint_sliders_layout.addWidget(value_label, row, col * 3 + 2)
            # slider.sliderReleased.connect(
            #     partial(self.on_joint_slider_release, joint_name)
            # )
            # Visualize robot pose even during slider movement
            slider.valueChanged.connect(
                partial(self.on_joint_slider_moving, joint_name)
            )
            joint_sliders_layout.addWidget(slider, row, col * 3 + 1)

            self.joint_sliders[joint_name] = slider
            self.joint_labels[joint_name] = value_label
            num_sliders += 1

        joint_sliders_frame = QWidget()
        joint_sliders_frame.setLayout(joint_sliders_layout)
        hbox_layout.addWidget(joint_sliders_frame, stretch=4)

        horizontal_frame = QWidget()
        horizontal_frame.setLayout(hbox_layout)
        layout.addWidget(horizontal_frame)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(1280, 720)

    def test_qpos_trajectory(self):
        if len(self.sequence_list) < 2:
            self.show_warning("Need at least 2 keyframes in sequence to preview.")
            return

        # Find selected start index
        start_idx = 0
        if self.sequence_listbox.selectedItems():
            start_idx = self.sequence_listbox.currentRow()

        # Extract full qpos and arrival times
        qpos_list = []
        times = []
        for keyframe_name, arrival_time in self.sequence_list:
            for keyframe in self.keyframes:
                if keyframe_name == keyframe.name:
                    qpos_list.append(keyframe.qpos)
                    times.append(arrival_time)
                    break

        if np.any(np.diff(times) <= 0):
            self.show_warning("The arrival times are not sorted correctly!")
            return

        times = np.array(times) - times[0]
        self.traj_times = np.arange(0, times[-1], self.dt)

        # Interpolate full qpos over time
        qpos_arr = np.array(qpos_list)
        qpos_traj = []

        # Start from selected keyframe’s time
        traj_start = int(np.searchsorted(self.traj_times, times[start_idx]))

        for t in self.traj_times:
            if t < times[-1]:
                qpos_t = interpolate_action(t, times, qpos_arr)
            else:
                qpos_t = qpos_arr[-1]
            qpos_traj.append(qpos_t)

        self.is_qpos_traj = True

        self.is_relative_frame = self.relative_frame_checked.isChecked()

        self.sim_thread.request_trajectory_test(
            qpos_list[start_idx],
            qpos_traj[traj_start:],
            self.dt,
            physics_enabled=False,
            is_qpos_traj=self.is_qpos_traj,
            is_relative_frame=self.is_relative_frame,
        )

    @Slot()
    def stop_trajectory(self):
        """Stops the ongoing trajectory execution."""
        with QMutexLocker(self.sim_thread.mutex):
            self.sim_thread.is_testing = False
        print("Trajectory execution stopped by user.")

    def save_left_foot_pose(self):
        with QMutexLocker(self.mutex):
            self.saved_left_foot_pose = self.sim.get_site_transform("left_foot_center")
        print("Saved left foot pose!")

    def save_right_foot_pose(self):
        with QMutexLocker(self.mutex):
            self.saved_right_foot_pose = self.sim.get_site_transform(
                "right_foot_center"
            )
        print("Saved right foot pose!")

    def apply_left_foot_pose(self):
        if self.saved_left_foot_pose is None:
            print("No saved left foot pose.")
            return
        self.align_foot_to_pose("left_foot_center", self.saved_left_foot_pose)

    def apply_right_foot_pose(self):
        if self.saved_right_foot_pose is None:
            print("No saved right foot pose.")
            return
        self.align_foot_to_pose("right_foot_center", self.saved_right_foot_pose)

    def align_foot_to_pose(self, foot_site_name, target_pose):
        with QMutexLocker(self.mutex):
            torso_t_curr = self.sim.get_body_transform("torso")
            foot_t_curr = self.sim.get_site_transform(foot_site_name)

            aligned_torso_t = target_pose @ np.linalg.inv(foot_t_curr) @ torso_t_curr

            self.sim.data.qpos[:3] = aligned_torso_t[:3, 3]
            self.sim.data.qpos[3:7] = R.from_matrix(aligned_torso_t[:3, :3]).as_quat(
                scalar_first=True
            )
            self.sim.forward()
        print(f"{foot_site_name} aligned to saved pose.")

    def align_hand_to_pose(self, hand_site_name, target_pose):
        with QMutexLocker(self.mutex):
            torso_t_curr = self.sim.get_body_transform("torso")
            hand_t_curr = self.sim.get_site_transform(hand_site_name)

            aligned_torso_t = target_pose @ np.linalg.inv(hand_t_curr) @ torso_t_curr

            self.sim.data.qpos[:3] = aligned_torso_t[:3, 3]
            self.sim.data.qpos[3:7] = R.from_matrix(aligned_torso_t[:3, :3]).as_quat(
                scalar_first=True
            )
            self.sim.forward()
        print(f"{hand_site_name} aligned to saved pose.")

    def save_left_hand_pose(self):
        with QMutexLocker(self.mutex):
            self.saved_left_hand_pose = self.sim.get_site_transform("left_hand_center")
        print("Saved left hand pose!")

    def apply_left_hand_pose(self):
        if self.saved_left_hand_pose is None:
            print("No saved left hand pose.")
            return
        self.align_hand_to_pose("left_hand_center", self.saved_left_hand_pose)

    def add_keyframe(self):
        """Adds a new keyframe to the keyframe list, ensuring a unique name.

        If a keyframe is selected in the listbox, a deep copy of the selected keyframe is created. The new keyframe's name is updated if it contains "default". The new keyframe is then appended to the keyframe list and displayed in the listbox.

        Attributes:
            keyframe_listbox (QListWidget): The listbox widget displaying keyframes.
            keyframes (list): A list of keyframe objects.
            motion_name_entry (QLineEdit): The input field for the motion name.

        """
        idx = -1
        if self.keyframe_listbox.selectedItems():
            idx = self.keyframe_listbox.currentRow()
            original_name = self.keyframes[idx].name
            base_name = original_name

            # Collect all existing copy names of the form original_copy_N
            existing_suffixes = []
            for kf in self.keyframes:
                if kf.name.startswith(base_name + "_"):
                    suffix_str = kf.name[len(base_name) + 1 :]
                    if suffix_str.isdigit():
                        existing_suffixes.append(int(suffix_str))

            next_suffix = max(existing_suffixes, default=0) + 1
            new_name = f"{base_name}_{next_suffix}"

            new_keyframe = copy.deepcopy(self.keyframes[idx])
            new_keyframe.name = new_name
        else:
            new_keyframe = Keyframe(
                name="default",
                motor_pos=np.array(
                    list(self.robot.default_motor_angles.values()), dtype=np.float32
                ),
                joint_pos=np.array(
                    list(self.robot.default_joint_angles.values()), dtype=np.float32
                ),
                qpos=self.sim.home_qpos.copy(),
            )

        self.keyframes.append(new_keyframe)
        item = QListWidgetItem(new_keyframe.name)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.keyframe_listbox.addItem(item)

    def on_keyframe_name_changed(self, item):
        row = self.keyframe_listbox.row(item)
        new_name = item.text().strip()

        if not new_name:
            self.show_warning("Keyframe name cannot be empty.")
            item.setText(self.keyframes[row].name)
            return

        # Prevent duplicates
        for i, kf in enumerate(self.keyframes):
            if i != row and kf.name == new_name:
                self.show_warning(f"Duplicate name '{new_name}' already exists.")
                item.setText(self.keyframes[row].name)
                return

        old_name = self.keyframes[row].name
        self.keyframes[row].name = new_name

        # Update all sequence entries using this keyframe name
        for i, (name, t) in enumerate(self.sequence_list):
            if name == old_name:
                self.sequence_list[i] = (new_name, t)

        # Update sequence list UI in-place (if needed)
        self.sequence_listbox.blockSignals(True)
        self.update_sequence_listbox()
        self.sequence_listbox.blockSignals(False)

    def sync_keyframe_order(self, parent, start, end, dest, row):
        """Update internal keyframe list to match QListWidget order."""
        items = []
        for i in range(self.keyframe_listbox.count()):
            items.append(self.keyframe_listbox.item(i).text())

        reordered = []
        for name in items:
            for kf in self.keyframes:
                if kf.name == name:
                    reordered.append(kf)
                    break

        if len(reordered) == len(self.keyframes):
            self.keyframes = reordered
            print("Keyframe order updated.")
        else:
            print("[Warning] Mismatch in keyframe reordering logic.")

    def remove_keyframe(self):
        """Removes the currently selected keyframe(s) and their associated sequences from the lists.

        This method retrieves all selected keyframes from the keyframe listbox and removes them from the `keyframes` list.
        It also removes any corresponding entries in `sequence_list`.
        After removal, it updates the sequence and keyframe listboxes to reflect the changes.
        """
        selected_items = self.keyframe_listbox.selectedItems()

        if not selected_items:
            return  # No keyframes selected

        # Get selected indices in descending order to avoid shifting issues
        selected_indices = sorted(
            [self.keyframe_listbox.row(item) for item in selected_items], reverse=True
        )

        for index in selected_indices:
            keyframe = self.keyframes[index]

            # Remove associated sequence entry
            self.sequence_list = [
                (name, arrival_time)
                for name, arrival_time in self.sequence_list
                if name != keyframe.name
            ]

            # Remove keyframe
            self.keyframes.pop(index)

        self.selected_keyframe -= 1

        # Update UI elements
        self.update_sequence_listbox()
        self.update_keyframe_listbox()

    def load_keyframe(self):
        """Loads the currently selected keyframe and updates the simulation thread's position.

        This method checks if the object has a `selected_keyframe` attribute. If it does, it retrieves the keyframe from the `keyframes` list using the `selected_keyframe` index and updates the simulation thread's position (`qpos`) with the keyframe's position data.

        """
        if hasattr(self, "selected_keyframe"):
            keyframe = self.keyframes[self.selected_keyframe]
            self.sim_thread.update_qpos(keyframe.qpos)

    def update_keyframe(self):
        """Updates the keyframe by requesting state data if a keyframe is selected.

        This method checks if the object has an attribute `selected_keyframe`. If it does, it triggers a request for state data from the simulation thread, which is expected to update or refresh the keyframe data accordingly.
        """
        if hasattr(self, "selected_keyframe"):
            self.sim_thread.request_state_data()

    @Slot(np.ndarray, np.ndarray, np.ndarray)
    def update_keyframe_with_signal(self, motor_pos, joint_pos, qpos):
        """Updates the selected keyframe with new motor angles, joint angles, and position data.

        Args:
            motor_pos (np.ndarray): The new motor positions to update the keyframe with.
            joint_pos (np.ndarray): The new joint positions to update the keyframe with.
            qpos (np.ndarray): The new position data to update the keyframe with.

        """
        if hasattr(self, "selected_keyframe"):
            idx = self.selected_keyframe
            self.keyframes[idx].motor_pos = motor_pos.copy()
            self.keyframes[idx].joint_pos = joint_pos.copy()
            self.keyframes[idx].qpos = qpos.copy()
            print(f"Keyframe {idx} updated!")

    def test_keyframe(self):
        """Tests a selected keyframe by sending a request to the simulation thread.

        This method checks if a keyframe is selected and, if so, retrieves it from the
        keyframes list. It then parses the time delta from the user interface and sends
        a request to the simulation thread to test the keyframe with the specified time
        delta.

        """
        if hasattr(self, "selected_keyframe"):
            keyframe = self.keyframes[self.selected_keyframe]
            self.sim_thread.request_keyframe_test(keyframe, self.dt)

    def put_on_ground(self):
        """Requests the simulation thread to place the feet on the ground.

        This method sends a request to the simulation thread to ensure that the feet are positioned on the ground. It acts as an interface to the simulation's functionality for managing the feet's contact with the ground.
        """
        self.sim_thread.request_on_ground()

    def on_mirror_checked(self, checked):
        """Sets the reverse mirror checkbox to unchecked if the mirror checkbox is checked.

        Args:
            checked (bool): The current state of the mirror checkbox. If True, the reverse mirror checkbox will be unchecked.
        """
        if checked:
            self.rev_mirror_checked.setChecked(False)

    def on_rev_mirror_checked(self, checked):
        """Handles the event when the reverse mirror checkbox is checked.

        If the reverse mirror checkbox is checked, this function will uncheck the
        standard mirror checkbox to ensure only one of the mirror options is selected
        at a time.

        Args:
            checked (bool): The state of the reverse mirror checkbox. True if checked,
            False otherwise.
        """
        if checked:
            self.mirror_checked.setChecked(False)

    def on_collision_geom_checked(self, checked):
        if checked:
            self.viewport.opt.geomgroup[3] = 1
            self.viewport.opt.geomgroup[2] = 0
        else:
            self.viewport.opt.geomgroup[3] = 0
            self.viewport.opt.geomgroup[2] = 1

    def add_to_sequence(self):
        """Adds the selected keyframe or sequences to the sequence list with proper timing.

        If less than one sequence is selected in the sequence listbox, this method retrieves the selected keyframe's name and index,
        constructs a unique keyframe name, and appends it along with the specified arrival time to the sequence list.

        If multiple sequences are selected, instead of adding a keyframe, the selected sequences are appended to the end
        of the sequence list while preserving the relative time gaps between them. The arrival time of the first selected
        sequence is used as the starting reference, and the subsequent sequences maintain their original spacing.
        """
        selected_sequences = self.sequence_listbox.selectedItems()

        if len(selected_sequences) <= 1:
            # Normal behavior: Add keyframe if less than one sequence is selected
            if hasattr(self, "selected_keyframe"):
                keyframe = self.keyframes[self.selected_keyframe]
                self.sequence_list.append((keyframe.name, 0.0))
                self.update_sequence_listbox()
            return

        # Handle multiple selected sequences
        selected_indices = sorted(
            [self.sequence_listbox.row(item) for item in selected_sequences]
        )

        # Use first selected sequence's arrival time
        arrival_time = 0.0
        # Calculate time gap between sequences
        time_gaps = [
            self.sequence_list[selected_indices[i + 1]][1]
            - self.sequence_list[selected_indices[i]][1]
            for i in range(len(selected_indices) - 1)
        ]

        # Append selected sequences with adjusted times
        for i, index in enumerate(selected_indices):
            name, _ = self.sequence_list[index]
            sequence_entry = (name, arrival_time)
            self.sequence_list.append(sequence_entry)
            if i < len(time_gaps):
                arrival_time += time_gaps[i]

        self.update_sequence_listbox()

    def sync_sequence_order(self, parent, start, end, dest, row):
        """Update internal sequence_list to match QListWidget order."""
        new_sequence = []
        for i in range(self.sequence_listbox.count()):
            text = self.sequence_listbox.item(i).text()
            name, time_str = text.split("    t=")
            arrival_time = float(time_str.strip())
            new_sequence.append((name.strip(), arrival_time))

        if len(new_sequence) == len(self.sequence_list):
            self.sequence_list = new_sequence
            print("Sequence order updated.")
        else:
            print("[Warning] Sequence list size mismatch after drag.")

    def edit_sequence_arrival_time(self, item):
        row = self.sequence_listbox.row(item)
        name, prev_time_str = item.text().split("    t=")
        name = name.strip()
        old_time = float(prev_time_str.strip())

        new_time, ok = QInputDialog.getDouble(
            self,
            "Edit Arrival Time",
            f"Set arrival time for '{name}':",
            old_time,
            decimals=4,
        )

        if ok:
            delta = new_time - old_time
            for i in range(row, len(self.sequence_list)):
                self.sequence_list[i] = (
                    self.sequence_list[i][0],
                    self.sequence_list[i][1] + delta,
                )
            self.update_sequence_listbox()

    def remove_from_sequence(self):
        """Removes the currently selected sequence(s) from the sequence list and updates the listbox.

        This method retrieves all selected sequences from the sequence listbox and removes them from
        `sequence_list`. After removal, it updates the sequence listbox to reflect the changes.
        """
        selected_items = self.sequence_listbox.selectedItems()

        if not selected_items:
            return  # No sequences selected

        # Get selected indices in descending order to avoid shifting issues
        selected_indices = sorted(
            [self.sequence_listbox.row(item) for item in selected_items], reverse=True
        )

        # Remove selected sequences
        for index in selected_indices:
            self.sequence_list.pop(index)

        # Update UI elements
        self.update_sequence_listbox()

    def test_trajectory(self):
        """Tests the trajectory of a sequence of keyframes by interpolating motor positions over time and initiating a simulation.

        This method extracts motor positions and arrival times from a sequence of keyframes, checks for correct sorting of arrival times, and interpolates motor positions to create a trajectory. It then requests a simulation test of the trajectory starting from a specified keyframe.

        Raises:
            Warning: If the arrival times are not sorted in ascending order.

        """
        # Extract positions and arrival times from the sequence
        start_idx = 0
        if self.sequence_listbox.selectedItems():
            start_idx = self.sequence_listbox.currentRow()

        action_list = []
        qpos_list = []
        times = []
        for keyframe_name, arrival_time in self.sequence_list:
            for keyframe in self.keyframes:
                if keyframe_name == keyframe.name:
                    action_list.append(keyframe.motor_pos)
                    qpos_list.append(keyframe.qpos)
                    times.append(arrival_time)
                    break

        if np.any(np.diff(times) <= 0):
            self.show_warning("The arrival times are not sorted correctly!")
            return

        qpos_start = qpos_list[start_idx]
        enabled = self.physics_enabled.isChecked()

        action_arr = np.array(action_list)
        times = np.array(times) - times[0]

        self.traj_times = np.array([t for t in np.arange(0, times[-1], self.dt)])
        self.action_traj = []
        for t in self.traj_times:
            if t < times[-1]:
                motor_pos = interpolate_action(t, times, action_arr)
            else:
                motor_pos = action_arr[-1]

            self.action_traj.append(motor_pos)

        traj_start = int(np.searchsorted(self.traj_times, times[start_idx]))

        self.is_qpos_traj = False

        self.is_relative_frame = self.relative_frame_checked.isChecked()

        self.sim_thread.request_trajectory_test(
            qpos_start,
            self.action_traj[traj_start:],
            self.dt,
            enabled,
            self.is_qpos_traj,
            self.is_relative_frame,
        )

    @Slot()
    def update_traj_with_signal(
        self,
        qpos_replay,
        motor_vel_replay,
        joint_vel_replay,
        body_pos_replay,
        body_quat_replay,
        body_lin_vel_replay,
        body_ang_vel_replay,
        site_pos_replay,
        site_quat_replay,
    ):
        """Updates the body pose trajectories with the provided signals."""
        self.qpos_replay = qpos_replay
        self.motor_vel_replay = motor_vel_replay
        self.joint_vel_replay = joint_vel_replay
        self.body_pos_replay = body_pos_replay
        self.body_quat_replay = body_quat_replay
        self.body_lin_vel_replay = body_lin_vel_replay
        self.body_ang_vel_replay = body_ang_vel_replay
        self.site_pos_replay = site_pos_replay
        self.site_quat_replay = site_quat_replay

    def save_data(self):
        """Saves the current motion data and keyframes to specified directories in lz4 format.

        This method serializes the current state of keyframes, sequences, and trajectories into a dictionary and saves it as a lz4 file in the results directory. It also saves the motion data to a separate file in the 'motion' directory, with a prompt to confirm overwriting if the file already exists.

        """
        result_dict = {}
        saved_keyframes = []
        for keyframe in self.keyframes:
            saved_keyframes.append(asdict(keyframe))

        result_dict["time"] = self.traj_times
        result_dict["qpos"] = np.array(
            self.qpos_replay
        )  # Contains global torso position in qpos[:3]
        result_dict["motor_vel"] = np.array(self.motor_vel_replay)
        result_dict["joint_vel"] = np.array(self.joint_vel_replay)
        result_dict["body_pos"] = np.array(self.body_pos_replay)
        result_dict["body_quat"] = np.array(self.body_quat_replay)
        result_dict["body_lin_vel"] = np.array(self.body_lin_vel_replay)
        result_dict["body_ang_vel"] = np.array(self.body_ang_vel_replay)
        result_dict["site_pos"] = np.array(self.site_pos_replay)
        result_dict["site_quat"] = np.array(self.site_quat_replay)

        if self.is_qpos_traj:
            result_dict["action"] = None
        else:
            result_dict["action"] = np.array(self.action_traj)

        result_dict["keyframes"] = saved_keyframes  # Keep as list of dicts
        result_dict["timed_sequence"] = self.sequence_list  # Keep as list of tuples
        result_dict["is_robot_relative_frame"] = self.is_relative_frame

        time_str = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.result_dir, f"{self.task_name}_{time_str}.lz4")
        print(f"Saving the results to {result_path}")
        joblib.dump(result_dict, result_path, compress="lz4")

        motion_name = self.motion_name_entry.text()
        motion_file_path = os.path.join("motion", f"{motion_name}.lz4")
        # Check if file exists before saving
        if os.path.exists(motion_file_path):
            reply = QMessageBox.question(
                self,
                "Overwrite Confirmation",
                "The file is already saved in the results folder."
                + f" Do you want to update {motion_name}.lz4 in the motion/ directory?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return  # User canceled, do not save

        # Proceed with saving
        print(f"Saving the results to {motion_file_path}...")
        joblib.dump(result_dict, motion_file_path, compress="lz4")
        print("The motion data is saved!")

    def load_data(self):
        """Loads and processes keyframe data from a specified file path.

        This method initializes and clears existing keyframe and trajectory data, then attempts to load new data from a file specified by `self.data_path`. If the file contains valid data, it updates the keyframes, sequence list, and trajectory information. If the data is incompatible with the current robot configuration, it stops the simulation and raises an error. If no keyframes are found, it initializes a default keyframe.

        Raises:
            ValueError: If the loaded data is incompatible with the current robot configuration.

        """
        self.keyframes = []
        self.sequence_list = []
        self.qpos_replay = []
        self.motor_vel_replay = []
        self.joint_vel_replay = []
        self.body_pos_replay = []
        self.body_quat_replay = []
        self.body_lin_vel_replay = []
        self.body_ang_vel_replay = []
        self.site_pos_replay = []
        self.site_quat_replay = []
        self.keyframe_listbox.clear()

        keyframes = []
        if len(self.data_path) > 0:
            print(f"Loading inputs from {self.data_path}")
            data = joblib.load(self.data_path)

            if isinstance(data, dict):
                motion_file = os.path.basename(self.data_path)
                keyframes = [Keyframe(**k) for k in data.get("keyframes", [])]
                if "2xc" in self.robot.name and "2xc" not in motion_file:
                    right_hip_roll_idx = self.robot.joint_ordering.index(
                        "right_hip_roll"
                    )
                    mj_right_hip_roll_id = mujoco.mj_name2id(
                        self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll"
                    )
                    for kf in keyframes:
                        kf.joint_pos[right_hip_roll_idx] *= -1
                        kf.motor_pos[right_hip_roll_idx] *= -1
                        kf.qpos[self.sim.model.jnt_qposadr[mj_right_hip_roll_id]] *= -1
                elif "2xm" in self.robot.name and "2xm" not in motion_file:
                    right_hip_roll_idx = self.robot.joint_ordering.index(
                        "right_hip_roll"
                    )
                    mj_right_hip_roll_id = mujoco.mj_name2id(
                        self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll"
                    )
                    for kf in keyframes:
                        kf.joint_pos[right_hip_roll_idx] *= -1
                        kf.motor_pos[right_hip_roll_idx] *= -1
                        kf.qpos[self.sim.model.jnt_qposadr[mj_right_hip_roll_id]] *= -1

                self.sequence_list = data.get("timed_sequence", [])
                for i, (name, arrival_time) in enumerate(self.sequence_list):
                    self.sequence_list[i] = (name.replace(" ", "_"), arrival_time)

                self.traj_times = data.get("time", [])
                self.action_traj = data.get("action", [])

                self.update_sequence_listbox()

            else:
                keyframes = data

            if len(keyframes[0].motor_pos) != self.robot.nu:
                self.sim_thread.stop()
                raise ValueError(
                    "This data is saved for a different robot! Consider changing the robot name."
                )

        if len(keyframes) == 0:
            # Create default keyframe when no keyframes exist
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
            item = QListWidgetItem("default")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.keyframe_listbox.addItem(item)
        else:
            for i, keyframe in enumerate(keyframes):
                # Overwrite "default" keyframe to ensure consistent default pose across motion files
                if keyframe.name == "default":
                    keyframe = Keyframe(
                        name="default",
                        motor_pos=np.array(
                            list(self.robot.default_motor_angles.values()),
                            dtype=np.float32,
                        ),
                        joint_pos=np.array(
                            list(self.robot.default_joint_angles.values()),
                            dtype=np.float32,
                        ),
                        qpos=self.sim.home_qpos.copy(),
                    )
                self.keyframes.append(keyframe)
                item = QListWidgetItem(f"{keyframe.name}")
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.keyframe_listbox.addItem(item)

    def on_keyframe_select(self):
        """Handles the event when a keyframe is selected from the listbox.

        This method checks if there are any selected items in the keyframe listbox.
        If a keyframe is selected, it updates the `selected_keyframe` attribute with
        the index of the currently selected keyframe and calls the `load_keyframe`
        method to load the selected keyframe's data.
        Also, the selected keyframe's joint angles are applied to the UI.
        """
        if self.keyframe_listbox.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)  # Show "loading" cursor

            current_row = self.keyframe_listbox.currentRow()
            if current_row < 0 or current_row >= len(self.keyframes):
                QApplication.restoreOverrideCursor()
                return

            self.selected_keyframe = current_row
            self.load_keyframe()

            if hasattr(self, "selected_sequence"):
                del self.selected_sequence

            keyframe = self.keyframes[self.selected_keyframe]

            # Block signals temporarily to prevent excessive UI refreshes
            for joint_name in self.robot.joint_ordering:
                self.joint_sliders[joint_name].blockSignals(True)

            # Apply the keyframe's joint angles to the UI (sliders & labels)
            for joint_name, value in zip(self.robot.joint_ordering, keyframe.joint_pos):
                joint_range = self.robot.joint_limits[joint_name]

                slider_value = int(np.interp(value, joint_range, self.normalized_range))

                # Update UI
                self.joint_sliders[joint_name].setValue(slider_value)
                self.joint_labels[joint_name].setText(f"{value:.2f}")

            # Re-enable signals after bulk update
            for joint_name in self.robot.joint_ordering:
                self.joint_sliders[joint_name].blockSignals(False)

            # Force UI to reflect changes instantly
            QApplication.processEvents()
            # self.repaint()

            QApplication.restoreOverrideCursor()  # Restore normal cursor

    def on_sequence_select(self):
        """Selects the current sequence from the listbox if any item is selected.

        This method checks if there are any selected items in the sequence listbox. If there are, it updates the `selected_sequence` attribute with the index of the currently selected item.
        """
        if self.sequence_listbox.selectedItems():
            self.selected_sequence = self.sequence_listbox.currentRow()
            if hasattr(self, "selected_keyframe"):
                del self.selected_keyframe

    def update_keyframe_listbox(self):
        """Updates the keyframe listbox with the current keyframes.

        This method clears the existing items in the keyframe listbox and repopulates it with the current keyframes.
        """
        self.keyframe_listbox.clear()
        for keyframe in self.keyframes:
            item = QListWidgetItem(f"{keyframe.name}")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.keyframe_listbox.addItem(item)

    def update_sequence_listbox(self):
        """Updates the sequence listbox with formatted items from the sequence list.

        This method clears the current contents of the sequence listbox and repopulates it with items from the `sequence_list`. Each item in the listbox is formatted to replace spaces in the name with underscores and includes the arrival time.
        """
        self.sequence_listbox.clear()
        # self.sequence_list.sort(key=lambda x: x[1])
        for name, arrival_time in self.sequence_list:
            item = QListWidgetItem(f"{name.replace(' ', '_')}    t={arrival_time}")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.sequence_listbox.addItem(item)

    def update_joint_pos(self, joint_name, value):
        """Updates the position of a specified joint and its mirrored counterpart if applicable.

        This method updates the angle of a given joint and, if mirror options are enabled,
        also updates the angle of its mirrored joint. The mirrored joint's angle is calculated
        based on the mirror and reverse mirror settings.

        Args:
            joint_name (str): The name of the joint to update.
            value (float): The new angle value for the specified joint.

        Returns:
            dict: A dictionary containing the updated joint angles, including any mirrored joints.
        """
        joint_angles_to_update = {joint_name: value}

        mirror_checked, rev_mirror_checked = (
            self.mirror_checked.isChecked(),
            self.rev_mirror_checked.isChecked(),
        )
        if mirror_checked or rev_mirror_checked:
            if "left" in joint_name or "right" in joint_name:
                mirrored_joint_name = (
                    joint_name.replace("left", "right")
                    if "left" in joint_name
                    else joint_name.replace("right", "left")
                )
                mirror_sign = (
                    self.mirror_joint_signs[joint_name]
                    if "left" in joint_name
                    else self.mirror_joint_signs[mirrored_joint_name]
                )
                joint_angles_to_update[mirrored_joint_name] = (
                    mirror_checked * value * mirror_sign
                    - rev_mirror_checked * value * mirror_sign
                )

        self.sim_thread.update_joint_angles(joint_angles_to_update)

        return joint_angles_to_update

    def on_joint_slider_release(self, joint_name):
        """Handles the event when a joint slider is released, updating the joint's position and reflecting the changes in the UI.

        Args:
            joint_name (str): The name of the joint associated with the slider that was released.

        Updates the joint's position based on the slider's value, adjusts the corresponding label to display the new value, and updates any other affected joint sliders and labels to reflect their new positions.
        """
        slider = self.joint_sliders[joint_name]
        value_label = self.joint_labels[joint_name]

        joint_range = self.robot.joint_limits[joint_name]
        slider_value = np.interp(slider.value(), self.normalized_range, joint_range)

        joint_angles_to_update = self.update_joint_pos(joint_name, slider_value)

        value_label.setText(f"{slider_value:.2f}")

        for name, value in joint_angles_to_update.items():
            if name != joint_name:
                self.joint_labels[name].setText(f"{value:.2f}")

                joint_range = self.robot.joint_limits[name]
                self.joint_sliders[name].setValue(
                    int(np.interp(value, joint_range, self.normalized_range))
                )

    def on_joint_slider_moving(self, joint_name, value):
        slider = self.joint_sliders[joint_name]
        joint_range = self.robot.joint_limits[joint_name]
        slider_value = np.interp(slider.value(), self.normalized_range, joint_range)

        # Compute motor updates (main + mirrored if mirror/rev-mirror enabled)
        joint_angles_to_update = self.update_joint_pos(joint_name, slider_value)

        for name, angle in joint_angles_to_update.items():
            # Update label
            self.joint_labels[name].setText(f"{angle:.2f}")

            # Update slider visually only if this is a different joint
            if name != joint_name:
                joint_range = self.robot.joint_limits[name]
                slider_value = int(np.interp(angle, joint_range, self.normalized_range))
                mirror_slider = self.joint_sliders[name]

                # Block signals to prevent another on_joint_slider_moving from being called
                mirror_slider.blockSignals(True)
                mirror_slider.setValue(slider_value)
                mirror_slider.blockSignals(False)

    def on_joint_label_change(self, joint_name):
        """Handles changes to the joint label by validating and updating the joint's position.

        This method is triggered when the text in a joint's label is changed. It attempts to convert the text to a float and checks if the value is within the joint's allowable range. If the value is valid, it updates the joint's position and adjusts the corresponding slider and labels. If the value is invalid or out of range, it displays a warning and resets the label to "0.00".

        Args:
            joint_name (str): The name of the joint whose label has changed.

        """
        slider = self.joint_sliders[joint_name]
        value_label = self.joint_labels[joint_name]

        text = value_label.text()
        try:
            text_value = float(text)  # Convert input to float
        except ValueError:
            self.show_warning("The input value is not a valid number!")
            value_label.setText("0.00")
            return

        joint_range = self.robot.joint_limits[joint_name]  # Get joint range
        if not (joint_range[0] <= text_value <= joint_range[1]):
            self.show_warning(
                f"The input value {text_value} is out of range [{joint_range[0]:.2f}, {joint_range[1]:.2f}]!"
            )
            value_label.setText("0.00")
            return

        joint_angles_to_update = self.update_joint_pos(joint_name, text_value)

        slider.setValue(int(np.interp(text_value, joint_range, self.normalized_range)))

        for name, value in joint_angles_to_update.items():
            if name != joint_name:
                self.joint_labels[name].setText(f"{value:.2f}")

                joint_range = self.robot.joint_limits[name]
                self.joint_sliders[name].setValue(
                    int(np.interp(value, joint_range, self.normalized_range))
                )

    def show_warning(self, message, title="Warning"):
        """Displays a warning message box with a specified message and title.

        Args:
            message (str): The warning message to be displayed in the message box.
            title (str, optional): The title of the message box window. Defaults to "Warning".
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo Keyframe Editor.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="push_up",
        help="The name of the task.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="The path of the keyframes. If not provided, a new folder will be created."
        + "If the same as the task name, a copy of the data in the motion folder will be created in the results folder."
        + "Othewerwise, please make sure this is a valid folder name inside the results folder.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Scene to load (e.g., 'scene', 'scene_climb_up_box').",
    )
    args = parser.parse_args()

    app = QApplication()

    robot = Robot(args.robot)
    if len(args.scene) > 0:
        xml_path = os.path.join(
            "toddlerbot", "descriptions", robot.name, args.scene + ".xml"
        )
    else:
        xml_path = ""
    sim = MuJoCoSim(robot, xml_path=xml_path, vis_type="none")

    window = MuJoCoApp(sim, robot, args.task, args.run_name)
    window.show()
    app.exec()
    window.sim_thread.stop()
