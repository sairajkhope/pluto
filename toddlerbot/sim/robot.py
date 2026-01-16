"""Robot configuration and kinematics for ToddlerBot.

Defines the Robot class with motor configurations, joint limits,
forward/inverse kinematics, and transmission mappings.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy
import yaml

from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class Robot:
    """This class defines some data strucutres, FK, IK of ToddlerBot."""

    def __init__(self, robot_name: str):
        """Initializes a robot with specified configurations and paths.

        Args:
            robot_name (str): The name of the robot, used to set up directory paths and configurations.
        """
        self.name = robot_name
        description_dir = os.path.join("toddlerbot", "descriptions")

        global_config_path = os.path.join(description_dir, "default.yml")
        with open(global_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
            return d

        robot_config_path = os.path.join(description_dir, self.name, "robot.yml")
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
            deep_update(self.config, robot_config)

        motor_config_path = os.path.join(description_dir, self.name, "motors.yml")
        if os.path.exists(motor_config_path):
            with open(motor_config_path, "r") as f:
                motor_config = yaml.safe_load(f)
                if motor_config is not None:
                    deep_update(self.config, motor_config)

        self.hand_name = ""
        if "hand_name" in self.config["robot"]:
            self.hand_name = self.config["robot"]["hand_name"]

        self.foot_name = ""
        if "foot_name" in self.config["robot"]:
            self.foot_name = self.config["robot"]["foot_name"]

        self.has_gripper = "gripper" in self.hand_name

        robot_xml_path = os.path.join(
            description_dir, robot_name, f"{robot_name}_fixed.xml"
        )
        robot_tree = ET.parse(robot_xml_path)
        robot_root = robot_tree.getroot()

        self.motor_ordering: List[str] = []
        gripper_motor_names = []
        for motor_name in self.config["motors"]:
            if robot_root.find(f".//joint[@name='{motor_name}']") is not None:
                if "gripper" in motor_name:
                    gripper_motor_names.append(motor_name)

                self.motor_ordering.append(motor_name)

        self.neck_passive_dof_names = ["neck_pitch_front", "neck_pitch_back"]

        self.gripper_dof_names = []
        for motor_name in gripper_motor_names:
            self.gripper_dof_names.append(motor_name)
            self.gripper_dof_names.append(motor_name.replace("_rack", "_pinion"))
            self.gripper_dof_names.append(motor_name.replace("_rack", "_pinion_mirror"))

        self.nu = len(self.motor_ordering)
        self.motor_ids = list(range(self.nu))
        if "teleop_leader" in robot_name:
            self.motor_ids = [16 + i for i in self.motor_ids]

        self.motor_gear_ratios: Dict[str, float] = {}
        for motor_name in self.motor_ordering:
            joint_equality = robot_root.find(
                f".//equality/joint[@joint2='{motor_name}']"
            )
            if joint_equality is not None:
                self.motor_gear_ratios[motor_name] = float(
                    joint_equality.attrib["polycoef"].split()[1]
                )
            else:
                self.motor_gear_ratios[motor_name] = 1.0

        self.default_motor_angles: Dict[str, float] = {}
        self.motor_groups = []
        self.motor_models = []
        self.motor_zero_pos = []
        self.motor_kp_real = []
        self.motor_kd_real = []
        self.motor_kp_sim = []
        self.motor_kd_sim = []
        for motor_name in self.motor_ordering:
            motor_config = self.config["motors"][motor_name]
            self.default_motor_angles[motor_name] = motor_config["home_pos"]
            self.motor_groups.append(motor_config["group"])
            self.motor_models.append(motor_config["motor"])
            self.motor_zero_pos.append(motor_config["zero_pos"])
            self.motor_kp_real.append(motor_config["kp"])
            self.motor_kd_real.append(motor_config["kd"])
            self.motor_kp_sim.append(
                motor_config["kp"] / self.config["actuators"]["kp_ratio"]
            )
            self.motor_kd_sim.append(
                motor_config["kd"] / self.config["actuators"]["kd_ratio"]
            )

        self.passive_active_ratio = self.config["actuators"]["passive_active_ratio"]
        self.cur_sensor_mask = [
            1 if model.startswith("XM430") or model.startswith("XC330") else 0
            for model in self.motor_models
        ]

        self.joint_groups = self.motor_groups

        motor_group_arr = numpy.array(self.motor_groups)
        motor_name_arr = numpy.array(self.motor_ordering)
        self.neck_gear_ratio = np.array(
            [
                self.motor_gear_ratios[name]
                for name in motor_name_arr[motor_group_arr == "neck"]
            ],
            dtype=np.float32,
        )
        self.waist_coef = np.array(
            [
                self.config["kinematics"]["waist_roll_coef"],
                self.config["kinematics"]["waist_yaw_coef"],
            ],
            dtype=np.float32,
        )
        self.leg_gear_ratio = np.array(
            [
                self.motor_gear_ratios[name]
                for name in motor_name_arr[motor_group_arr == "leg"]
            ],
            dtype=np.float32,
        )
        self.arm_gear_ratio = np.array(
            [
                self.motor_gear_ratios[name]
                for name in motor_name_arr[motor_group_arr == "arm"]
            ],
            dtype=np.float32,
        )

        self.motor_tau_max = []
        self.motor_q_dot_max = []
        self.motor_tau_q_dot_max = []
        self.motor_q_dot_tau_max = []
        self.motor_tau_brake_max = []
        self.motor_kd_min = []
        self.motor_ki = []
        self.motor_bi = []
        for model_name in self.motor_models:
            motor_model = self.config["actuators"][model_name]
            self.motor_tau_max.append(motor_model["tau_max"])
            self.motor_q_dot_max.append(motor_model["q_dot_max"])
            self.motor_tau_q_dot_max.append(motor_model["tau_q_dot_max"])
            self.motor_q_dot_tau_max.append(motor_model["q_dot_tau_max"])
            self.motor_tau_brake_max.append(motor_model["tau_brake_max"])
            self.motor_kd_min.append(motor_model["kd_min"])
            self.motor_ki.append(motor_model["ki"])
            self.motor_bi.append(motor_model["bi"])

        self.default_joint_angles = self.motor_to_joint_angles(
            self.default_motor_angles
        )
        self.joint_ordering = list(self.default_joint_angles.keys())

        def get_limits(name: str) -> List[float]:
            joint_elem = robot_root.find(f".//joint[@name='{name}']")
            return list(map(float, joint_elem.attrib["range"].split()))

        self.motor_limits: Dict[str, List[float]] = {}
        for motor_name in self.motor_ordering:
            self.motor_limits[motor_name] = get_limits(motor_name)

        self.joint_limits: Dict[str, List[float]] = {}
        for joint_name in self.joint_ordering:
            self.joint_limits[joint_name] = get_limits(joint_name)

        # Build joint_to_motor_name mapping based on transmission type
        self.joint_to_motor_name: Dict[str, List[str]] = {}
        for motor_name in self.motor_ordering:
            transmission = self.get_transmission(motor_name)
            if transmission == "none":
                # Simple 1:1 mapping: joint and motor have the same name
                self.joint_to_motor_name[motor_name] = [motor_name]
            # Add other transmission types as needed

        if "neck_yaw_driven" in self.joint_limits and "neck_pitch" in self.joint_limits:
            self.neck_joint_limits = np.array(
                [self.joint_limits["neck_yaw_driven"], self.joint_limits["neck_pitch"]],
                dtype=np.float32,
            ).T

        if "waist_roll" in self.joint_limits and "waist_yaw" in self.joint_limits:
            self.waist_joint_limits = np.array(
                [self.joint_limits["waist_roll"], self.joint_limits["waist_yaw"]],
                dtype=np.float32,
            ).T

    def get_transmission(self, motor_name: str) -> str:
        """Determine transmission type based on motor name suffix.

        Args:
            motor_name: Name of the motor.

        Returns:
            Transmission type string.
        """
        if motor_name.endswith("_drive"):
            return "spur_gear"
        elif motor_name.endswith("_act"):
            return "parallel_linkage"
        elif re.search(r"_act_[12]$", motor_name):
            return "bevel_gear"
        elif motor_name.endswith("_rack"):
            return "rack_and_pinion"
        else:
            return "none"

    def motor_to_joint_angles(self, motor_angles: Dict[str, float]) -> Dict[str, float]:
        """Convert motor angles to joint angles using transmission ratios.

        Args:
            motor_angles: Dictionary of motor angles.

        Returns:
            Dictionary of joint angles.
        """
        joint_angles: Dict[str, float] = {}

        waist_act_1_pos = None
        for motor_name, motor_pos in motor_angles.items():
            transmission = self.get_transmission(motor_name)
            if transmission == "spur_gear":
                joint_name = motor_name.replace("_drive", "_driven")
                joint_angles[joint_name] = (
                    motor_pos * self.motor_gear_ratios[motor_name]
                )
            elif transmission == "parallel_linkage":
                joint_name = motor_name.replace("_act", "")
                joint_angles[joint_name] = motor_pos
            elif transmission == "bevel_gear":
                # Placeholder to ensure the correct order
                if waist_act_1_pos is None:
                    waist_act_1_pos = motor_pos
                    continue
                else:
                    joint_angles["waist_roll"], joint_angles["waist_yaw"] = (
                        self.waist_fk([waist_act_1_pos, motor_pos])
                    )
            elif transmission == "rack_and_pinion":
                joint_name = motor_name.replace("_rack", "_pinion")
                joint_angles[joint_name] = (
                    motor_pos * self.motor_gear_ratios[motor_name]
                )
            elif transmission == "none":
                joint_angles[motor_name] = motor_pos

        return joint_angles

    def motor_to_dof_angles(self, motor_angles: Dict[str, float]) -> Dict[str, float]:
        """Convert motor angles to degree-of-freedom angles.

        Args:
            motor_angles: Dictionary of motor angles.

        Returns:
            Dictionary including both joint and motor angles.
        """
        dof_angles = self.motor_to_joint_angles(motor_angles)
        dof_angles.update(motor_angles)
        for motor_name, motor_pos in motor_angles.items():
            transmission = self.get_transmission(motor_name)
            if transmission == "parallel_linkage":
                joint_name = motor_name.replace("_act", "")
                for suffix in ["_front", "_back"]:
                    dof_angles[joint_name + suffix] = -motor_pos
            elif transmission == "rack_and_pinion":
                joint_name = motor_name.replace("_rack", "_pinion")
                dof_angles[joint_name + "_mirror"] = dof_angles[joint_name]

        return dof_angles

    def joint_to_motor_angles(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        """Converts joint angles to motor angles based on the transmission type specified in the configuration.

        Args:
            joint_angles (Dict[str, float]): A dictionary mapping joint names to their respective angles.

        Returns:
            Dict[str, float]: A dictionary mapping motor names to their calculated angles.
        """
        motor_angles: Dict[str, float] = {}
        waist_roll_pos = None
        for joint_name, joint_pos in joint_angles.items():
            motor_name = self.motor_ordering[self.joint_ordering.index(joint_name)]
            transmission = self.get_transmission(motor_name)
            if transmission == "spur_gear":
                motor_angles[motor_name] = (
                    joint_pos / self.motor_gear_ratios[motor_name]
                )
            elif transmission == "parallel_linkage":
                motor_angles[motor_name] = joint_pos
            elif transmission == "bevel_gear":
                # Placeholder to ensure the correct order
                if waist_roll_pos is None:
                    waist_roll_pos = joint_pos
                    continue
                else:
                    motor_angles["waist_act_1"], motor_angles["waist_act_2"] = (
                        self.waist_ik([waist_roll_pos, joint_pos])
                    )
            elif transmission == "rack_and_pinion":
                motor_name = joint_name.replace("_pinion", "_rack")
                motor_angles[motor_name] = (
                    joint_pos / self.motor_gear_ratios[motor_name]
                )
            elif transmission == "none":
                motor_angles[motor_name] = joint_pos

        return motor_angles

    def joint_to_dof_angles(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        """Convert joint angles to degree-of-freedom angles.

        Args:
            joint_angles: Dictionary of joint angles.

        Returns:
            Dictionary including both motor and joint angles.
        """
        dof_angles = self.joint_to_motor_angles(joint_angles)
        dof_angles.update(joint_angles)
        for joint_name, joint_pos in joint_angles.items():
            motor_name = self.motor_ordering[self.joint_ordering.index(joint_name)]
            transmission = self.get_transmission(motor_name)
            if transmission == "parallel_linkage":
                for suffix in ["_front", "_back"]:
                    dof_angles[joint_name + suffix] = -joint_pos
            elif transmission == "rack_and_pinion":
                joint_name = motor_name.replace("_rack", "_pinion")
                dof_angles[joint_name + "_mirror"] = dof_angles[joint_name]

        return dof_angles

    def neck_fk(self, neck_motor_pos: ArrayType) -> ArrayType:
        """Calculates the neck joint positions from the neck motor positions.

        Args:
            neck_motor_pos (ArrayType): The positions of the neck motors.

        Returns:
            ArrayType: The calculated positions of the neck joints.
        """
        neck_joint_pos = neck_motor_pos * self.neck_gear_ratio
        return neck_joint_pos

    def neck_ik(self, neck_joint_pos: ArrayType) -> ArrayType:
        """Calculates the motor positions for the neck based on the joint positions.

        Args:
            neck_joint_pos (ArrayType): The positions of the neck joints.

        Returns:
            ArrayType: The calculated motor positions for the neck.
        """
        neck_motor_pos = neck_joint_pos / self.neck_gear_ratio
        return neck_motor_pos

    def waist_fk(self, waist_motor_pos: ArrayType) -> ArrayType:
        """Calculates the forward kinematics for the waist joint based on motor positions.

        Args:
            waist_motor_pos (ArrayType): An array containing the positions of the waist motors.

        Returns:
            ArrayType: An array containing the calculated waist roll and yaw angles.
        """
        waist_roll = self.waist_coef[0] * (-waist_motor_pos[0] + waist_motor_pos[1])
        waist_yaw = self.waist_coef[1] * (waist_motor_pos[0] + waist_motor_pos[1])
        return np.array([waist_roll, waist_yaw], dtype=np.float32)

    def waist_ik(self, waist_joint_pos: ArrayType) -> ArrayType:
        """Calculates the inverse kinematics for the waist joint actuators.

        Args:
            waist_joint_pos (ArrayType): The position of the waist joint, represented as an array.

        Returns:
            ArrayType: An array containing the calculated positions for the two waist joint actuators.
        """
        waist_roll, waist_yaw = waist_joint_pos / self.waist_coef
        waist_act_1 = (-waist_roll + waist_yaw) / 2
        waist_act_2 = (waist_roll + waist_yaw) / 2
        return np.array([waist_act_1, waist_act_2], dtype=np.float32)

    def leg_fk(self, leg_motor_pos: ArrayType) -> ArrayType:
        """Converts leg motor positions to leg joint positions using the gear ratio.

        Args:
            leg_motor_pos (ArrayType): An array of motor positions for the leg.

        Returns:
            ArrayType: An array of joint positions for the leg, calculated by applying the gear ratio to the motor positions.
        """
        leg_joint_pos = leg_motor_pos * self.leg_gear_ratio
        return leg_joint_pos

    def leg_ik(self, leg_joint_pos: ArrayType) -> ArrayType:
        """Calculates the motor positions for a leg based on joint positions and gear ratio.

        Args:
            leg_joint_pos (ArrayType): The joint positions of the leg.

        Returns:
            ArrayType: The calculated motor positions for the leg.
        """
        leg_motor_pos = leg_joint_pos / self.leg_gear_ratio
        return leg_motor_pos

    def arm_fk(self, arm_motor_pos: ArrayType) -> ArrayType:
        """Calculates the forward kinematics for an arm by converting motor positions to joint positions.

        Args:
            arm_motor_pos (ArrayType): An array of motor positions for the arm.

        Returns:
            ArrayType: An array of joint positions corresponding to the given motor positions.
        """
        arm_joint_pos = arm_motor_pos * self.arm_gear_ratio
        return arm_joint_pos

    def arm_ik(self, arm_joint_pos: ArrayType) -> ArrayType:
        """Calculates the motor positions for an arm based on joint positions and gear ratio.

        Args:
            arm_joint_pos (ArrayType): The joint positions of the arm.

        Returns:
            ArrayType: The calculated motor positions for the arm.
        """
        arm_motor_pos = arm_joint_pos / self.arm_gear_ratio
        return arm_motor_pos
