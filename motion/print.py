"""Motion file inspection and analysis tool.

Provides detailed analysis of motion reference files including data format validation,
keyframe inspection, and trajectory information display.
"""

import argparse
import os
import sys

import joblib

REQUIRED_KEYS = [
    "time",
    "qpos",
    "action",
    "keyframes",
    "timed_sequence",
]

# New format keys (optional, for backward compatibility)
NEW_FORMAT_KEYS = [
    "body_pos",
    "body_quat",
    "site_pos",
    "site_quat",
    "body_lin_vel",
    "body_ang_vel",
    "motor_vel",
    "joint_vel",
]

# Old format keys (for backward compatibility)
OLD_FORMAT_KEYS = [
    "body_pose",
    "site_pose",
]


def inspect_motion_file(name):
    """Inspect and validate motion file contents and structure.

    Analyzes motion reference files and displays comprehensive information about data format and contents.
    """
    # Always look under the motion/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    motion_file = os.path.join(script_dir, f"{name}.lz4")

    if not os.path.exists(motion_file):
        print(f"[Error] File not found: {motion_file}")
        return

    print(f"\nInspecting file: {motion_file}")
    try:
        data = joblib.load(motion_file)

        # === Validate top-level keys ===
        print("\n== Top-level keys ==")
        top_keys = list(data.keys())
        for key in top_keys:
            print(f"- {key}")

        missing = [key for key in REQUIRED_KEYS if key not in top_keys]
        if missing:
            print(f"Missing required keys: {missing}")

        # === Data format detection ===
        print("\n== Data Format ==")
        is_relative = data.get("is_robot_relative_frame", False)
        frame_str = "robot-relative" if is_relative else "global"
        print(f"Frame type: {frame_str}")
        if "body_pos" in data and "body_quat" in data:
            print("New format detected (separate pos/quat arrays)")
            print("  Global torso position: Available in qpos[:3]")
            if "body_lin_vel" in data and "body_ang_vel" in data:
                print("Velocity data available")
            else:
                print("No velocity data found")
        elif "body_pose" in data:
            print("Old format detected (concatenated pos/quat arrays)")
            print("  Body poses: World frame")
        else:
            print("Unknown format - no body data found")

        # === Keyframe info ===
        print("\n== Keyframe info ==")
        keyframes = data.get("keyframes", [])
        print(f"Number of keyframes: {len(keyframes)}")
        if len(keyframes) > 0:
            if isinstance(keyframes[0], dict):
                print(f"Keys in first keyframe: {list(keyframes[0].keys())}")
                print(f"'name' in first keyframe: {keyframes[0]['name']}")
                print(
                    f"Shape of 'motor_pos' in first keyframe: {keyframes[0]['motor_pos'].shape}"
                )
                print(
                    f"Shape of 'joint_pos' in first keyframe: {keyframes[0]['joint_pos'].shape}"
                )
                print(
                    f"Shape of 'qpos' in first keyframe: {keyframes[0]['qpos'].shape}"
                )
            else:
                print(f"Keyframe type: {type(keyframes[0])}")

        # === Sequence info ===
        print("\n== Sequence info ==")
        sequence = data.get("timed_sequence", [])
        print(f"Sequence length: {sequence[-1][-1]} seconds")
        if len(sequence) > 0:
            print(f"First sequence item: {sequence[0]}")

        # === Time ===
        print("\n== Time vector ==")
        time = data.get("time", [])
        print(f"Length: {len(time)}")
        if len(time) >= 10:
            print(f"First 10 values: {time[:10]}")
        if hasattr(time, "dtype"):
            print(f"Time data type: {time.dtype}")
        elif hasattr(time, "__array__"):
            print(f"Time is array-like with type: {type(time)}")

        # === Action trajectory ===
        print("\n== Action Trajectory ==")
        actions = data.get("action", [])
        if actions is None:
            print(
                "This motion file is generated from qpos interpolation, no actions available."
            )
        else:
            if len(actions) > 0:
                print(f"Length: {len(actions)}")
                if hasattr(actions, "shape"):
                    print(f"Action array shape: {actions.shape}")
                    if len(actions.shape) > 1:
                        print(f"Individual action shape: {actions[0].shape}")
                elif hasattr(actions[0], "shape"):
                    print(f"Shape of first: {actions[0].shape}")
            if hasattr(actions, "dtype"):
                print(f"Action data type: {actions.dtype}")

        # === Motor velocity (optional, new data for policy observation) ===
        print("\n== Motor Velocity Data ==")
        if "motor_vel" in data:
            motor_vel = data.get("motor_vel", [])
            print(f"Length: {len(motor_vel)}")
            if len(motor_vel) > 0:
                if hasattr(motor_vel, "shape"):
                    print(f"Motor vel array shape: {motor_vel.shape}")
                    if len(motor_vel.shape) > 1:
                        print(f"Individual motor vel shape: {motor_vel[0].shape}")
                elif hasattr(motor_vel[0], "shape"):
                    print(f"Shape of first: {motor_vel[0].shape}")
            if hasattr(motor_vel, "dtype"):
                print(f"Motor vel data type: {motor_vel.dtype}")
        else:
            print("No motor velocity data found (new data for policy observation)")

        # === Joint velocity (optional, new data for qvel reconstruction) ===
        print("\n== Joint Velocity Data ==")
        if "joint_vel" in data:
            joint_vel = data.get("joint_vel", [])
            print(f"Length: {len(joint_vel)}")
            if len(joint_vel) > 0:
                if hasattr(joint_vel, "shape"):
                    print(f"Joint vel array shape: {joint_vel.shape}")
                    if len(joint_vel.shape) > 1:
                        print(f"Individual joint vel shape: {joint_vel[0].shape}")
                elif hasattr(joint_vel[0], "shape"):
                    print(f"Shape of first: {joint_vel[0].shape}")
            if hasattr(joint_vel, "dtype"):
                print(f"Joint vel data type: {joint_vel.dtype}")
        else:
            print("No joint velocity data found (new data for qvel reconstruction)")

        # === Qpos ===
        print("\n== Qpos Replay ==")
        qpos = data.get("qpos", [])
        print(f"Length: {len(qpos)}")
        print("Contains global torso position in qpos[:3] for reward calculations")
        if len(qpos) > 0:
            if hasattr(qpos, "shape"):
                print(f"Qpos array shape: {qpos.shape}")
                if len(qpos.shape) > 1:
                    print(f"Individual qpos shape: {qpos[0].shape}")
            elif hasattr(qpos[0], "shape"):
                print(f"Shape of first: {qpos[0].shape}")
        if hasattr(qpos, "dtype"):
            print(f"Qpos data type: {qpos.dtype}")

        # === Body data ===
        print("\n== Body Data ==")
        if "body_pos" in data and "body_quat" in data:
            # New format
            body_pos = data.get("body_pos", [])
            body_quat = data.get("body_quat", [])
            print(f"Body positions - Length: {len(body_pos)}")
            if len(body_pos) > 0:
                if hasattr(body_pos, "shape"):
                    print(f"  Body pos array shape: {body_pos.shape}")
                elif hasattr(body_pos[0], "shape"):
                    print(f"  Shape of first: {body_pos[0].shape}")
            if hasattr(body_pos, "dtype"):
                print(f"  Body pos data type: {body_pos.dtype}")

            print(f"Body quaternions - Length: {len(body_quat)}")
            if len(body_quat) > 0:
                if hasattr(body_quat, "shape"):
                    print(f"  Body quat array shape: {body_quat.shape}")
                elif hasattr(body_quat[0], "shape"):
                    print(f"  Shape of first: {body_quat[0].shape}")
            if hasattr(body_quat, "dtype"):
                print(f"  Body quat data type: {body_quat.dtype}")
        elif "body_pose" in data:
            # Old format
            body_pose = data.get("body_pose", [])
            print(f"Body poses (old format, world frame) - Length: {len(body_pose)}")
            if len(body_pose) > 0:
                if hasattr(body_pose, "shape"):
                    print(f"  Body pose array shape: {body_pose.shape}")
                elif hasattr(body_pose[0], "shape"):
                    print(f"  Shape of first: {body_pose[0].shape}")
            if hasattr(body_pose, "dtype"):
                print(f"  Body pose data type: {body_pose.dtype}")
        else:
            print("No body data found")

        # === Site data ===
        print("\n== Site Data ==")
        if "site_pos" in data and "site_quat" in data:
            # New format
            site_pos = data.get("site_pos", [])
            site_quat = data.get("site_quat", [])
            print(f"Site positions - Length: {len(site_pos)}")
            if len(site_pos) > 0:
                if hasattr(site_pos, "shape"):
                    print(f"  Site pos array shape: {site_pos.shape}")
                elif hasattr(site_pos[0], "shape"):
                    print(f"  Shape of first: {site_pos[0].shape}")
            if hasattr(site_pos, "dtype"):
                print(f"  Site pos data type: {site_pos.dtype}")

            print(f"Site quaternions - Length: {len(site_quat)}")
            if len(site_quat) > 0:
                if hasattr(site_quat, "shape"):
                    print(f"  Site quat array shape: {site_quat.shape}")
                elif hasattr(site_quat[0], "shape"):
                    print(f"  Shape of first: {site_quat[0].shape}")
            if hasattr(site_quat, "dtype"):
                print(f"  Site quat data type: {site_quat.dtype}")
        elif "site_pose" in data:
            # Old format
            site_pose = data.get("site_pose", [])
            print(f"Site poses (old format, world frame) - Length: {len(site_pose)}")
            if len(site_pose) > 0:
                if hasattr(site_pose, "shape"):
                    print(f"  Site pose array shape: {site_pose.shape}")
                elif hasattr(site_pose[0], "shape"):
                    print(f"  Shape of first: {site_pose[0].shape}")
            if hasattr(site_pose, "dtype"):
                print(f"  Site pose data type: {site_pose.dtype}")
        else:
            print("No site data found")

        # === Velocity data (new format only) ===
        print("\n== Velocity Data ==")
        if "body_lin_vel" in data and "body_ang_vel" in data:
            body_lin_vel = data.get("body_lin_vel", [])
            body_ang_vel = data.get("body_ang_vel", [])
            print(f"Body linear velocities - Length: {len(body_lin_vel)}")
            if len(body_lin_vel) > 0:
                if hasattr(body_lin_vel, "shape"):
                    print(f"  Body lin vel array shape: {body_lin_vel.shape}")
                elif hasattr(body_lin_vel[0], "shape"):
                    print(f"  Shape of first: {body_lin_vel[0].shape}")
            if hasattr(body_lin_vel, "dtype"):
                print(f"  Body lin vel data type: {body_lin_vel.dtype}")

            print(f"Body angular velocities - Length: {len(body_ang_vel)}")
            if len(body_ang_vel) > 0:
                if hasattr(body_ang_vel, "shape"):
                    print(f"  Body ang vel array shape: {body_ang_vel.shape}")
                elif hasattr(body_ang_vel[0], "shape"):
                    print(f"  Shape of first: {body_ang_vel[0].shape}")
            if hasattr(body_ang_vel, "dtype"):
                print(f"  Body ang vel data type: {body_ang_vel.dtype}")
        else:
            print("No velocity data found (only available in new format)")

    except Exception as e:
        print(f"[Error] Failed to load or inspect file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="Base name of motion file (e.g., 'push_up' for 'push_up.lz4')",
    )
    args = parser.parse_args()

    inspect_motion_file(args.name)
