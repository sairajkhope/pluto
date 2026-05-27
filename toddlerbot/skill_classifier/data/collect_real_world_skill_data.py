"""
Real-world stereo camera data collection script with dynamic skill switching.
Saves original fisheye + rectified RGB images with skill labels for later processing.

Collects data in a format compatible with future robologger integration.

Controls:
- Keyboard keys: Switch active skill instantly (mapping shown at startup)
- Spacebar: Pause/resume collection
- ESC: Exit and save summary
"""

import argparse
import json
import os
import pickle
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nReceived interrupt signal. Exiting...")
    sys.exit(0)


def create_key_mapping(skill_classes):
    """Create keyboard to skill mapping dynamically based on number of skills."""
    # Use number keys first (1-9), then letters if more skills
    keys = "123456789abcdefghijklmnopqrstuvwxyz"
    if len(skill_classes) > len(keys):
        raise ValueError(
            f"Too many skills ({len(skill_classes)}), max supported is {len(keys)}"
        )

    key_to_skill = {}
    for i, skill in enumerate(skill_classes):
        key_to_skill[keys[i]] = skill

    return key_to_skill


def print_controls(key_to_skill, opencv_note=False):
    """Print keyboard controls."""
    print("\n" + "=" * 70)
    if opencv_note:
        print("Skill Numbers (enter keyboard input on OpenCV window)")
    else:
        print("Skill Numbers")
    print("=" * 70)
    for key, skill in key_to_skill.items():
        print(f"  [{key}] {skill}")
    print("  [SPACE] Pause/resume collection")
    print("  [ESC]   Exit and save summary")
    print("=" * 70)


def load_camera_params_and_maps(calib_params_path, rec_params_path, width, height):
    """Load camera parameters and create rectification maps."""
    # Load calibration parameters
    with open(calib_params_path, "rb") as f:
        calib_params = pickle.load(f)

    # Load rectification parameters
    rec_params = np.load(rec_params_path, allow_pickle=True)

    # Create rectification maps (using fisheye model for fisheye cameras)
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        calib_params["K1"],
        calib_params["D1"],
        rec_params["R1"],
        rec_params["P1"],
        (width, height),
        cv2.CV_16SC2,
    )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        calib_params["K2"],
        calib_params["D2"],
        rec_params["R2"],
        rec_params["P2"],
        (width, height),
        cv2.CV_16SC2,
    )

    # Convert to JSON-serializable format
    camera_params = {
        "calibration": {
            "K1": calib_params["K1"].tolist(),
            "K2": calib_params["K2"].tolist(),
            "D1": calib_params["D1"].tolist(),
            "D2": calib_params["D2"].tolist(),
            "R": calib_params["R"].tolist() if "R" in calib_params else None,
            "T": calib_params["T"].tolist() if "T" in calib_params else None,
            "image_size": calib_params.get("image_size", [640, 480]),
        },
        "rectification": {
            "R1": rec_params["R1"].tolist(),
            "R2": rec_params["R2"].tolist(),
            "P1": rec_params["P1"].tolist(),
            "P2": rec_params["P2"].tolist(),
            "Q": rec_params["Q"].tolist(),
        },
    }

    return camera_params, map1_left, map2_left, map1_right, map2_right


def save_metadata(output_dir, args, camera_params, skill_classes, v4l2_settings):
    """Save comprehensive metadata for future robologger integration."""
    metadata = {
        "dataset_info": {
            "collection_script": "collect_real_world_skill_data.py",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_test_set": args.test_set,
        },
        "camera_system": {
            "system_name": "toddlerbot_stereo_fisheye",
            "left_camera_id": 4,  # video_devices[4]
            "right_camera_id": 0,  # video_devices[0]
            "resolution": {"width": 640, "height": 480},
            "format": "MJPG",
            "v4l2_settings": v4l2_settings,
        },
        "camera_params": camera_params,
        "skill_classes": skill_classes,
        "data_structure": {
            "description": "All images stored directly in rgb/ directory",
            "file_naming": "{frame_index}_{camera}_{view}_{skill_label}.jpg",
            "examples": [
                "0_left_fisheye_walk.jpg",
                "0_right_fisheye_walk.jpg",
                "0_left_rectified_walk.jpg",
                "0_right_rectified_walk.jpg",
            ],
            "timestamps": "Microsecond timestamps for each frame (synchronized)",
        },
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


def get_v4l2_settings():
    """Get current v4l2-ctl settings for both cameras."""
    import subprocess

    settings = {}
    for camera_name, video_id in [("left", 4), ("right", 0)]:
        # Get all device video IDs
        video_devices = sorted(
            [int(dev[5:]) for dev in os.listdir("/dev") if dev.startswith("video")]
        )
        actual_id = (
            video_devices[video_id] if video_id < len(video_devices) else video_id
        )

        try:
            result = subprocess.run(
                f"v4l2-ctl --device=/dev/video{actual_id} --list-ctrls",
                shell=True,
                capture_output=True,
                text=True,
            )
            settings[camera_name] = result.stdout
        except Exception as e:
            settings[camera_name] = f"Error retrieving settings: {e}"

    return settings


def main():
    parser = argparse.ArgumentParser(
        description="Collect real-world stereo camera data with dynamic skill switching"
    )
    parser.add_argument(
        "--test-set",
        action="store_true",
        help="Mark this as test set data (recorded in metadata)",
    )

    args = parser.parse_args()

    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # Import skill classes from config
    try:
        from toddlerbot.skill_classifier.config import SKILL_CLASSES
    except ImportError as e:
        print(f"Error importing SKILL_CLASSES from config: {e}")
        return

    # Import camera module
    try:
        from toddlerbot.sensing.camera import Camera
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        return

    # Camera parameters
    CALIB_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "calibration.pkl")
    DEFAULT_CALIB_HEIGHT = 480
    DEFAULT_CALIB_WIDTH = 640
    REC_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "rectification.npz")

    # Load camera parameters and rectification maps
    print("Loading camera parameters and rectification maps...")
    camera_params, map1_left, map2_left, map1_right, map2_right = (
        load_camera_params_and_maps(
            CALIB_PARAMS_PATH,
            REC_PARAMS_PATH,
            DEFAULT_CALIB_WIDTH,
            DEFAULT_CALIB_HEIGHT,
        )
    )
    print("Camera parameters loaded successfully")

    # Get v4l2 settings
    print("Reading v4l2-ctl settings...")
    v4l2_settings = get_v4l2_settings()

    # Initialize cameras
    print("Initializing cameras...")
    try:
        camera_left = Camera(
            "left", width=DEFAULT_CALIB_WIDTH, height=DEFAULT_CALIB_HEIGHT
        )
        camera_right = Camera(
            "right", width=DEFAULT_CALIB_WIDTH, height=DEFAULT_CALIB_HEIGHT
        )
        print("Cameras initialized successfully")
        print()
    except Exception as e:
        print(f"Failed to initialize cameras: {e}")
        return

    # Create keyboard mapping
    key_to_skill = create_key_mapping(SKILL_CLASSES)
    skill_to_key = {skill: key for key, skill in key_to_skill.items()}

    # Show controls and get initial selection
    print_controls(key_to_skill)

    valid_keys = list(key_to_skill.keys())
    while True:
        choice = input(
            f"\nSelect initial skill ({'/'.join(valid_keys[:3])}...): "
        ).lower()
        if choice in key_to_skill:
            current_skill = key_to_skill[choice]
            break
        else:
            print(f"Invalid key. Please use one of: {', '.join(valid_keys)}")

    # Create timestamped output directory
    time_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/collect_real_world_skill_data_{time_str}")
    rgb_dir = output_dir / "rgb"

    # Create rgb directory
    rgb_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata in rgb directory
    save_metadata(rgb_dir, args, camera_params, SKILL_CLASSES, v4l2_settings)

    # Open timestamps file in rgb directory
    timestamps_file = open(rgb_dir / "timestamps.txt", "w")
    timestamps_file.write("# frame_index timestamp_us skill_label\n")

    print("\n" + "=" * 70)
    print("Real-world Stereo Camera Data Collection")
    print("=" * 70)
    if args.test_set:
        print("Dataset type: TEST SET")
    print(f"Initial skill: {current_skill}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()

    # Data collection state
    frame_index = 0
    skill_sample_counts = defaultdict(int)
    start_time = time.time()
    is_paused = False

    print("Starting data collection...")
    print_controls(key_to_skill, opencv_note=True)
    print()

    try:
        while True:
            # Capture from cameras
            try:
                capture_start = time.time()
                img_left_fisheye = camera_left.get_frame()
                img_right_fisheye = camera_right.get_frame()
                capture_time_ms = (time.time() - capture_start) * 1000

                # Rectify images
                img_left_rectified = cv2.remap(
                    img_left_fisheye,
                    map1_left,
                    map2_left,
                    cv2.INTER_LINEAR,
                )
                img_right_rectified = cv2.remap(
                    img_right_fisheye,
                    map1_right,
                    map2_right,
                    cv2.INTER_LINEAR,
                )

                # Save data if not paused
                if not is_paused:
                    timestamp_us = int(time.time() * 1000000)

                    # Save all images in rgb directory with flat naming (index first)
                    cv2.imwrite(
                        str(
                            rgb_dir / f"{frame_index}_left_fisheye_{current_skill}.jpg"
                        ),
                        img_left_fisheye,
                    )
                    cv2.imwrite(
                        str(
                            rgb_dir / f"{frame_index}_right_fisheye_{current_skill}.jpg"
                        ),
                        img_right_fisheye,
                    )
                    cv2.imwrite(
                        str(
                            rgb_dir
                            / f"{frame_index}_left_rectified_{current_skill}.jpg"
                        ),
                        img_left_rectified,
                    )
                    cv2.imwrite(
                        str(
                            rgb_dir
                            / f"{frame_index}_right_rectified_{current_skill}.jpg"
                        ),
                        img_right_rectified,
                    )

                    # Save timestamp
                    timestamps_file.write(
                        f"{frame_index} {timestamp_us} {current_skill}\n"
                    )
                    timestamps_file.flush()

                    frame_index += 1
                    skill_sample_counts[current_skill] += 1

                    # Print status
                    elapsed = time.time() - start_time
                    skill_key = skill_to_key[current_skill]
                    status = (
                        f"[{skill_key}: {current_skill}] {skill_sample_counts[current_skill]} | "
                        f"Total: {frame_index} | "
                        f"Capture: {capture_time_ms:.1f}ms | "
                        f"Time: {elapsed:.1f}s"
                    )
                    print(f"\r{status}", end="", flush=True)

                # Show rectified images (downsampled for faster visualization)
                combined = np.hstack([img_left_rectified, img_right_rectified])
                combined_small = cv2.resize(
                    combined, (combined.shape[1] // 2, combined.shape[0] // 2)
                )
                cv2.imshow("Stereo Camera Collection (Rectified)", combined_small)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                # ESC to exit
                if key == 27:
                    print("\n\nExiting...")
                    break

                # Spacebar to pause/resume
                elif key == ord(" "):
                    is_paused = not is_paused
                    pause_status = "PAUSED" if is_paused else "RESUMED"
                    print(f"\n{pause_status}")

                # Check if valid skill key
                elif chr(key) in key_to_skill:
                    new_skill = key_to_skill[chr(key)]
                    if new_skill != current_skill:
                        current_skill = new_skill
                        print(f"\nSwitched to: {current_skill}")

            except Exception as e:
                print(f"\rError capturing images: {e}", end="", flush=True)
                continue

    except KeyboardInterrupt:
        print("\n\nCollection stopped by user")

    # Cleanup
    timestamps_file.close()
    camera_left.close()
    camera_right.close()
    cv2.destroyAllWindows()

    # Update metadata with final stats
    metadata_path = rgb_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    total_time = time.time() - start_time
    metadata["collection_stats"] = {
        "total_frames": frame_index,
        "total_time_seconds": round(total_time, 2),
        "average_fps": round(frame_index / total_time, 2) if total_time > 0 else 0,
        "frames_per_skill": dict(skill_sample_counts),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("Collection Summary")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"RGB data directory: {rgb_dir}")
    print(f"Total frames: {frame_index}")
    print(f"Total time: {total_time:.1f}s")
    if total_time > 0:
        print(f"Average FPS: {frame_index / total_time:.2f}")
    print()
    print("Frames per skill:")
    for skill in SKILL_CLASSES:
        count = skill_sample_counts[skill]
        if count > 0:
            print(f"  {skill}: {count} frames")
    print("=" * 70)


if __name__ == "__main__":
    main()
