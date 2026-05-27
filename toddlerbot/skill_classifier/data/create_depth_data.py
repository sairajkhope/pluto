"""
Run depth estimation on collected RGB stereo data.

Processes rectified stereo images from collect_real_world_skill_data.py output
and generates depth maps using Foundation Stereo models.

Multiple depth estimations per image pair can be performed since learning-based
models can produce varying results.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from toddlerbot.depth.depth_utils import vis_disparity
from toddlerbot.sim.terrain.get_depth import process_depth_map


def get_collection_dirs(results_dir="results", timestamps=None):
    """Get list of collection directories to process."""
    results_path = Path(results_dir)

    if timestamps:
        # Process specific timestamps
        dirs = [
            results_path / f"collect_real_world_skill_data_{ts}" for ts in timestamps
        ]
        # Verify all exist
        for d in dirs:
            if not d.exists():
                raise ValueError(f"Collection directory not found: {d}")
        return dirs
    else:
        # Process all collection directories
        pattern = re.compile(r"collect_real_world_skill_data_\d{8}_\d{6}")
        dirs = [
            d for d in results_path.iterdir() if d.is_dir() and pattern.match(d.name)
        ]
        return sorted(dirs)


def get_model_name(engine_path):
    """Extract model name from engine path (e.g., vitl_192x256_16.engine -> vitl_192x256_16.engine)."""
    return Path(engine_path).name


def process_collection(collection_dir, engine_path, calib_params_path, rec_params_path):
    """Process a single collection directory."""
    rgb_dir = collection_dir / "rgb"

    if not rgb_dir.exists():
        print(f"Warning: {rgb_dir} does not exist, skipping")
        return

    # Read metadata
    metadata_path = rgb_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Warning: {metadata_path} does not exist, skipping")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Get image resolution from metadata
    img_width = metadata["camera_system"]["resolution"]["width"]
    img_height = metadata["camera_system"]["resolution"]["height"]

    # Read timestamps to get frame info
    timestamps_path = rgb_dir / "timestamps.txt"
    if not timestamps_path.exists():
        print(f"Warning: {timestamps_path} does not exist, skipping")
        return

    # Parse timestamps file to get frame indices and skills
    frames = []
    with open(timestamps_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                frame_idx = int(parts[0])
                timestamp_us = int(parts[1])
                skill = parts[2]
                frames.append((frame_idx, timestamp_us, skill))

    print(f"\nProcessing: {collection_dir.name}")
    print(f"Found {len(frames)} frames")

    # Initialize depth estimator
    from toddlerbot.depth.depth_estimator_foundation_stereo import (
        DepthEstimatorFoundationStereo,
    )

    print("Initializing depth estimator...")
    depth_estimator = DepthEstimatorFoundationStereo(
        calib_params_path=calib_params_path,
        rec_params_path=rec_params_path,
        engine_path=engine_path,
        calib_width=img_width,
        calib_height=img_height,
    )

    # Create output directory
    model_name = get_model_name(engine_path)
    depth_dir = collection_dir / "depth" / model_name
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {depth_dir}")
    print(f"Processing {len(frames)} frames...")

    # Process each frame
    for frame_idx, timestamp_us, skill in tqdm(frames, desc="Processing frames"):
        # Load FISHEYE images (get_depth will rectify them)
        left_path = rgb_dir / f"{frame_idx}_left_fisheye_{skill}.jpg"
        right_path = rgb_dir / f"{frame_idx}_right_fisheye_{skill}.jpg"

        if not left_path.exists() or not right_path.exists():
            print(f"\nWarning: Missing images for frame {frame_idx}, skipping")
            continue

        img_left = cv2.imread(str(left_path))
        img_right = cv2.imread(str(right_path))

        if img_left is None or img_right is None:
            print(f"\nWarning: Failed to load images for frame {frame_idx}, skipping")
            continue

        # Run depth estimation
        try:
            depth_result = depth_estimator.get_depth(
                img_left,
                img_right,
                remove_invisible=True,
                return_all=True,
            )

            if depth_result is not None and depth_result.depth is not None:
                # Save depth map as .npy (raw depth measurement)
                depth_filename = f"{frame_idx}_depth_{skill}.npy"
                depth_path = depth_dir / depth_filename
                np.save(str(depth_path), depth_result.depth)

                # Create combined visualization (exactly like test_foundation_stereo.py)
                processed_depth = process_depth_map(depth_result.depth)
                depth_vis = vis_disparity(
                    processed_depth,
                    min_val=0,
                    max_val=1.0,  # zmax default
                    invalid_upper_thres=1.0,
                    invalid_bottom_thres=0.0,
                    no_color=True,
                )
                disp_vis = vis_disparity(depth_result.disparity)
                combined = np.hstack(
                    [
                        depth_result.rectified_left,
                        depth_result.rectified_right,
                        disp_vis,
                        depth_vis,
                    ],
                    dtype=np.uint8,
                )

                # Save combined visualization
                combined_filename = f"{frame_idx}_combined_{skill}.jpg"
                combined_path = depth_dir / combined_filename
                cv2.imwrite(str(combined_path), combined)

            else:
                print(f"\nWarning: Depth estimation failed for frame {frame_idx}")

        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {e}")
            continue

    # Save metadata for this depth processing
    depth_metadata = {
        "source_collection": collection_dir.name,
        "model_name": model_name,
        "engine_path": engine_path,
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_frames": len(frames),
    }

    depth_metadata_path = depth_dir / "depth_metadata.json"
    with open(depth_metadata_path, "w") as f:
        json.dump(depth_metadata, f, indent=2)

    print(f"Depth metadata saved to: {depth_metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run depth estimation on collected RGB stereo data"
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default="toddlerbot/depth/models/foundation_stereo_vitl_192x256_16.engine",
        help="Path to TensorRT engine file (default: toddlerbot/depth/models/foundation_stereo_vitl_192x256_16.engine)",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        nargs="+",
        help="Specific collection timestamps to process (e.g., 20241031_143022). If not specified, processes all.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory containing collection data (default: results)",
    )

    args = parser.parse_args()

    # Camera parameters (same as collection script)
    CALIB_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "calibration.pkl")
    REC_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "rectification.npz")

    # Verify engine path exists
    if not os.path.exists(args.engine_path):
        print(f"Error: Engine path does not exist: {args.engine_path}")
        return

    # Get collection directories to process
    try:
        collection_dirs = get_collection_dirs(args.results_dir, args.timestamps)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not collection_dirs:
        print(f"No collection directories found in {args.results_dir}")
        return

    print("=" * 70)
    print("Depth Estimation on RGB Stereo Data")
    print("=" * 70)
    print(f"Engine: {args.engine_path}")
    print(f"Collections to process: {len(collection_dirs)}")
    for d in collection_dirs:
        print(f"  - {d.name}")
    print("=" * 70)

    # Process each collection
    for collection_dir in collection_dirs:
        try:
            process_collection(
                collection_dir,
                args.engine_path,
                CALIB_PARAMS_PATH,
                REC_PARAMS_PATH,
            )
        except Exception as e:
            print(f"\nError processing {collection_dir.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("Depth estimation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
