"""
Create videos from collected RGB and depth data.

Generates MP4 videos for:
- RGB: Side-by-side left and right rectified images
- Depth: Visualized depth maps for each model (processed and colorized)
"""

import argparse
import re
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


def create_rgb_video(collection_dir, fps=30):
    """Create video from RGB rectified images (left and right side-by-side)."""
    rgb_dir = collection_dir / "rgb"
    if not rgb_dir.exists():
        print(f"Warning: {rgb_dir} does not exist, skipping RGB video")
        return

    # Read timestamps to get frame order
    timestamps_path = rgb_dir / "timestamps.txt"
    if not timestamps_path.exists():
        print(f"Warning: {timestamps_path} does not exist, skipping RGB video")
        return

    # Parse timestamps file
    frames = []
    with open(timestamps_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                frame_idx = int(parts[0])
                skill = parts[2]
                frames.append((frame_idx, skill))

    if not frames:
        print("No frames found in timestamps.txt")
        return

    print(f"\nCreating RGB video from {len(frames)} frames...")

    # Get first frame to determine size
    first_idx, first_skill = frames[0]
    left_path = rgb_dir / f"{first_idx}_left_rectified_{first_skill}.jpg"
    right_path = rgb_dir / f"{first_idx}_right_rectified_{first_skill}.jpg"

    if not left_path.exists() or not right_path.exists():
        print("Error: First frame images not found")
        return

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        print("Error: Could not read first frame")
        return

    # Side-by-side dimensions
    h, w = left_img.shape[:2]
    combined_width = w * 2
    combined_height = h

    # Create video writer
    output_path = collection_dir / "rgb_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (combined_width, combined_height)
    )

    # Process each frame
    for frame_idx, skill in tqdm(frames, desc="RGB video"):
        left_path = rgb_dir / f"{frame_idx}_left_rectified_{skill}.jpg"
        right_path = rgb_dir / f"{frame_idx}_right_rectified_{skill}.jpg"

        if not left_path.exists() or not right_path.exists():
            print(f"\nWarning: Missing frame {frame_idx}, skipping")
            continue

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        if left_img is None or right_img is None:
            print(f"\nWarning: Could not read frame {frame_idx}, skipping")
            continue

        # Stack side-by-side
        combined = np.hstack([left_img, right_img])
        writer.write(combined)

    writer.release()
    print(f"RGB video saved to: {output_path}")


def create_depth_video(collection_dir, model_name, fps=30):
    """Create video from depth maps for a specific model."""
    depth_dir = collection_dir / "depth" / model_name
    rgb_dir = collection_dir / "rgb"

    if not depth_dir.exists():
        print(
            f"Warning: {depth_dir} does not exist, skipping depth video for {model_name}"
        )
        return

    # Read timestamps to get frame order
    timestamps_path = rgb_dir / "timestamps.txt"
    if not timestamps_path.exists():
        print(f"Warning: {timestamps_path} does not exist, skipping depth video")
        return

    # Parse timestamps file
    frames = []
    with open(timestamps_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                frame_idx = int(parts[0])
                skill = parts[2]
                frames.append((frame_idx, skill))

    if not frames:
        print("No frames found in timestamps.txt")
        return

    print(f"\nCreating depth video for {model_name} from {len(frames)} frames...")

    # Get first frame to determine size
    first_idx, first_skill = frames[0]
    depth_path = depth_dir / f"{first_idx}_depth_{first_skill}.npy"

    if not depth_path.exists():
        print(f"Error: First depth frame not found: {depth_path}")
        return

    # Load and process first depth frame
    raw_depth = np.load(str(depth_path))
    processed_depth = process_depth_map(raw_depth, add_noise=False, max_depth=1.0)
    depth_vis = vis_disparity(
        processed_depth,
        min_val=0,
        max_val=1.0,
        invalid_upper_thres=1.0,
        invalid_bottom_thres=0.0,
        no_color=True,
    )

    h, w = depth_vis.shape[:2]

    # Create video writer
    output_path = collection_dir / f"depth_{model_name.replace('.engine', '')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Process each frame
    for frame_idx, skill in tqdm(frames, desc=f"Depth video ({model_name})"):
        depth_path = depth_dir / f"{frame_idx}_depth_{skill}.npy"

        if not depth_path.exists():
            print(f"\nWarning: Missing depth frame {frame_idx}, skipping")
            continue

        try:
            # Load raw depth (.npy contains raw meter values, not processed)
            raw_depth = np.load(str(depth_path))

            # Apply same processing as in create_depth_data.py:
            # 1. Crop leftmost 10 pixels and clip to 1.0m
            processed_depth = process_depth_map(
                raw_depth, add_noise=False, max_depth=1.0
            )

            # 2. Visualize with grayscale colormap
            depth_vis = vis_disparity(
                processed_depth,
                min_val=0,
                max_val=1.0,
                invalid_upper_thres=1.0,
                invalid_bottom_thres=0.0,
                no_color=True,
            )

            writer.write(depth_vis)

        except Exception as e:
            print(f"\nError processing depth frame {frame_idx}: {e}")
            continue

    writer.release()
    print(f"Depth video saved to: {output_path}")


def process_collection(collection_dir, fps=30):
    """Process a single collection directory."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {collection_dir.name}")
    print(f"{'=' * 70}")

    # Create RGB video
    create_rgb_video(collection_dir, fps)

    # Create depth videos for each model
    depth_parent_dir = collection_dir / "depth"
    if depth_parent_dir.exists():
        model_dirs = [d for d in depth_parent_dir.iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            create_depth_video(collection_dir, model_dir.name, fps)
    else:
        print("No depth directory found, skipping depth videos")


def main():
    parser = argparse.ArgumentParser(
        description="Create videos from collected RGB and depth data"
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        nargs="+",
        help="Specific collection timestamps to process (e.g., 20241031_143022). If not specified, processes all.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video framerate (default: 30)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory containing collection data (default: results)",
    )

    args = parser.parse_args()

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
    print("Video Creation from RGB and Depth Data")
    print("=" * 70)
    print(f"FPS: {args.fps}")
    print(f"Collections to process: {len(collection_dirs)}")
    for d in collection_dirs:
        print(f"  - {d.name}")
    print("=" * 70)

    # Process each collection
    for collection_dir in collection_dirs:
        try:
            process_collection(collection_dir, args.fps)
        except Exception as e:
            print(f"\nError processing {collection_dir.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("Video creation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
