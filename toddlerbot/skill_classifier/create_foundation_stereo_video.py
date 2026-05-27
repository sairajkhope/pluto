"""
Create videos from Foundation Stereo server saved frames.

This script converts saved .npy depth frames and RGB images from run_foundation_stereo.py --save-output
into MP4 videos with average FPS calculated from timestamps.

Creates three videos named after the directory:
- {dir_name}_depth.mp4: Depth visualization
- {dir_name}_rgb_left.mp4: Left RGB camera
- {dir_name}_rgb_right.mp4: Right RGB camera

Usage:
    python toddlerbot/skill_classifier/create_foundation_stereo_video.py \
        /path/to/stair_wall_box_chair
    # Creates: stair_wall_box_chair_depth.mp4, stair_wall_box_chair_rgb_left.mp4, etc.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create videos from Foundation Stereo saved frames (depth, rgb_left, rgb_right)"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing saved frames (e.g., foundation_stereo_server_*)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path for depth (default: {data_dir}/depth_video.mp4)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return 1

    # Read timestamps
    timestamps_path = data_dir / "timestamps.txt"
    if not timestamps_path.exists():
        print(f"Error: timestamps.txt not found in {data_dir}")
        return 1

    timestamps = []
    frame_indices = []
    with open(timestamps_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                frame_idx = int(parts[0])
                timestamp = float(parts[1])
                frame_indices.append(frame_idx)
                timestamps.append(timestamp)

    if not frame_indices:
        print("Error: No frames found in timestamps.txt")
        return 1

    # Calculate average FPS
    total_time = timestamps[-1] - timestamps[0]
    num_frames = len(frame_indices)
    avg_fps = num_frames / total_time if total_time > 0 else 30.0

    print("=" * 70)
    print("Foundation Stereo Video Creation")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Total frames: {num_frames}")
    print(f"Capture duration: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print("=" * 70)
    print()

    # Determine output paths based on directory name
    dir_name = data_dir.name
    if args.output:
        depth_output_path = Path(args.output)
    else:
        depth_output_path = data_dir / f"{dir_name}_depth.mp4"
    rgb_left_output_path = data_dir / f"{dir_name}_rgb_left.mp4"
    rgb_right_output_path = data_dir / f"{dir_name}_rgb_right.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Check which data types are available
    first_depth_path = data_dir / f"{frame_indices[0]:06d}_depth.npy"
    first_rgb_left_path = data_dir / f"{frame_indices[0]:06d}_rgb_left.png"
    first_rgb_right_path = data_dir / f"{frame_indices[0]:06d}_rgb_right.png"

    has_depth = first_depth_path.exists()
    has_rgb_left = first_rgb_left_path.exists()
    has_rgb_right = first_rgb_right_path.exists()

    if not has_depth and not has_rgb_left and not has_rgb_right:
        print("Error: No depth or RGB frames found")
        return 1

    # Create depth video
    if has_depth:
        first_depth = np.load(str(first_depth_path))
        h, w = first_depth.shape

        depth_writer = cv2.VideoWriter(
            str(depth_output_path), fourcc, avg_fps, (w, h), isColor=False
        )

        if not depth_writer.isOpened():
            print(f"Error: Could not open video writer for {depth_output_path}")
            return 1

        print(f"Creating depth video at {avg_fps:.2f} fps...")

        for frame_idx in tqdm(frame_indices, desc="Processing depth frames"):
            depth_path = data_dir / f"{frame_idx:06d}_depth.npy"

            if not depth_path.exists():
                print(f"\nWarning: Missing depth frame {frame_idx}, skipping")
                continue

            try:
                depth = np.load(str(depth_path))
                depth_vis = (depth * 255.0).astype(np.uint8)
                depth_writer.write(depth_vis)
            except Exception as e:
                print(f"\nError processing depth frame {frame_idx}: {e}")
                continue

        depth_writer.release()
        print(f"Depth video saved to: {depth_output_path}")
        print()

    # Create RGB left video
    if has_rgb_left:
        first_rgb_left = cv2.imread(str(first_rgb_left_path))
        h, w = first_rgb_left.shape[:2]

        rgb_left_writer = cv2.VideoWriter(
            str(rgb_left_output_path), fourcc, avg_fps, (w, h), isColor=True
        )

        if not rgb_left_writer.isOpened():
            print(f"Error: Could not open video writer for {rgb_left_output_path}")
            return 1

        print(f"Creating RGB left video at {avg_fps:.2f} fps...")

        for frame_idx in tqdm(frame_indices, desc="Processing RGB left frames"):
            rgb_path = data_dir / f"{frame_idx:06d}_rgb_left.png"

            if not rgb_path.exists():
                print(f"\nWarning: Missing RGB left frame {frame_idx}, skipping")
                continue

            try:
                # OpenCV reads PNG as BGR, which is what VideoWriter expects
                rgb_bgr = cv2.imread(str(rgb_path))
                rgb_left_writer.write(rgb_bgr)
            except Exception as e:
                print(f"\nError processing RGB left frame {frame_idx}: {e}")
                continue

        rgb_left_writer.release()
        print(f"RGB left video saved to: {rgb_left_output_path}")
        print()

    # Create RGB right video
    if has_rgb_right:
        first_rgb_right = cv2.imread(str(first_rgb_right_path))
        h, w = first_rgb_right.shape[:2]

        rgb_right_writer = cv2.VideoWriter(
            str(rgb_right_output_path), fourcc, avg_fps, (w, h), isColor=True
        )

        if not rgb_right_writer.isOpened():
            print(f"Error: Could not open video writer for {rgb_right_output_path}")
            return 1

        print(f"Creating RGB right video at {avg_fps:.2f} fps...")

        for frame_idx in tqdm(frame_indices, desc="Processing RGB right frames"):
            rgb_path = data_dir / f"{frame_idx:06d}_rgb_right.png"

            if not rgb_path.exists():
                print(f"\nWarning: Missing RGB right frame {frame_idx}, skipping")
                continue

            try:
                # OpenCV reads PNG as BGR, which is what VideoWriter expects
                rgb_bgr = cv2.imread(str(rgb_path))
                rgb_right_writer.write(rgb_bgr)
            except Exception as e:
                print(f"\nError processing RGB right frame {frame_idx}: {e}")
                continue

        rgb_right_writer.release()
        print(f"RGB right video saved to: {rgb_right_output_path}")
        print()

    print("=" * 70)
    print("Video creation complete!")
    print(f"Duration: {total_time:.2f}s at {avg_fps:.2f} fps")
    if has_depth:
        print(f"  - Depth: {depth_output_path}")
    if has_rgb_left:
        print(f"  - RGB Left: {rgb_left_output_path}")
    if has_rgb_right:
        print(f"  - RGB Right: {rgb_right_output_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
