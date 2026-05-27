#!/usr/bin/env python3
"""
Render motion videos for the motion gallery.

This script renders motion files to MP4 videos and generates thumbnails.
Videos are rendered using MuJoCo and saved to the docs static directory.

Usage:
    python render_motion_videos.py [--output DIR] [--force] [--motion NAME]

Options:
    --output: Output directory (default: _static/motion_gallery)
    --force: Re-render even if video already exists
    --motion: Render only a specific motion (e.g., cartwheel_2xc)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot


def extract_robot_type(motion_name: str) -> str:
    """Extract robot type from motion filename."""
    if "_2xc" in motion_name:
        return "toddlerbot_2xc"
    elif "_2xm" in motion_name:
        return "toddlerbot_2xm"
    else:
        return "toddlerbot_2xc"  # default


def render_motion_video(
    motion_path: str,
    robot_type: str,
    output_dir: str,
    force: bool = False
) -> tuple[bool, str]:
    """Render motion to MP4 video."""
    motion_name = Path(motion_path).stem
    output_path = os.path.join(output_dir, f"{motion_name}.mp4")

    # Skip if already exists
    if os.path.exists(output_path) and not force:
        print(f"  [skip] Video exists: {motion_name}.mp4")
        return True, output_path

    try:
        print(f"  Rendering {motion_name}...")

        # Create simulator
        robot = Robot(robot_type)
        sim = MuJoCoSim(
            robot,
            vis_type="render",
            fixed_base="fixed" in robot_type
        )

        # Load motion data
        motion_data = joblib.load(motion_path)
        qpos_seq = motion_data.get("qpos", motion_data if isinstance(motion_data, list) else [])

        if len(qpos_seq) == 0:
            print(f"  [warn] No qpos data in {motion_name}")
            sim.close()
            return False, ""

        # Render each frame
        pbar = tqdm(total=len(qpos_seq), desc=f"     ", leave=False)
        for qpos in qpos_seq:
            sim.set_qpos(np.array(qpos, dtype=np.float32))
            sim.forward()
            pbar.update(1)
        pbar.close()

        # Save video
        sim.save_recording(output_dir, sim.control_dt, 2, f"{motion_name}.mp4")
        sim.close()

        print(f"  [done] Saved: {motion_name}.mp4")
        return True, output_path

    except Exception as e:
        print(f"  [error] Failed {motion_name}: {e}")
        return False, ""


def generate_thumbnail(video_path: str, output_path: str, frame_time: float = 0.5) -> bool:
    """Generate thumbnail from video."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = min(int(fps * frame_time), total_frames // 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((400, 300))
            img.save(output_path, "JPEG", quality=90)
            return True
        return False

    except Exception as e:
        print(f"  [warn] Thumbnail error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Render motion videos for gallery")
    parser.add_argument("--output", type=str, default="_static/motion_gallery",
                        help="Output directory")
    parser.add_argument("--force", action="store_true",
                        help="Force re-render existing videos")
    parser.add_argument("--motion", type=str,
                        help="Render only a specific motion")
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    motion_dir = base_dir / "motion"
    output_dir = Path(args.output)
    videos_dir = output_dir / "videos"
    thumbnails_dir = output_dir / "thumbnails"

    # Create output directories
    videos_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    print("Motion Gallery Video Renderer")
    print("=" * 50)

    # Find motion files
    if args.motion:
        motion_name = args.motion if args.motion.endswith(".lz4") else f"{args.motion}.lz4"
        motion_files = [motion_dir / motion_name]
        if not motion_files[0].exists():
            print(f"Motion not found: {motion_name}")
            return
    else:
        motion_files = sorted(motion_dir.glob("*.lz4"))
        # Skip dataset files
        motion_files = [f for f in motion_files if "dataset" not in f.name.lower()]

    print(f"Found {len(motion_files)} motion file(s)")
    print(f"Output: {videos_dir}")
    print()

    success_count = 0
    fail_count = 0

    for i, motion_file in enumerate(motion_files, 1):
        print(f"[{i}/{len(motion_files)}] {motion_file.name}")

        robot_type = extract_robot_type(motion_file.stem)
        success, video_path = render_motion_video(
            str(motion_file),
            robot_type,
            str(videos_dir),
            args.force
        )

        if success and os.path.exists(video_path):
            thumb_path = thumbnails_dir / f"{motion_file.stem}.jpg"
            if not thumb_path.exists() or args.force:
                generate_thumbnail(video_path, str(thumb_path))
            success_count += 1
        else:
            fail_count += 1

    print()
    print("=" * 50)
    print(f"Done: {success_count} rendered, {fail_count} failed")


if __name__ == "__main__":
    main()
