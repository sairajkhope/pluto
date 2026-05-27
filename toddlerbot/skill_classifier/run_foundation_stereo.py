"""
Foundation Stereo ZMQ Server for Real-time Depth Streaming.

This server runs Foundation Stereo depth estimation in a separate process and publishes
processed depth frames via ZMQ for non-blocking consumption by control loops.

Usage:
    python toddlerbot/skill_classifier/run_foundation_stereo.py [--port 5555] [--vis] [--save-output]
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from toddlerbot.depth.depth_estimator_foundation_stereo import (
    DepthEstimatorFoundationStereo,
)
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim.terrain.get_depth import process_depth_map

# Foundation Stereo default parameters (same as test_foundation_stereo.py)
CALIB_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "calibration.pkl")
DEFAULT_CALIB_HEIGHT = 480
DEFAULT_CALIB_WIDTH = 640
REC_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "rectification.npz")
ENGINE_PATH = os.path.join(
    "toddlerbot",
    "depth",
    "models",
    "foundation_stereo_vitl_192x256_16.engine",
)

# Warmup frames to skip for stable initialization
WARMUP_FRAMES = 20

# Global flag for graceful shutdown
keep_running = True


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global keep_running
    print("\nReceived interrupt signal. Shutting down...")
    keep_running = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Foundation Stereo ZMQ Server for real-time depth streaming"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ port to publish depth frames (default: 5555)",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize depth frames in OpenCV window",
    )
    parser.add_argument(
        "--skip-rectify",
        action="store_true",
        help="Skip rectification step",
    )
    parser.add_argument(
        "--calib-params",
        type=str,
        default=CALIB_PARAMS_PATH,
        help="Path to calibration parameters file",
    )
    parser.add_argument(
        "--rec-params",
        type=str,
        default=REC_PARAMS_PATH,
        help="Path to rectification parameters file",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=ENGINE_PATH,
        help="Path to Foundation Stereo engine file",
    )
    parser.add_argument(
        "--calib-width",
        type=int,
        default=DEFAULT_CALIB_WIDTH,
        help="Width of images used for calibration",
    )
    parser.add_argument(
        "--calib-height",
        type=int,
        default=DEFAULT_CALIB_HEIGHT,
        help="Height of images used for calibration",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save processed depth frames to results directory after exiting",
    )
    return parser.parse_args()


def main():
    """Run Foundation Stereo ZMQ server."""
    global keep_running

    args = parse_args()

    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("Foundation Stereo ZMQ Server")
    print("=" * 70)
    print(f"Publishing on port: {args.port}")
    print(f"Visualization: {'enabled' if args.vis else 'disabled'}")
    print(f"Save output: {'enabled' if args.save_output else 'disabled'}")
    print("=" * 70)
    print()

    # Initialize ZMQ publisher with optimizations
    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    # Set high water mark to 1 - only keep latest frame, drop old ones
    socket.setsockopt(zmq.SNDHWM, 1)

    # Bind to all interfaces
    socket.bind(f"tcp://*:{args.port}")
    print(f"ZMQ publisher bound to tcp://*:{args.port}")
    print("High water mark: 1 (drops old frames if subscriber is slow)")
    print()

    # Initialize Foundation Stereo
    print("Initializing Foundation Stereo...")
    try:
        depth_estimator = DepthEstimatorFoundationStereo(
            calib_params_path=args.calib_params,
            rec_params_path=args.rec_params,
            engine_path=args.engine,
            calib_width=args.calib_width,
            calib_height=args.calib_height,
        )
        print("Foundation Stereo initialized successfully")
        print()
    except Exception as e:
        print(f"Failed to initialize Foundation Stereo: {e}")
        return 1

    # Main loop
    frame_count = 0
    publish_count = 0
    loop_times = []

    # Data collection for saving
    saved_frames = [] if args.save_output else None
    saved_rgb_left = [] if args.save_output else None
    saved_rgb_right = [] if args.save_output else None
    saved_timestamps = [] if args.save_output else None

    camera_left = Camera("left", width=args.calib_width, height=args.calib_height)
    camera_right = Camera("right", width=args.calib_width, height=args.calib_height)

    print("Starting depth streaming...")
    print("Press Ctrl+C to stop")
    print()

    try:
        while keep_running:
            loop_start = time.time()

            # Get depth from Foundation Stereo
            try:
                img_left = camera_left.get_frame()
                img_right = camera_right.get_frame()

                depth_result = depth_estimator.get_depth(
                    img_left,
                    img_right,
                    remove_invisible=1,
                    return_all=args.save_output,
                )

                if depth_result is None or depth_result.depth is None:
                    print(
                        f"\rWarning: Failed to get depth (frame {frame_count})",
                        end="",
                        flush=True,
                    )
                    continue

                frame_count += 1

                # Warmup period - skip first N frames
                if frame_count <= WARMUP_FRAMES:
                    print(
                        f"\rWarming up... ({frame_count}/{WARMUP_FRAMES})",
                        end="",
                        flush=True,
                    )
                    continue
                elif frame_count == WARMUP_FRAMES + 1:
                    print("\r" + " " * 50)  # Clear warmup message
                    print("Warmup complete. Publishing depth frames...")

                # Process depth map (same as simulation and data collection)
                # Returns float32 array in [0, 1.0] range (meters, cropped, clipped)
                processed_depth = process_depth_map(
                    depth_result.depth, add_noise=False, max_depth=1.0
                )

                # Publish via ZMQ (non-blocking, drops if no subscribers)
                # Send float32 directly - no precision loss, localhost bandwidth is not a concern
                # Format: numpy array serialized with tobytes()
                socket.send(processed_depth.tobytes(), zmq.NOBLOCK)
                publish_count += 1

                # Save frame if output saving is enabled
                if args.save_output:
                    saved_frames.append(processed_depth.copy())
                    saved_rgb_left.append(depth_result.rectified_left.copy())
                    saved_rgb_right.append(depth_result.rectified_right.copy())
                    saved_timestamps.append(time.time())

                # Visualization (optional)
                if args.vis:
                    depth_vis = (processed_depth * 255.0).astype(np.uint8)
                    cv2.imshow("Foundation Stereo Server", depth_vis)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                        print("\nESC pressed. Exiting...")
                        break

                # Compute loop timing
                loop_time = (time.time() - loop_start) * 1000  # ms
                loop_times.append(loop_time)

                # Print status every 10 frames
                if publish_count % 10 == 0:
                    avg_time = np.mean(loop_times[-10:])
                    fps = 1000.0 / avg_time if avg_time > 0 else 0
                    print(
                        f"\rPublished {publish_count} frames | "
                        f"Loop: {loop_time:.1f}ms | "
                        f"Avg: {avg_time:.1f}ms ({fps:.1f} Hz)",
                        end="",
                        flush=True,
                    )

            except Exception as e:
                print(f"\nError during depth capture: {e}")
                continue

    except KeyboardInterrupt:
        print("\n\nShutdown requested...")

    finally:
        # Cleanup
        print("\n\nShutting down...")

        if args.vis:
            cv2.destroyAllWindows()

        socket.close()
        context.term()

        # Print statistics
        if loop_times:
            print("\n" + "=" * 70)
            print("Session Statistics")
            print("=" * 70)
            print(f"Total frames published: {publish_count}")
            print(f"Average loop time: {np.mean(loop_times):.2f}ms")
            print(f"Std loop time: {np.std(loop_times):.2f}ms")
            print(f"Min loop time: {np.min(loop_times):.2f}ms")
            print(f"Max loop time: {np.max(loop_times):.2f}ms")
            print(f"Average FPS: {1000.0 / np.mean(loop_times):.2f} Hz")
            print("=" * 70)

        # Save collected frames if output saving was enabled
        if args.save_output and saved_frames:
            print("\n" + "=" * 70)
            print("Saving collected depth frames...")
            print("=" * 70)

            # Create output directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            engine_name = os.path.splitext(os.path.basename(args.engine))[0]
            output_dir = Path(
                f"results/foundation_stereo_server_{engine_name}_{timestamp}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save each frame as .npy for depth and .png for RGB
            print(f"Saving {len(saved_frames)} frames to {output_dir}")
            for i, (depth, rgb_left, rgb_right, timestamp_val) in enumerate(
                zip(saved_frames, saved_rgb_left, saved_rgb_right, saved_timestamps)
            ):
                frame_path = output_dir / f"{i:06d}_depth.npy"
                np.save(str(frame_path), depth)
                # Save rectified RGB images
                rgb_left_path = output_dir / f"{i:06d}_rgb_left.png"
                rgb_right_path = output_dir / f"{i:06d}_rgb_right.png"
                cv2.imwrite(str(rgb_left_path), rgb_left)
                cv2.imwrite(str(rgb_right_path), rgb_right)

            # Save timestamps
            timestamps_path = output_dir / "timestamps.txt"
            with open(timestamps_path, "w") as f:
                f.write("# frame_index timestamp_unix\n")
                for i, timestamp_val in enumerate(saved_timestamps):
                    f.write(f"{i} {timestamp_val:.6f}\n")

            # Save metadata
            metadata = {
                "script": "run_foundation_stereo.py",
                "session_timestamp": timestamp,
                "engine_path": args.engine,
                "engine_name": engine_name,
                "calib_params": args.calib_params,
                "rec_params": args.rec_params,
                "calib_width": args.calib_width,
                "calib_height": args.calib_height,
                "total_frames": len(saved_frames),
                "zmq_port": args.port,
                "performance": {
                    "avg_loop_time_ms": float(np.mean(loop_times))
                    if loop_times
                    else None,
                    "std_loop_time_ms": float(np.std(loop_times))
                    if loop_times
                    else None,
                    "avg_fps": float(1000.0 / np.mean(loop_times))
                    if loop_times
                    else None,
                },
            }

            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {len(saved_frames)} depth frames and rectified RGB pairs")
            print(f"Output directory: {output_dir}")
            print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
