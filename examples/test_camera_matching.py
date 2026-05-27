"""Automatically find right camera settings to match left camera using histogram analysis."""

import subprocess
import time

import cv2
import numpy as np

from toddlerbot.sensing.camera import (
    LEFT_CAMERA_BRIGHTNESS,
    LEFT_CAMERA_CONTRAST,
    LEFT_CAMERA_EXPOSURE,
    Camera,
)


def set_camera_controls(camera_id, exposure, brightness, contrast):
    """Set v4l2-ctl controls for a camera."""
    controls = f"auto_exposure=1,exposure_time_absolute={exposure},brightness={brightness},contrast={contrast}"
    subprocess.run(
        f"v4l2-ctl --device=/dev/video{camera_id} --set-ctrl={controls}",
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)  # Let camera adjust


def analyze_histogram(frame):
    """Analyze histogram and return statistics."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Calculate statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    return {"mean": mean_brightness, "std": std_brightness, "hist": hist.flatten()}


def compute_histogram_difference(left_stats, right_stats):
    """Compute histogram difference between left and right."""
    mean_diff = abs(left_stats["mean"] - right_stats["mean"])
    std_diff = abs(left_stats["std"] - right_stats["std"])
    hist_corr = cv2.compareHist(
        left_stats["hist"].astype(np.float32),
        right_stats["hist"].astype(np.float32),
        cv2.HISTCMP_CORREL,
    )

    # Combined score (lower mean/std diff and higher correlation is better)
    score = mean_diff + std_diff - (hist_corr * 100)
    return score, mean_diff, std_diff, hist_corr


def capture_and_analyze(left_cam, right_cam, frames=5):
    """Capture frames and analyze histograms."""
    left_stats_list = []
    right_stats_list = []
    last_left = None
    last_right = None

    for _ in range(frames):
        left_frame = left_cam.get_frame()
        right_frame = right_cam.get_frame()

        left_stats_list.append(analyze_histogram(left_frame))
        right_stats_list.append(analyze_histogram(right_frame))

        last_left = left_frame
        last_right = right_frame

    # Average statistics
    left_stats = {
        "mean": np.mean([s["mean"] for s in left_stats_list]),
        "std": np.mean([s["std"] for s in left_stats_list]),
        "hist": np.mean([s["hist"] for s in left_stats_list], axis=0),
    }
    right_stats = {
        "mean": np.mean([s["mean"] for s in right_stats_list]),
        "std": np.mean([s["std"] for s in right_stats_list]),
        "hist": np.mean([s["hist"] for s in right_stats_list], axis=0),
    }

    return left_stats, right_stats, last_left, last_right


def save_camera_info(camera_id, output_dir, prefix):
    """Save detailed camera information."""
    # List all controls
    result = subprocess.run(
        f"v4l2-ctl --device=/dev/video{camera_id} --list-ctrls",
        shell=True,
        capture_output=True,
        text=True,
    )
    with open(f"{output_dir}/{prefix}_list_ctrls.txt", "w") as f:
        f.write(result.stdout)

    # Get parameters
    result = subprocess.run(
        f"v4l2-ctl --device=/dev/video{camera_id} --get-parm",
        shell=True,
        capture_output=True,
        text=True,
    )
    with open(f"{output_dir}/{prefix}_get_parm.txt", "w") as f:
        f.write(result.stdout)

    # Get video format
    result = subprocess.run(
        f"v4l2-ctl --device=/dev/video{camera_id} --get-fmt-video",
        shell=True,
        capture_output=True,
        text=True,
    )
    with open(f"{output_dir}/{prefix}_get_fmt_video.txt", "w") as f:
        f.write(result.stdout)


def main():
    # Use reference settings from camera.py (left camera - locked)
    left_exposure = LEFT_CAMERA_EXPOSURE
    left_brightness = LEFT_CAMERA_BRIGHTNESS
    left_contrast = LEFT_CAMERA_CONTRAST

    # Create results directory
    import os

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"camera_matching_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("Starting automatic camera matching with histogram analysis...")
    print(
        f"Left camera (fixed): exposure={left_exposure}, brightness={left_brightness}, contrast={left_contrast}"
    )
    print(f"Results will be saved to: {output_dir}")
    print()

    # Initialize cameras
    left_cam = Camera("left")
    right_cam = Camera("right")

    # Get camera IDs
    video_devices = sorted(
        [int(dev[5:]) for dev in os.listdir("/dev") if dev.startswith("video")]
    )
    left_id = video_devices[4]
    right_id = video_devices[0]

    # Set left camera to target settings
    set_camera_controls(left_id, left_exposure, left_brightness, left_contrast)

    # Start with same settings on right
    set_camera_controls(right_id, left_exposure, left_brightness, left_contrast)
    time.sleep(0.5)

    # Get baseline histogram difference
    left_stats, right_stats, _, _ = capture_and_analyze(left_cam, right_cam)
    baseline_score, baseline_mean_diff, baseline_std_diff, baseline_corr = (
        compute_histogram_difference(left_stats, right_stats)
    )

    print("Baseline (same settings):")
    print(f"  Mean brightness diff: {baseline_mean_diff:.2f}")
    print(f"  Std dev diff: {baseline_std_diff:.2f}")
    print(f"  Histogram correlation: {baseline_corr:.4f}")
    print(f"  Combined score: {baseline_score:.2f}")
    print()

    best_score = baseline_score
    best_exposure = left_exposure
    best_brightness = left_brightness
    best_contrast = left_contrast

    # Coarse search first
    print("Phase 1: Coarse search...")
    exposure_range = range(max(1, left_exposure - 40), min(200, left_exposure + 41), 10)
    brightness_range = range(
        max(-64, left_brightness - 40), min(65, left_brightness + 41), 10
    )
    contrast_range = range(max(0, left_contrast - 24), min(65, left_contrast + 25), 8)

    for exposure in exposure_range:
        for brightness in brightness_range:
            for contrast in contrast_range:
                set_camera_controls(right_id, exposure, brightness, contrast)
                _, right_stats, _, _ = capture_and_analyze(
                    left_cam, right_cam, frames=3
                )
                score, mean_diff, std_diff, corr = compute_histogram_difference(
                    left_stats, right_stats
                )

                if score < best_score:
                    best_score = score
                    best_exposure = exposure
                    best_brightness = brightness
                    best_contrast = contrast
                    print(
                        f"✓ E={exposure:3d} B={brightness:+3d} C={contrast:2d} → mean_diff={mean_diff:.1f} std_diff={std_diff:.1f} corr={corr:.3f} score={score:.2f}"
                    )

    # Fine search around best
    print()
    print("Phase 2: Fine search around best...")
    exposure_range = range(max(1, best_exposure - 10), min(200, best_exposure + 11), 2)
    brightness_range = range(
        max(-64, best_brightness - 10), min(65, best_brightness + 11), 2
    )
    contrast_range = range(max(0, best_contrast - 8), min(65, best_contrast + 9), 2)

    for exposure in exposure_range:
        for brightness in brightness_range:
            for contrast in contrast_range:
                set_camera_controls(right_id, exposure, brightness, contrast)
                _, right_stats, _, _ = capture_and_analyze(
                    left_cam, right_cam, frames=3
                )
                score, mean_diff, std_diff, corr = compute_histogram_difference(
                    left_stats, right_stats
                )

                if score < best_score:
                    best_score = score
                    best_exposure = exposure
                    best_brightness = brightness
                    best_contrast = contrast
                    print(
                        f"✓ E={exposure:3d} B={brightness:+3d} C={contrast:2d} → mean_diff={mean_diff:.1f} std_diff={std_diff:.1f} corr={corr:.3f} score={score:.2f}"
                    )

    print()
    print("=" * 70)
    print("FINAL RESULTS:")
    print(
        f"  Left camera:  exposure={left_exposure}, brightness={left_brightness}, contrast={left_contrast}"
    )
    print(
        f"  Right camera: exposure={best_exposure}, brightness={best_brightness}, contrast={best_contrast}"
    )
    print(f"  Final score: {best_score:.2f} (baseline: {baseline_score:.2f})")
    print()
    print("Update right camera controls in camera.py to:")
    print(
        f'  controls = "auto_exposure=1,exposure_time_absolute={best_exposure},brightness={best_brightness},contrast={best_contrast}"'
    )
    print("=" * 70)

    # Apply best settings and show result
    set_camera_controls(right_id, best_exposure, best_brightness, best_contrast)
    time.sleep(0.3)

    # Save camera information
    print()
    print("Saving camera information...")
    save_camera_info(left_id, output_dir, "left_camera")
    save_camera_info(right_id, output_dir, "right_camera")

    # Capture final result
    _, _, left_frame, right_frame = capture_and_analyze(left_cam, right_cam)
    combined = np.hstack([left_frame, right_frame])
    cv2.imwrite(f"{output_dir}/camera_match_result.jpg", combined)

    # Save summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("Camera Matching Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Left camera (fixed):\n")
        f.write(f"  exposure_time_absolute: {left_exposure}\n")
        f.write(f"  brightness: {left_brightness}\n")
        f.write(f"  contrast: {left_contrast}\n\n")
        f.write("Right camera (matched):\n")
        f.write(f"  exposure_time_absolute: {best_exposure}\n")
        f.write(f"  brightness: {best_brightness}\n")
        f.write(f"  contrast: {best_contrast}\n\n")
        f.write(f"Baseline score: {baseline_score:.2f}\n")
        f.write(f"Final score: {best_score:.2f}\n\n")
        f.write("Update right camera controls in camera.py to:\n")
        f.write(
            f'controls = "auto_exposure=1,exposure_time_absolute={best_exposure},brightness={best_brightness},contrast={best_contrast}"\n'
        )

    print()
    print(f"Results saved to: {output_dir}/")
    print("  - camera_match_result.jpg")
    print("  - summary.txt")
    print("  - left_camera_*.txt")
    print("  - right_camera_*.txt")

    left_cam.close()
    right_cam.close()


if __name__ == "__main__":
    main()
