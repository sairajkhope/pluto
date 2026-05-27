"""Test Butterworth filter implementation on logged IMU data."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter

from toddlerbot.utils.math_utils import butterworth


def load_log_data(file_path):
    """Load log data from LZ4 compressed file."""
    data = joblib.load(file_path)
    return data


def get_butterworth_coefficients(cutoff_freq, sampling_freq, order=4):
    """Get Butterworth filter coefficients."""
    nyquist = 0.5 * sampling_freq
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return b, a


def apply_butterworth_filter(b, a, signal):
    """Apply Butterworth filter to signal using the custom implementation."""
    filtered_signal = np.zeros_like(signal)

    # Apply filter to each axis separately
    for axis in range(signal.shape[1]):
        # Initialize state for filter history for this axis
        past_inputs = np.zeros((len(b) - 1, 1))
        past_outputs = np.zeros((len(a) - 1, 1))

        for i in range(len(signal)):
            y, past_inputs, past_outputs = butterworth(
                b, a, signal[i, axis], past_inputs, past_outputs
            )
            filtered_signal[i, axis] = y.item()

    return filtered_signal


def main():
    """Load IMU data, apply Butterworth filter, and save filtered results."""
    # Load the log data
    log_data_path = (
        "results/toddlerbot_2xc_replay_real_world_20250814_133142/log_data.lz4"
    )
    data = load_log_data(log_data_path)

    # Extract observations list
    obs_list = data["obs_list"]

    # Extract time and angular velocity data
    time_data = []
    ang_vel_data = []

    for obs in obs_list:
        time_data.append(obs.time)
        ang_vel_data.append(obs.ang_vel)

    time_data = np.array(time_data)
    ang_vel_data = np.array(ang_vel_data)  # Shape: (n_samples, 3) for [x, y, z]

    print(f"Loaded {len(time_data)} data points")
    print(f"Time range: {time_data[0]:.3f} to {time_data[-1]:.3f} seconds")
    print(f"Angular velocity shape: {ang_vel_data.shape}")

    # Calculate sampling frequency
    dt = np.mean(np.diff(time_data))
    sampling_freq = 1.0 / dt
    print(f"Average sampling frequency: {sampling_freq:.1f} Hz")

    # Set filter parameters
    cutoff_freq = 10.0  # Hz
    filter_order = 4

    # Get Butterworth coefficients
    b, a = get_butterworth_coefficients(cutoff_freq, sampling_freq, filter_order)

    # Apply the filter
    filtered_ang_vel = apply_butterworth_filter(b, a, ang_vel_data)

    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axis_labels = ["X (Roll)", "Y (Pitch)", "Z (Yaw)"]

    for i in range(3):
        axes[i].plot(
            time_data,
            ang_vel_data[:, i],
            "b-",
            alpha=0.7,
            linewidth=1,
            label="Original",
        )
        axes[i].plot(
            time_data,
            filtered_ang_vel[:, i],
            "r-",
            linewidth=2,
            label=f"Butterworth Filtered ({cutoff_freq} Hz)",
        )
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Angular Velocity (rad/s)")
        axes[i].set_title(f"Angular Velocity {axis_labels[i]} - Original vs Filtered")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\nFiltering Statistics:")
    for i, axis in enumerate(["X", "Y", "Z"]):
        orig_std = np.std(ang_vel_data[:, i])
        filt_std = np.std(filtered_ang_vel[:, i])
        noise_reduction = (1 - filt_std / orig_std) * 100
        print(
            f"{axis}-axis: Original std={orig_std:.4f}, Filtered std={filt_std:.4f}, "
            f"Noise reduction: {noise_reduction:.1f}%"
        )

    output_path = (
        "results/toddlerbot_2xc_replay_real_world_20250814_133143/log_data.lz4"
    )
    for i in range(len(obs_list)):
        data["obs_list"][i].ang_vel = filtered_ang_vel[i]

    joblib.dump(data, output_path)
    print(f"Filtered data saved to {output_path}")


if __name__ == "__main__":
    main()
