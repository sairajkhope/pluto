"""
Skill classifier evaluation script.
Evaluates trained skill classifiers on real-world test data.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from toddlerbot.skill_classifier.config import SKILL_CLASSES


def load_test_data(
    data_dirs: List[str],
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Load test depth maps from NPY files.

    Args:
        data_dirs: List of directories containing depth NPY files (e.g., depth/engine_name/)

    Returns:
        Tuple of (depth_maps, labels, filenames)
    """
    depth_maps = []
    labels = []
    filenames = []

    # Pattern to extract skill label from filename
    # Format: {frame_idx}_depth_{skill_label}.npy
    pattern = re.compile(r"^\d+_depth_(.+)\.npy$")

    print(f"Loading test data from {len(data_dirs)} directories...")

    for data_dir in data_dirs:
        data_path = Path(data_dir)

        if not data_path.exists():
            print(f"Warning: Directory does not exist, skipping: {data_dir}")
            continue

        # Check if this is a test set by reading metadata
        collection_root = data_path.parent.parent
        metadata_path = collection_root / "rgb" / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            is_test = metadata.get("dataset_info", {}).get("is_test_set", False)
            if not is_test:
                print(f"Warning: Not a test set, skipping: {data_dir}")
                continue

        print(f"\n  Loading from TEST SET: {data_dir}")
        dir_count = 0

        for npy_file in sorted(data_path.glob("*.npy")):
            match = pattern.match(npy_file.name)
            if not match:
                # Skip non-depth files (combined visualizations, etc.)
                continue

            skill_label = match.group(1)

            # Verify skill is in SKILL_CLASSES
            if skill_label not in SKILL_CLASSES:
                print(
                    f"Warning: Unknown skill '{skill_label}' - skipping {npy_file.name}"
                )
                continue

            # Load raw depth map from NPY (float32, meters, unprocessed from Foundation Stereo)
            depth_raw = np.load(str(npy_file))
            if depth_raw is None or depth_raw.size == 0:
                print(f"Warning: Could not load depth map: {npy_file}")
                continue

            # Store RAW depth - will preprocess before passing to predictor
            depth_maps.append(depth_raw)
            labels.append(skill_label)
            filenames.append(npy_file.name)
            dir_count += 1

        print(f"    Loaded {dir_count} depth maps")

    print(f"\nTotal: Loaded {len(depth_maps)} test depth maps")

    # Print label distribution
    from collections import Counter

    label_counts = Counter(labels)
    print("\nTest data distribution:")
    for skill, count in sorted(label_counts.items()):
        print(f"  {skill}: {count} depth maps")

    return depth_maps, labels, filenames


def evaluate_model(
    model_path: str,
    test_images: List[np.ndarray],
    true_labels: List[str],
    device: str = "cuda",
) -> Tuple[List[str], float, List[float]]:
    """Evaluate model on test data using ONNX Runtime.

    Args:
        model_path: Path to model checkpoint (or model name in ckpts/)
        test_images: List of raw test depth maps
        true_labels: List of true skill labels
        device: Device to run on

    Returns:
        Tuple of (predicted_labels, accuracy, inference_times_list_ms)
    """
    from toddlerbot.skill_classifier.inference.predictor import SkillPredictor
    from toddlerbot.skill_classifier.inference.skill_mapper import (
        _resolve_model_path,
    )

    predictions = []
    correct = 0
    inference_times = []

    # Resolve model path (checks ckpts/, downloads from wandb if needed)
    resolved_path = _resolve_model_path(model_path)
    print(f"\nLoading ONNX model: {resolved_path}")
    predictor = SkillPredictor(model_path=resolved_path, device=device)

    print(f"\nEvaluating {len(test_images)} depth maps with ONNX...")

    for i, (depth_map, true_label) in enumerate(zip(test_images, true_labels)):
        # Preprocess depth before passing to predictor
        from toddlerbot.sim.terrain.get_depth import process_depth_map
        depth_processed = process_depth_map(depth_map, add_noise=False, max_depth=1.0)

        result = predictor.predict_from_depth(depth_processed)

        pred_label = result["predicted_skill"]
        inference_time = result["inference_time"] * 1000  # Convert to ms

        predictions.append(pred_label)
        inference_times.append(inference_time)

        # Check accuracy
        if pred_label == true_label:
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_images)} depth maps")

    accuracy = correct / len(test_images)
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(test_images)})")
    print(f"Inference Time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
    print(f"  Min: {np.min(inference_times):.2f} ms")
    print(f"  Max: {np.max(inference_times):.2f} ms")
    print(f"  Median: {np.median(inference_times):.2f} ms")

    return predictions, accuracy, inference_times


def save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
):
    """Save confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save PNG file
        title: Title for the plot
    """
    # Get unique classes present in data
    all_labels = sorted(set(y_true + y_pred))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_labels,
        yticklabels=all_labels,
        cbar_kws={"label": "Count"},
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def save_results(
    output_dir: Path,
    model_name: str,
    accuracy: float,
    inference_times: List[float],
    predictions: List[str],
    true_labels: List[str],
    filenames: List[str],
):
    """Save evaluation results to JSON and confusion matrix.

    Args:
        output_dir: Output directory
        model_name: Name of evaluated model
        accuracy: Overall accuracy
        inference_times: List of all inference times in ms
        predictions: Predicted labels
        true_labels: True labels
        filenames: Image filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results JSON with inference profiling
    results = {
        "model": model_name,
        "accuracy": float(accuracy),
        "inference_time_ms": {
            "mean": float(np.mean(inference_times)),
            "std": float(np.std(inference_times)),
            "min": float(np.min(inference_times)),
            "max": float(np.max(inference_times)),
            "median": float(np.median(inference_times)),
        },
        "num_samples": len(predictions),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    # Save detailed predictions
    predictions_data = []
    for fname, true, pred in zip(filenames, true_labels, predictions):
        predictions_data.append(
            {"filename": fname, "true_label": true, "predicted_label": pred}
        )

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions_data, f, indent=2)

    print(f"Detailed predictions saved to {predictions_path}")

    # Save classification report
    report = classification_report(true_labels, predictions)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Classification report saved to {report_path}")
    print(f"\n{report}")

    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(
        true_labels,
        predictions,
        str(cm_path),
        title=f"Confusion Matrix - {model_name}\nAccuracy: {accuracy:.2%}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate skill classifier on real-world test data"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model checkpoint (or model name in ckpts/)",
    )
    parser.add_argument(
        "test_data_dirs",
        type=str,
        nargs="+",
        help="Collection directories containing test data (e.g., collect_real_world_skill_data_20251102_165458)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Foundation Stereo engine name (e.g., foundation_stereo_vitl_192x256_16.engine). If provided, will look for depth/{engine}/ subdirectory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: results/skill_classifier_evaluation_{timestamp})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Skill Classifier Evaluation")
    print("=" * 80)

    # Build depth data directories from collection dirs + engine name (same as train.py)
    processed_data_dirs = []

    if args.engine:
        # Collection dir + engine mode: collection_dir/depth/engine_name/
        print(f"Using engine: {args.engine}")
        for data_dir in args.test_data_dirs:
            collection_path = Path(data_dir)
            depth_dir = collection_path / "depth" / args.engine

            if not depth_dir.exists():
                print(f"Warning: Depth directory does not exist: {depth_dir}")
                continue

            processed_data_dirs.append(str(depth_dir))
            print(f"  Adding test depth data: {depth_dir}")
    else:
        # Direct depth directory mode (backward compatibility)
        processed_data_dirs = args.test_data_dirs
        print("Using direct depth directories (no engine specified)")

    if not processed_data_dirs:
        print("Error: No valid depth directories found!")
        return

    # Load test data
    test_images, true_labels, filenames = load_test_data(processed_data_dirs)

    if len(test_images) == 0:
        print("Error: No test images loaded!")
        return

    # Evaluate model (always uses ONNX)
    predictions, accuracy, inference_times = evaluate_model(
        args.model_path,
        test_images,
        true_labels,
        device=args.device,
    )

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        time_str = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/skill_classifier_evaluation_{time_str}")

    # Save results
    model_name = Path(args.model_path).stem
    save_results(
        output_dir,
        model_name,
        accuracy,
        inference_times,
        predictions,
        true_labels,
        filenames,
    )

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
