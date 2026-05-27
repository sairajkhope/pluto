"""Training script for skill classifier."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import wandb
from toddlerbot.skill_classifier.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    WANDB_CONFIG,
)
from toddlerbot.skill_classifier.data.dataset import create_datasets
from toddlerbot.skill_classifier.models.resnet_classifier import create_skill_classifier


class Tee:
    """Class to redirect output to both terminal and log file."""

    def __init__(self, log_path):
        self.terminal = sys.__stdout__  # Use sys.__stdout__ to avoid redirect loops
        self.log = open(log_path, "w", buffering=1)  # Line-buffered
        self.closed = False

    def write(self, message):
        msg = str(message)
        try:
            self.terminal.write(msg)
            self.terminal.flush()
            if not self.closed:
                self.log.write(msg)
                self.log.flush()
        except (ValueError, OSError):
            self.closed = True

    def flush(self):
        try:
            self.terminal.flush()
            if not self.closed:
                self.log.flush()
        except (ValueError, OSError):
            self.closed = True

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    def close(self):
        if not self.closed:
            try:
                self.log.close()
            except Exception:
                pass
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
):
    """Save confusion matrix as PNG file.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        output_path: Path to save PNG file
        title: Title for the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate for one epoch.

    Returns:
        Tuple of (average_loss, accuracy, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def train_skill_classifier(
    data_dirs: str | List[str],
    output_dir: str = "results/skill_classifier",
    device: str = "auto",
    use_wandb: bool = True,
    wandb_name: str = None,
    wandb_notes: str = "",
    resume_from: str = None,
    add_noise: bool = False,
) -> str:
    """Train the skill classifier.

    Args:
        data_dirs: Directory or list of directories containing depth images
        output_dir: Directory to save model and training results
        device: Device to train on ("auto", "cuda", or "cpu")
        use_wandb: Whether to use wandb for logging
        wandb_name: Custom name for wandb run
        wandb_notes: Notes for wandb run
        resume_from: Path to checkpoint to resume training from
        add_noise: If True, applies local noise + blur augmentation to BOTH train and val (default: False)

    Returns:
        Path to saved model checkpoint
    """

    # Setup
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Training on device: {device}")
    print(f"Data directories: {data_dirs}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    with Tee(os.path.join(output_dir, "output.log")) as tee_output:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = tee_output
        sys.stderr = sys.stdout
        # Initialize wandb
        if use_wandb:
            # Generate run name if not provided
            if wandb_name is None:
                time_str = time.strftime("%Y%m%d_%H%M%S")
                wandb_name = f"skill_classifier_{time_str}"

            wandb.init(
                project=WANDB_CONFIG["project"],
                entity=WANDB_CONFIG["entity"],
                name=wandb_name,
                notes=wandb_notes,
                job_type="train",
                config={
                    "model": MODEL_CONFIG,
                    "training": TRAINING_CONFIG,
                    "data_dirs": data_dirs
                    if isinstance(data_dirs, list)
                    else [data_dirs],
                    "output_dir": str(output_dir),
                    "device": str(device),
                    "add_noise": add_noise,
                },
            )

            # Define metrics
            wandb.define_metric("epoch")
            wandb.define_metric("train/loss", step_metric="epoch")
            wandb.define_metric("train/accuracy", step_metric="epoch")
            wandb.define_metric("val/loss", step_metric="epoch")
            wandb.define_metric("val/accuracy", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")

        # Create datasets
        print("Creating datasets...")
        if add_noise:
            print(
                "Noise augmentation ENABLED: applying local noise + blur to BOTH train and val data"
            )
        else:
            print("Noise augmentation DISABLED: using clean depth images")

        train_dataset, val_dataset = create_datasets(
            data_dirs=data_dirs,
            test_size=TRAINING_CONFIG["val_split"],
            random_state=TRAINING_CONFIG["random_state"],
            add_noise=add_noise,
        )

        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")

        # Get actual input shape from dataset (varies based on Foundation Stereo engine)
        sample_input, _ = train_dataset[0]
        input_shape = tuple(
            sample_input.shape
        )  # (C, H, W) e.g., (1, 192, 256) or (1, 480, 640)
        print(f"Input shape: {input_shape}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
            num_workers=4,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=4,
        )

        # Create model or resume from checkpoint
        start_epoch = 0
        best_val_loss = float("inf")
        best_val_acc = 0.0

        if resume_from:
            print(f"Resuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)

            # Load model from checkpoint
            from toddlerbot.skill_classifier.models.resnet_classifier import (
                SkillClassifier,
            )

            model = SkillClassifier.load_checkpoint(resume_from, device=str(device))
            print(f"Loaded model from checkpoint at epoch {checkpoint.get('epoch', 0)}")

            # Extract training state if available
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("loss", float("inf"))
            # Note: We don't track best_val_acc in checkpoint, so we'll start fresh
            print(
                f"Resuming from epoch {start_epoch}, previous best val loss: {best_val_loss:.4f}"
            )
        else:
            print("Creating new model...")
            model = create_skill_classifier(device=str(device))

        print(
            f"Model: {MODEL_CONFIG['backbone']} with {sum(p.numel() for p in model.parameters())} parameters"
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # Use class weights if dataset is imbalanced
        class_weights = train_dataset.get_class_weights()
        if class_weights.max() / class_weights.min() > 2.0:  # If imbalanced
            print("Using class weights for imbalanced dataset")
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

        # Resume optimizer state if available
        if resume_from and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Resumed optimizer state")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training loop
        print("Starting training...")
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(start_epoch, TRAINING_CONFIG["num_epochs"]):
            start_time = time.time()

            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_acc, val_preds, val_targets = validate_epoch(
                model, val_loader, criterion, device
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch + 1}/{TRAINING_CONFIG['num_epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.1f}s, LR: {current_lr:.6f}")

            # Log to wandb
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "lr": current_lr,
                        "epoch_time": epoch_time,
                    }
                )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                model_path = output_dir / "best_model.pth"
                model.save_checkpoint(str(model_path), epoch, optimizer, val_loss)

                # Export to ONNX
                try:
                    onnx_path = str(model_path).replace(".pth", ".onnx")
                    dummy_input = torch.zeros(1, *input_shape).to(device)
                    model.eval()
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={
                            "input": {0: "batch_size"},
                            "output": {0: "batch_size"},
                        },
                    )
                    print(f"  ✓ ONNX model exported to {onnx_path}")
                except Exception as e:
                    print(f"  ⚠ ONNX export failed: {e}")

                # Save detailed classification report
                # Get unique classes present in validation data
                unique_classes = sorted(set(val_targets))
                present_skill_names = [
                    train_dataset.idx_to_skill[i] for i in unique_classes
                ]

                report = classification_report(
                    val_targets,
                    val_preds,
                    target_names=present_skill_names,
                    labels=unique_classes,
                    output_dict=True,
                )

                report_path = output_dir / "classification_report.txt"
                with open(report_path, "w") as f:
                    f.write(
                        classification_report(
                            val_targets,
                            val_preds,
                            target_names=present_skill_names,
                            labels=unique_classes,
                        )
                    )

                # Save confusion matrix
                cm_path = output_dir / "confusion_matrix.png"
                save_confusion_matrix(
                    val_targets,
                    val_preds,
                    present_skill_names,
                    str(cm_path),
                    f"Confusion Matrix (Best Model - Epoch {epoch + 1}, Acc: {val_acc:.2f}%)",
                )

                # Log best model metrics to wandb (but don't upload artifact yet)
                if use_wandb:
                    wandb.log(
                        {"best_val_accuracy": best_val_acc, "best_val_loss": val_loss}
                    )

                print(f"  ✓ New best model saved! Val Acc: {best_val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= TRAINING_CONFIG["patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            print()

        # Final model save
        final_model_path = output_dir / "final_model.pth"
        model.save_checkpoint(str(final_model_path), epoch, optimizer, val_loss)

        # Save training history
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "best_val_acc": best_val_acc,
            "config": {
                "model": MODEL_CONFIG,
                "training": TRAINING_CONFIG,
                "input_shape": input_shape,
            },
        }

        history_path = output_dir / "training_history.pt"
        torch.save(history, history_path)

        print("Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: {output_dir}")

        # Upload best model artifact to wandb (only once at end)
        if use_wandb:
            wandb.log(
                {
                    "final/best_val_accuracy": best_val_acc,
                    "final/total_epochs": epoch + 1,
                }
            )

            # Save best model as wandb artifact (like RL training - only at end)
            print("Uploading best model to wandb...")
            best_model_path = output_dir / "best_model.pth"
            best_onnx_path = output_dir / "best_model.onnx"
            report_path = output_dir / "classification_report.txt"

            artifact = wandb.Artifact(
                name=wandb_name,  # wandb_name already has skill_classifier_ prefix
                type="model",
                metadata={
                    "validation_accuracy": best_val_acc,
                    "final_epoch": epoch + 1,
                    "num_classes": MODEL_CONFIG["num_classes"],
                    "backbone": MODEL_CONFIG["backbone"],
                },
            )

            # Add best model files
            if best_model_path.exists():
                artifact.add_file(str(best_model_path))
            if best_onnx_path.exists():
                artifact.add_file(str(best_onnx_path))
            if report_path.exists():
                artifact.add_file(str(report_path))

            # Upload with "latest" alias (only 1 artifact per training run)
            wandb.log_artifact(artifact, aliases=["latest", "best"])
            print("✓ Model artifact uploaded to wandb")

            wandb.finish()

        return str(output_dir / "best_model.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train skill classifier")
    parser.add_argument(
        "data_dirs",
        nargs="+",
        help="Collection directories (e.g., collect_real_world_skill_data_20251031_223001) or direct depth directories",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Depth subdirectory name. For real-world: Foundation Stereo engine (e.g., foundation_stereo_vitl_192x256_16.engine). For simulation: resolution (e.g., 256x192). If provided, will look for depth/{engine}/ subdirectory.",
    )
    parser.add_argument(
        "--output", default="results/skill_classifier", help="Output directory"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-name", default=None, help="Custom wandb run name")
    parser.add_argument("--notes", default="", help="Notes for wandb run")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--add-noise",
        action="store_true",
        help="Apply local noise + blur augmentation during training",
    )

    args = parser.parse_args()

    # Build depth data directories from collection dirs + engine name
    processed_data_dirs = []

    if args.engine:
        # Collection dir + engine/resolution mode: collection_dir/depth/{engine_or_resolution}/
        # For real-world: engine = foundation_stereo_vitl_192x256_16.engine
        # For simulation: engine = 256x192 (resolution string)
        print(f"Using depth subdirectory: {args.engine}")
        for data_dir in args.data_dirs:
            collection_path = Path(data_dir)
            depth_dir = collection_path / "depth" / args.engine

            if not depth_dir.exists():
                print(f"Warning: Depth directory does not exist: {depth_dir}")
                print(
                    f"  Make sure you've run create_depth_data.py with --engine-path {args.engine}"
                )
                continue

            processed_data_dirs.append(str(depth_dir))
            print(f"  Adding depth data: {depth_dir}")
    else:
        # Direct depth directory mode (backward compatibility)
        processed_data_dirs = args.data_dirs
        print("Using direct depth directories (no engine specified)")

    if not processed_data_dirs:
        print("Error: No valid depth directories found!")
        if args.engine:
            print("Please run create_depth_data.py first to generate depth maps.")
        sys.exit(1)

    print(f"\nTraining with {len(processed_data_dirs)} depth dataset(s)")

    # Handle timestamped output directory like train_mjx.py
    output_dir = args.output
    if output_dir == "results/skill_classifier":
        # Generate timestamped directory when using default
        time_str = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/skill_classifier_{time_str}"

    # Train model
    model_path = train_skill_classifier(
        data_dirs=processed_data_dirs,
        output_dir=output_dir,
        device=args.device,
        use_wandb=not args.no_wandb,
        wandb_name=args.wandb_name,
        wandb_notes=args.notes,
        resume_from=args.resume,
        add_noise=args.add_noise,
    )

    print(f"Training completed. Best model: {model_path}")
