"""Dataset class for loading depth images with skill labels."""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from toddlerbot.sim.terrain.get_depth import process_depth_map
from toddlerbot.skill_classifier.config import (
    IDX_TO_SKILL,
    NUM_CLASSES,
    SKILL_CLASSES,
    SKILL_TO_IDX,
)


class DepthDataset(Dataset):
    """Dataset for depth images with terrain skill labels.

    Expected filename formats:
        - Real-world: {frame_idx}_depth_{skill_label}.npy
        - Simulation: {index}_cam_{cam_idx}_depth_{skill}.npy
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        add_noise: bool = False,
    ):
        """Initialize depth dataset with pre-split data.

        Args:
            image_paths: List of paths to depth NPY files
            labels: List of skill label indices corresponding to image_paths
            add_noise: If True, applies local noise + blur augmentation during training (default: False)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.add_noise = add_noise

        # Use centralized skill mapping
        self.skill_to_idx = SKILL_TO_IDX
        self.idx_to_skill = IDX_TO_SKILL
        self.skill_classes = SKILL_CLASSES
        self.num_classes = NUM_CLASSES

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single data sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (depth_tensor, label) where depth_tensor is single-channel float32
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load raw depth from NPY file (float32, meters, unprocessed from Foundation Stereo)
        # Shape: (H, W), e.g., (192, 256) or (480, 640)
        # Values: raw depth in meters (0-1.3m range, not yet clipped/cropped)
        depth_raw = np.load(str(img_path))

        if depth_raw is None or depth_raw.size == 0:
            raise ValueError(f"Could not load depth data: {img_path}")

        # Apply preprocessing ONCE: crop left 10px, clip to max_depth, optional noise augmentation
        # This matches simulation depth processing
        depth_processed = process_depth_map(
            depth_raw,
            add_noise=self.add_noise,  # Apply noise only during training if --add-noise flag set
            max_depth=1.0,  # Clip to 1.0m range
        )
        # After processing: (H, W_cropped), e.g., (192, 246), values in [0, 1.0] meters

        # Normalize to [0, 1] range (keep as float32, NO uint8 conversion!)
        depth_normalized = depth_processed / 1.0  # Now in [0.0, 1.0] range

        # Convert to tensor: (H, W) → (1, H, W) single-channel
        # NO transforms, NO resize - feed native resolution directly to ResNet!
        depth_tensor = torch.from_numpy(depth_normalized[np.newaxis, ...]).float()

        return depth_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling class imbalance."""
        label_counts = np.bincount(self.labels, minlength=self.num_classes)
        total_samples = len(self.labels)

        # Inverse frequency weighting
        weights = total_samples / (self.num_classes * label_counts + 1e-6)
        return torch.FloatTensor(weights)


def _load_all_depth_paths(data_dirs: str | List[str]) -> Tuple[List[Path], List[int]]:
    """Load all depth NPY paths and extract labels from filenames.

    This function is called ONCE to load all paths, then split is done in create_datasets.

    Args:
        data_dirs: Directory or list of directories containing depth NPY files

    Returns:
        Tuple of (image_paths, labels)
    """
    # Support both single directory and multiple directories
    if isinstance(data_dirs, str):
        dirs = [Path(data_dirs)]
    else:
        dirs = [Path(d) for d in data_dirs]

    image_paths = []
    labels = []

    # Pattern to extract skill label from filename
    # Supports two formats:
    #   1. Real-world: {frame_idx}_depth_{skill_label}.npy (e.g., 0_depth_walk.npy)
    #   2. Simulation: {index}_cam_{cam_idx}_depth_{skill}.npy (e.g., 00000000_cam_0_depth_walk.npy)
    pattern = re.compile(r"^\d+(?:_cam_\d+)?_depth_(.+)\.npy$")

    # Load from all data directories
    for data_dir in dirs:
        if not data_dir.exists():
            print(f"Warning: Data directory does not exist: {data_dir}")
            continue

        # Check if this is a test set by looking for metadata.json in parent directories
        # Directory structure:
        #   Real-world: collect_real_world_skill_data_TIMESTAMP/depth/ENGINE_NAME/
        #   Simulation: collect_sim_skill_data_TIMESTAMP/depth/RESOLUTION/
        # Metadata is at: COLLECTION_ROOT/rgb/metadata.json
        collection_root = data_dir.parent.parent  # Go up from depth/ENGINE_OR_RESOLUTION to collection root
        metadata_path = collection_root / "rgb" / "metadata.json"

        if metadata_path.exists():
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if metadata.get("dataset_info", {}).get("is_test_set", False):
                print(f"Skipping TEST SET: {data_dir}")
                continue

        print(f"Loading data from: {data_dir}")
        dir_image_count = 0

        # Load NPY files (depth data from Foundation Stereo)
        for img_file in data_dir.glob("*.npy"):
            match = pattern.match(img_file.name)
            if not match:
                # Skip non-depth files (combined visualizations, etc.)
                continue

            skill_label = match.group(1)

            if skill_label not in SKILL_TO_IDX:
                print(
                    f"Warning: Unknown skill label '{skill_label}' in file: {img_file.name}"
                )
                continue

            image_paths.append(img_file)
            labels.append(SKILL_TO_IDX[skill_label])
            dir_image_count += 1

        print(f"  Found {dir_image_count} valid depth files in {data_dir}")

    print(
        f"Total: Found {len(image_paths)} depth files with valid labels across {len(dirs)} directories"
    )

    # Print label distribution
    label_counts = {}
    for label in labels:
        skill = IDX_TO_SKILL[label]
        label_counts[skill] = label_counts.get(skill, 0) + 1

    print("Label distribution:")
    for skill, count in sorted(label_counts.items()):
        print(f"  {skill}: {count} images")

    return image_paths, labels


def create_datasets(
    data_dirs: str | List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    add_noise: bool = False,
) -> Tuple[DepthDataset, DepthDataset]:
    """Create train and validation datasets.

    Loads all data paths ONCE, splits ONCE, then creates two dataset objects.

    Args:
        data_dirs: Directory or list of directories containing depth NPY files
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        add_noise: If True, applies local noise + blur augmentation to BOTH train and val

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load all paths ONCE
    all_paths, all_labels = _load_all_depth_paths(data_dirs)

    if len(all_paths) == 0:
        raise ValueError("No depth files found in provided directories!")

    # Split ONCE using test_size and random_state
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=all_labels,  # Ensure balanced split across all skills
    )

    print("\nDataset split:")
    print(f"  Train: {len(train_paths)} samples")
    print(f"  Val: {len(val_paths)} samples")

    # Create datasets with pre-split data
    train_dataset = DepthDataset(
        image_paths=train_paths,
        labels=train_labels,
        add_noise=add_noise,  # Apply noise if requested
    )

    val_dataset = DepthDataset(
        image_paths=val_paths,
        labels=val_labels,
        add_noise=add_noise,  # Apply same noise to validation for realistic accuracy
    )

    return train_dataset, val_dataset
