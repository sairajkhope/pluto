"""Configuration for skill classifier training and inference.

MINIMUM DATASET REQUIREMENTS:
- Total: 10,000+ images across all skill classes
- Per class: 1,000+ images minimum for reliable training
"""

# Centralized skill class definitions
SKILL_CLASSES = [
    "walk",
    "get_down_chair",
    "crawl_chair",
    "get_up_chair",
    "climb_wall",
    "climb_up_box",
    "crawl_box",
    "rotate_box",
    "crawl_up_stairs",
]

# Create mapping dictionaries
SKILL_TO_IDX = {skill: idx for idx, skill in enumerate(SKILL_CLASSES)}
IDX_TO_SKILL = {idx: skill for idx, skill in enumerate(SKILL_CLASSES)}

# Number of classes
NUM_CLASSES = len(SKILL_CLASSES)

# Model configuration for depth images
MODEL_CONFIG = {
    "backbone": "resnet34",
    "dropout": 0.3,
    "num_classes": NUM_CLASSES,
    # Note: input_size removed - ResNet accepts any resolution via Global Average Pooling
    # Actual input shape determined by depth data (e.g., 192x256 or 480x640)
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 128,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "patience": 10,
    "val_split": 0.2,
    "random_state": 42,
    "warmup_epochs": 3,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.7,
    "gradient_clip": 1.0,
}

# Inference configuration
INFERENCE_CONFIG = {
    "confidence_threshold": 0.7,  # Minimum confidence for predictions
    "history_size": 3,  # Prediction smoothing window
    # Note: input_size removed - model accepts native resolution from depth camera
}

# Wandb configuration (consistent with locomotion training)
WANDB_CONFIG = {
    "project": "ToddlerBot",
    "entity": "toddlerbot",
}
