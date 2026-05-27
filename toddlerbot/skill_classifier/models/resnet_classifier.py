"""ResNet-based skill classifier for depth images."""

from typing import Any, Dict

import torch
import torch.nn as nn

from toddlerbot.skill_classifier.config import MODEL_CONFIG, NUM_CLASSES, SKILL_CLASSES


class SkillClassifier(nn.Module):
    """ResNet-based classifier for predicting robot skills from depth images.

    Takes single-channel depth images as input and predicts which skill/policy should be executed.
    Trained from scratch without ImageNet pretrained weights.
    Native resolution input (e.g., 192x256 or 480x640) supported via Global Average Pooling.
    """

    def __init__(
        self, num_classes: int = None, backbone: str = None, dropout: float = None
    ):
        """Initialize the skill classifier.

        Args:
            num_classes: Number of skill classes (uses config default if None)
            backbone: ResNet backbone architecture (uses config default if None)
            dropout: Dropout rate for regularization (uses config default if None)
        """
        super().__init__()
        # Use config defaults if not specified
        self.num_classes = num_classes or NUM_CLASSES
        self.backbone_name = backbone or MODEL_CONFIG["backbone"]
        dropout = dropout or MODEL_CONFIG["dropout"]

        # Load ResNet backbone without pretrained weights
        try:
            import torchvision.models as models

            torchvision_available = True
        except (RuntimeError, ImportError):
            torchvision_available = False

        if self.backbone_name == "resnet18":
            if torchvision_available:
                try:
                    self.backbone = models.resnet18(weights=None)
                except (RuntimeError, AttributeError):
                    self.backbone = self._create_resnet18()
            else:
                self.backbone = self._create_resnet18()
            feature_dim = 512
        elif self.backbone_name == "resnet34":
            if torchvision_available:
                try:
                    self.backbone = models.resnet34(weights=None)
                except (RuntimeError, AttributeError):
                    self.backbone = self._create_resnet34()
            else:
                self.backbone = self._create_resnet34()
            feature_dim = 512
        elif self.backbone_name == "resnet50":
            if torchvision_available:
                try:
                    self.backbone = models.resnet50(weights=None)
                except (RuntimeError, AttributeError):
                    self.backbone = self._create_resnet50()
            else:
                self.backbone = self._create_resnet50()
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Modify first conv layer to accept single-channel depth input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace the final classification layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, self.num_classes),
        )

        # Use centralized skill class mapping
        self.skill_classes = SKILL_CLASSES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Make predictions with confidence scores.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Dictionary containing predictions, probabilities, and confidence
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]

            # Convert to skill names
            skill_names = [self.skill_classes[pred.item()] for pred in predictions]

            return {
                "predictions": predictions.cpu().numpy(),
                "skill_names": skill_names,
                "probabilities": probabilities.cpu().numpy(),
                "confidences": confidences.cpu().numpy(),
                "logits": logits.cpu().numpy(),
            }

    def get_skill_name(self, class_idx: int) -> str:
        """Convert class index to skill name.

        Args:
            class_idx: Class index

        Returns:
            Skill name string
        """
        if 0 <= class_idx < len(self.skill_classes):
            return self.skill_classes[class_idx]
        else:
            return "unknown"

    def save_checkpoint(
        self, filepath: str, epoch: int, optimizer=None, loss: float = None
    ):
        """Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current training epoch
            optimizer: Optimizer state (optional)
            loss: Current loss value (optional)
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "backbone": self.backbone_name,
            "skill_classes": self.skill_classes,
            "epoch": epoch,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if loss is not None:
            checkpoint["loss"] = loss

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = "cpu"):
        """Load model from checkpoint.

        Args:
            filepath: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded SkillClassifier model
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Create model with saved configuration
        model = cls(
            num_classes=checkpoint["num_classes"], backbone=checkpoint["backbone"]
        )

        # Load trained weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.skill_classes = checkpoint["skill_classes"]

        # Move model to specified device
        model = model.to(device)

        print(
            f"Model loaded from {filepath} (epoch {checkpoint.get('epoch', 'unknown')}) on device {device}"
        )
        return model

    def _create_resnet18(self):
        """Create a ResNet-18 model without using torchvision."""
        return self._create_resnet([2, 2, 2, 2], 512)

    def _create_resnet34(self):
        """Create a ResNet-34 model without using torchvision."""
        return self._create_resnet([3, 4, 6, 3], 512)

    def _create_resnet50(self):
        """Create a ResNet-50 model without using torchvision."""
        return self._create_resnet([3, 4, 6, 3], 2048, bottleneck=True)

    def _create_resnet(self, layers, feature_dim, bottleneck=False):
        """Create a basic ResNet model structure for loading checkpoints."""

        # Simple ResNet structure for checkpoint loading compatibility
        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)

                self.downsample = None
                if stride != 1 or inplanes != planes:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                        nn.BatchNorm2d(planes),
                    )

            def forward(self, x):
                residual = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.downsample is not None:
                    residual = self.downsample(x)
                return self.relu(out + residual)

        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes * 4)
                self.relu = nn.ReLU(inplace=True)

                self.downsample = None
                if stride != 1 or inplanes != planes * 4:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(inplanes, planes * 4, 1, stride, bias=False),
                        nn.BatchNorm2d(planes * 4),
                    )

            def forward(self, x):
                residual = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                if self.downsample is not None:
                    residual = self.downsample(x)
                return self.relu(out + residual)

        class ResNet(nn.Module):
            def __init__(self, block, layers):
                super().__init__()
                self.inplanes = 64
                self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, 2, 1)

                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, 1000)  # Will be replaced

            def _make_layer(self, block, planes, blocks, stride=1):
                layers = []
                layers.append(block(self.inplanes, planes, stride))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))
                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)

        block = Bottleneck if bottleneck else BasicBlock
        return ResNet(block, layers)


def create_skill_classifier(
    num_classes: int = None, backbone: str = None, device: str = "cuda"
) -> SkillClassifier:
    """Factory function to create a skill classifier.

    Args:
        num_classes: Number of skill classes
        backbone: ResNet backbone architecture
        device: Device to place model on

    Returns:
        SkillClassifier model on specified device
    """
    model = SkillClassifier(num_classes=num_classes, backbone=backbone)

    # Move to device - handle both string and torch.device
    if isinstance(device, str):
        if torch.cuda.is_available() and device == "cuda":
            model = model.cuda()
        else:
            model = model.cpu()
    else:
        # device is torch.device object
        model = model.to(device)

    return model
