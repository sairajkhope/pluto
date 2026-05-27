"""Real-time skill prediction from depth maps using ONNX Runtime."""

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort


class SkillPredictor:
    """Real-time skill prediction from depth maps using ONNX Runtime.

    Uses single-channel depth at native resolution (e.g., 192x246).
    NO resize, NO ImageNet normalization - matches training pipeline.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        """Initialize the skill predictor.

        Args:
            model_path: Path to ONNX model (.onnx file) or .pth (will auto-convert to .onnx)
            device: Device to run inference on ("cuda" or "cpu")
        """
        model_path = Path(model_path)

        # Auto-resolve .onnx path if .pth provided
        if model_path.suffix == ".pth":
            model_path = model_path.parent / model_path.name.replace(".pth", ".onnx")

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        print(f"Loading ONNX model from {model_path}")

        # Setup ONNX Runtime session
        providers = []
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.ort_session = ort.InferenceSession(str(model_path), providers=providers)

        # Get model input/output info
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # Load skill classes from .pth checkpoint
        pth_path = model_path.parent / model_path.name.replace(".onnx", ".pth")
        if pth_path.exists():
            import torch
            checkpoint = torch.load(pth_path, map_location="cpu")
            self.skill_classes = checkpoint.get("skill_classes", [])
        else:
            # Fallback to config
            from toddlerbot.skill_classifier.config import SKILL_CLASSES
            self.skill_classes = SKILL_CLASSES

        print(f"SkillPredictor initialized on ONNX Runtime ({providers[0]})")
        print(f"Model classes: {self.skill_classes}")

    @property
    def expected_depth_resolution(self) -> tuple:
        """Get expected depth resolution BEFORE cropping.

        Returns:
            (width, height) tuple for setup_depth_capture() resolution parameter

        Note: ONNX input shape is [batch, channels, height, width_after_crop].
        Since process_depth_map() crops 10px from left, we add 10 back to get pre-crop width.
        # NOTE: Hardcoded 10px crop offset - must match process_depth_map() in get_depth.py
        """
        input_shape = self.ort_session.get_inputs()[0].shape
        # input_shape format: [batch, channels, height, width_after_crop]
        height = input_shape[2]
        width_after_crop = input_shape[3]
        width_before_crop = width_after_crop + 10  # NOTE: Hardcoded 10px left crop offset
        return (width_before_crop, height)

    def predict_from_depth(self, depth_map: np.ndarray) -> Dict[str, Any]:
        """Predict skill from PREPROCESSED depth map.

        Args:
            depth_map: Preprocessed depth map (float32, [0, 1.0] meters) after process_depth_map()
                      Caller must apply process_depth_map() before passing to this function

        Returns:
            Dictionary with raw prediction results
        """
        start_time = time.time()

        # Depth is already preprocessed by caller (cropped, clipped to 1.0m)
        # Just normalize to [0, 1] range
        depth_normalized = depth_map / 1.0

        # Convert to (1, 1, H, W) for ONNX: batch + single-channel
        input_array = depth_normalized[np.newaxis, np.newaxis, ...].astype(np.float32)

        # Run inference
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_array})
        logits = outputs[0][0]  # Remove batch dimension

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Get prediction
        pred_idx = np.argmax(probabilities)
        skill_name = self.skill_classes[pred_idx]
        confidence = probabilities[pred_idx]

        inference_time = time.time() - start_time

        return {
            "predicted_skill": skill_name,
            "confidence": float(confidence),
            "probabilities": probabilities,
            "inference_time": inference_time,
        }
