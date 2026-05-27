"""Vision-based skill map that replaces get_robot_skill()."""

from typing import Any, Dict, Optional

import numpy as np

from toddlerbot.skill_classifier.config import INFERENCE_CONFIG
from toddlerbot.skill_classifier.inference.predictor import SkillPredictor


class SkillMapper:
    """Drop-in replacement for terrain skill map using vision-based skill classification."""

    def __init__(
        self, model_path: str, device: str = "cuda", confidence_threshold: float = None
    ):
        """Initialize vision skill map.

        Args:
            model_path: Can be either:
                - Full path to model file (e.g., "ckpts/skill_classifier_20250829_162129/best_model.pth")
                - Just the run name (e.g., "skill_classifier_20250829_162129") - will check ckpts/ and download from wandb if needed
            device: Device for inference
            confidence_threshold: Minimum confidence for predictions (uses config default if None)
        """
        # Resolve model path, checking ckpts/ and downloading from wandb if needed
        resolved_path = _resolve_model_path(model_path)

        self.predictor = SkillPredictor(
            model_path=resolved_path,
            device=device,
        )
        self.confidence_threshold = (
            confidence_threshold or INFERENCE_CONFIG["confidence_threshold"]
        )
        self.last_prediction = None

        # Prediction history for smoothing
        self.prediction_history = []
        self.history_size = INFERENCE_CONFIG["history_size"]

        print(f"SkillMapper initialized with model: {resolved_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"History size: {self.history_size}")

    def get_robot_skill(
        self, depth_image: np.ndarray, fallback_skill: str = "walk"
    ) -> Optional[str]:
        """Get skill prediction from depth image.

        This replaces get_robot_skill(robot_x, robot_y, skill_map, skill_check_name).

        Args:
            depth_image: Current depth image (0-255 grayscale)
            fallback_skill: Skill to return if prediction fails or low confidence

        Returns:
            Predicted skill name or None if should stay with current skill
        """
        try:
            # Get raw prediction from depth
            result = self.predictor.predict_from_depth(depth_image)

            raw_skill = result["predicted_skill"]
            raw_confidence = result["confidence"]
            probabilities = result["probabilities"]

            # Update history
            self.prediction_history.append(
                {
                    "skill": raw_skill,
                    "confidence": raw_confidence,
                    "probabilities": probabilities,
                }
            )

            # Keep only recent history
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)

            # Get smoothed prediction
            smoothed_result = self._get_smoothed_prediction()
            predicted_skill = smoothed_result["skill"]
            confidence = smoothed_result["confidence"]
            is_confident = confidence >= self.confidence_threshold

            # Store for debugging
            self.last_prediction = {
                "skill": predicted_skill,
                "confidence": confidence,
                "is_confident": is_confident,
                "raw_skill": raw_skill,
                "raw_confidence": raw_confidence,
                "inference_time": result.get("inference_time", 0),
                "history_size": len(self.prediction_history),
                "probabilities": smoothed_result["probabilities"],
            }

            # Return prediction if confident, otherwise fallback
            if is_confident:
                return predicted_skill
            else:
                print(
                    f"Low confidence ({confidence:.2f}) for {predicted_skill}, using fallback: {fallback_skill}"
                )
                return fallback_skill

        except Exception as e:
            print(f"Vision skill prediction failed: {e}")
            return fallback_skill

    def _get_smoothed_prediction(self) -> Dict[str, Any]:
        """Get smoothed prediction from history using weighted averaging."""
        if not self.prediction_history:
            return {"skill": "unknown", "confidence": 0.0, "probabilities": None}

        # Weight recent predictions more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.prediction_history)))
        weights = weights / weights.sum()

        # Weighted average of probabilities
        prob_arrays = [pred["probabilities"] for pred in self.prediction_history]
        weighted_probs = np.average(prob_arrays, axis=0, weights=weights)

        # Get skill with highest weighted probability
        predicted_class_idx = np.argmax(weighted_probs)
        predicted_skill = self.predictor.skill_classes[predicted_class_idx]
        confidence = weighted_probs[predicted_class_idx]

        return {
            "skill": predicted_skill,
            "confidence": confidence,
            "probabilities": weighted_probs,
        }

    def reset_history(self):
        """Reset prediction history."""
        self.prediction_history = []

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about recent predictions."""
        if not self.prediction_history:
            return {"history_empty": True}

        recent_skills = [pred["skill"] for pred in self.prediction_history]
        recent_confidences = [pred["confidence"] for pred in self.prediction_history]

        # Count skill occurrences
        skill_counts = {}
        for skill in recent_skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

        return {
            "history_size": len(self.prediction_history),
            "skill_counts": skill_counts,
            "avg_confidence": np.mean(recent_confidences),
            "min_confidence": np.min(recent_confidences),
            "max_confidence": np.max(recent_confidences),
            "most_frequent_skill": max(skill_counts, key=skill_counts.get),
            "consistency": max(skill_counts.values()) / len(recent_skills),
        }

    def get_prediction_info(self):
        """Get info about last prediction for debugging."""
        return self.last_prediction


# Global instance for easy integration
_vision_skill_map: Optional[SkillMapper] = None


def initialize_vision_skill_map(model_path: str, device: str = "cuda") -> SkillMapper:
    """Initialize global vision skill map instance.

    Args:
        model_path: Can be either:
            - Full path to model file (e.g., "ckpts/skill_classifier_20250829_162129/best_model.pth")
            - Just the run name (e.g., "skill_classifier_20250829_162129") - will check ckpts/ and download from wandb if needed
        device: Device for inference

    Returns:
        SkillMapper instance
    """
    global _vision_skill_map

    # Resolve model path, checking ckpts/ and downloading from wandb if needed
    resolved_path = _resolve_model_path(model_path)

    _vision_skill_map = SkillMapper(model_path=resolved_path, device=device)
    return _vision_skill_map


def _resolve_model_path(model_path: str) -> str:
    """Resolve model path, checking ckpts/ and downloading from wandb if needed.

    Args:
        model_path: Model path or run name

    Returns:
        Full path to model file

    Raises:
        FileNotFoundError: If model cannot be found locally or downloaded
    """
    import glob
    import os

    # If it's already a full path to an existing file, use it
    if os.path.isfile(model_path):
        print(f"Using model: {model_path}")
        return model_path

    # Check in ckpts/ directory
    ckpts_path = f"ckpts/{model_path}"
    if not model_path.startswith("ckpts/"):
        ckpts_path = f"ckpts/{model_path}"
    else:
        ckpts_path = model_path

    # Add best_model.pth if it's just a directory
    if not ckpts_path.endswith(".pth"):
        ckpts_path = os.path.join(ckpts_path, "best_model.pth")

    # Check if the file exists in ckpts/
    if os.path.isfile(ckpts_path):
        print(f"Found model in ckpts: {ckpts_path}")
        return ckpts_path

    # Try to find it with glob pattern in case of slight name differences
    directory = os.path.dirname(ckpts_path)
    if os.path.isdir(directory):
        pth_files = glob.glob(os.path.join(directory, "*.pth"))
        if pth_files:
            best_model = next((f for f in pth_files if "best" in f), pth_files[0])
            print(f"Found alternative model in ckpts: {best_model}")
            return best_model

    # If not found locally, try to download from wandb
    print(f"Model not found locally: {ckpts_path}")
    print("Attempting to download from wandb...")

    try:
        downloaded_path = _download_model_from_wandb(model_path, ckpts_path)
        return downloaded_path
    except Exception as e:
        print(f"Failed to download from wandb: {e}")

        # Final fallback: try to copy from results/ directory
        print("Trying fallback: copying from results/ directory...")
        try:
            results_path = f"results/{model_path}/best_model.pth"
            if not model_path.startswith("results/"):
                results_path = f"results/{model_path}/best_model.pth"

            if os.path.isfile(results_path):
                import shutil

                print(f"Found model in results: {results_path}")
                os.makedirs(os.path.dirname(ckpts_path), exist_ok=True)
                shutil.copy2(results_path, ckpts_path)
                print(f"Copied model to ckpts: {ckpts_path}")
                return ckpts_path
            else:
                print(f"Model not found in results: {results_path}")
        except Exception as copy_error:
            print(f"Failed to copy from results: {copy_error}")

        raise FileNotFoundError(
            f"Model not found locally at '{ckpts_path}' and could not download from wandb.\n"
            f"Available options:\n"
            f"1. Check if the directory exists in ckpts/\n"
            f"2. Provide full path to existing model file\n"
            f"3. Train a new model first\n"
            f"4. Check wandb for available models\n"
            f"5. Copy manually from results/ to ckpts/"
        )


def _download_model_from_wandb(run_name: str, local_path: str) -> str:
    """Download model from wandb if available.

    Args:
        run_name: Run name (e.g., "skill_classifier_20250829_162129")
        local_path: Local path where model should be saved

    Returns:
        Path to downloaded model

    Raises:
        Exception: If download fails
    """
    import glob
    import os

    try:
        import wandb
    except ImportError:
        raise Exception("wandb not installed. Install with: pip install wandb")

    # Use the project name from config
    from toddlerbot.skill_classifier.config import WANDB_CONFIG

    project = WANDB_CONFIG["project"]
    entity = WANDB_CONFIG.get("entity", "toddlerbot")

    try:
        # Initialize wandb run (following mjx_policy pattern)
        if wandb.run:
            run = wandb.run
        else:
            run = wandb.init(
                project=project, entity=entity, name=run_name, job_type="eval"
            )

        # Try multiple artifact naming patterns
        # The training script creates artifacts with double prefix: skill_classifier_{wandb_name}
        # where wandb_name is already skill_classifier_{timestamp}
        artifact_patterns = [
            run_name,  # Direct name
            f"skill_classifier_{run_name}"
            if not run_name.startswith("skill_classifier_")
            else run_name,  # With single prefix
            f"skill_classifier_{run_name}",  # Double prefix (likely the actual name)
            run_name.replace("skill_classifier_", ""),  # Without prefix
        ]

        artifact_found = False
        for artifact_name in artifact_patterns:
            try:
                artifact = run.use_artifact(
                    f"{entity}/{project}/{artifact_name}:latest", type="model"
                )
                print(f"Found artifact: {artifact_name}")

                # Create output directory and download
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                artifact_dir = artifact.download(root=os.path.dirname(local_path))

                # Look for best_model.pth in the artifact
                for file_path in glob.glob(os.path.join(artifact_dir, "*.pth")):
                    if "best" in os.path.basename(file_path):
                        # Move to expected location
                        import shutil

                        shutil.move(file_path, local_path)
                        print(f"Downloaded model to: {local_path}")
                        return local_path

                # If no best_model.pth found, try any .pth file
                pth_files = glob.glob(os.path.join(artifact_dir, "*.pth"))
                if pth_files:
                    import shutil

                    shutil.move(pth_files[0], local_path)
                    print(f"Downloaded model to: {local_path}")
                    return local_path

                artifact_found = True
                break

            except Exception as artifact_error:
                print(f"Could not find artifact {artifact_name}: {artifact_error}")
                continue

        if not artifact_found:
            print(
                "Could not find artifact with any naming pattern, trying direct run search..."
            )

        # Fallback: search for runs with matching name
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        target_run = None
        for r in runs:
            if run_name in r.name or r.name == run_name:
                target_run = r
                break

        if target_run is None:
            raise Exception(f"No wandb run found with name containing '{run_name}'")

        print(f"Found wandb run: {target_run.name}")

        # Try to download best_model.pth directly from files
        for file in target_run.files():
            if file.name == "best_model.pth":
                print(f"Downloading {file.name} to {local_path}")
                file.download(root=os.path.dirname(local_path), replace=True)
                return local_path

        raise Exception(f"No best_model.pth found in wandb run {target_run.name}")

    except Exception as e:
        raise Exception(f"Wandb download failed: {e}")


def get_robot_skill(
    depth_image: np.ndarray, fallback_skill: str = "walk"
) -> Optional[str]:
    """Vision-based replacement for get_robot_skill().

    Drop-in replacement that uses depth perception instead of position.

    Args:
        depth_image: Current depth image
        fallback_skill: Fallback if prediction fails

    Returns:
        Predicted skill name
    """
    if _vision_skill_map is None:
        raise RuntimeError(
            "Vision skill map not initialized. Call initialize_vision_skill_map() first."
        )

    return _vision_skill_map.get_robot_skill(depth_image, fallback_skill)
