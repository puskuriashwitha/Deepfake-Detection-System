
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

# Project root
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.model import DeepfakeCNN


class DeepfakeDetector:
    """
    CNN-based experimental deepfake classifier.

    Input: Complete video frame in OpenCV BGR format.
    Output: Real/fake probabilities.
    """

    def __init__(self, model_path=None):
        self.model_path = (
            Path(model_path)
            if model_path
            else ROOT / "models" / "deepfake_model.pth"
        )

        self.device = torch.device("cpu")
        self.image_size = 128

        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        self.model = None
        self.model_ready = False
        self.model_error = None

        self.load_model()

    def load_model(self):
        """Load the trained CNN checkpoint."""

        if not self.model_path.exists():
            self.model_error = (
                f"Model checkpoint not found: {self.model_path}"
            )
            print(self.model_error)
            return

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
            )

            if checkpoint.get("architecture") != "DeepfakeCNN":
                raise ValueError("Unexpected model architecture.")

            self.image_size = int(
                checkpoint.get("image_size", 128)
            )

            self.model = DeepfakeCNN(num_classes=2)

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.model.to(self.device)
            self.model.eval()

            self.class_to_idx = checkpoint.get(
                "class_to_idx",
                {"real": 0, "fake": 1}
            )

            self.real_idx = self.class_to_idx["real"]
            self.fake_idx = self.class_to_idx["fake"]

            normalization = checkpoint.get("normalization", {})

            self.mean = np.array(
                normalization.get(
                    "mean", [0.485, 0.456, 0.406]
                ),
                dtype=np.float32
            )

            self.std = np.array(
                normalization.get(
                    "std", [0.229, 0.224, 0.225]
                ),
                dtype=np.float32
            )

            self.model_ready = True

            print("Deepfake CNN model loaded successfully.")
            print(f"Model path: {self.model_path}")
            print(f"Device: {self.device}")

        except Exception as exc:
            self.model = None
            self.model_ready = False
            self.model_error = str(exc)

            print("Could not load deepfake model:", exc)

    def preprocess(self, frame):
        """
        Convert OpenCV BGR frame into the exact
        tensor format used during training.
        """

        if frame is None or frame.size == 0:
            raise ValueError("Empty video frame received.")

        # OpenCV BGR -> RGB
        rgb = frame[:, :, ::-1].copy()

        image = Image.fromarray(rgb).convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size)
        )

        array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        array = (array - self.mean) / self.std

        array = array.transpose(2, 0, 1).copy()

        tensor = torch.from_numpy(array).unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, frame):
        """Predict whether a complete frame looks real or fake."""

        if not self.model_ready:
            return {
                "status": "model_unavailable",
                "label": "NOT AVAILABLE",
                "confidence": None,
                "real_probability": None,
                "fake_probability": None,
                "message": self.model_error or "Model is unavailable."
            }

        try:
            tensor = self.preprocess(frame)

            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            real_probability = float(
                probabilities[self.real_idx].item()
            )

            fake_probability = float(
                probabilities[self.fake_idx].item()
            )

            if fake_probability >= 0.5:
                label = "FAKE"
                confidence = fake_probability
            else:
                label = "REAL"
                confidence = real_probability

            return {
                "status": "prediction_available_experimental",
                "label": label,
                "confidence": round(confidence, 4),
                "real_probability": round(real_probability, 4),
                "fake_probability": round(fake_probability, 4),
                "message": (
                    "Experimental CNN prediction. "
                    "This result is not a reliable authenticity verdict."
                )
            }

        except Exception as exc:
            return {
                "status": "prediction_error",
                "label": "NOT AVAILABLE",
                "confidence": None,
                "real_probability": None,
                "fake_probability": None,
                "message": str(exc)
            }
