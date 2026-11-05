"""Run YOTO-NR crop inference and aggregate image quality scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frozen_model import make_model
from .preprocess import preprocess_image



@dataclass(frozen=True)
class ImageSpec:
    image_id: str
    distorted_relpath: str
    reference_relpath: str = ""


class YotoScorer:
    def __init__(self, *, dataset: str, asset_root: Path, data_root: Path,
                 checkpoint: Path, device: str,
                 crop_batch_size: int = 1):
        import torch

        self.dataset = dataset.upper()
        if self.dataset not in {"KONIQ", "LIVE", "CSIQ"}:
            raise ValueError("Unsupported YOTO-NR dataset")
        if crop_batch_size < 1 or crop_batch_size > 8:
            raise ValueError("Crop batch size must be between 1 and 8")
        self.asset_root = Path(asset_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not self.data_root.is_dir() or not self.checkpoint.is_file():
            raise FileNotFoundError("Image directory or model checkpoint not found")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = make_model(self.device, asset_root=self.asset_root)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(
            isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in state.values()
        ):
            raise ValueError("Expected a checkpoint containing a tensor state_dict")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        if not all(parameter.dtype == torch.float32 for parameter in self.model.parameters()):
            raise ValueError("YOTO-NR inference requires FP32 model weights")
        self.crop_batch_size = crop_batch_size

    def _inside_data_root(self, relative: str) -> Path:
        path = (self.data_root / relative).resolve()
        if not relative or not path.is_relative_to(self.data_root) or not path.is_file():
            raise FileNotFoundError(f"Image path not found or outside the image directory: {relative}")
        return path

    def load_image(self, spec: ImageSpec) -> np.ndarray:
        distorted = self._inside_data_root(spec.distorted_relpath)
        reference = self._inside_data_root(spec.reference_relpath) if self.dataset != "KONIQ" else None
        return preprocess_image(self.dataset, distorted, reference)

    def score_coordinates(self, image: np.ndarray, coordinates: list[tuple[int, int]]) -> dict:
        import torch

        if image.dtype != np.float32 or image.ndim != 3 or image.shape[0] != 3 or not coordinates:
            raise ValueError("Normalized real image and nonempty crop coordinates required")
        height, width = image.shape[1:]
        scores: list[float] = []
        batches: list[list[int]] = []
        with torch.inference_mode():
            for start in range(0, len(coordinates), self.crop_batch_size):
                positions = coordinates[start:start + self.crop_batch_size]
                patches = []
                for top, left in positions:
                    if not (0 <= top <= height - 224 and 0 <= left <= width - 224):
                        raise ValueError("Crop outside physically valid YOTO image domain")
                    patch = np.ascontiguousarray(image[:, top:top + 224, left:left + 224])
                    if patch.shape != (3, 224, 224):
                        raise ValueError("YOTO crop shape mismatch")
                    patches.append(patch)
                tensor = torch.from_numpy(np.stack(patches, axis=0)).to(self.device, dtype=torch.float32)
                values = self.model(tensor, tensor, mode="NR").reshape(-1)
                if values.numel() != len(positions) or not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("YOTO-NR network returned invalid crop scores")
                scores.extend(float(value) for value in values.detach().cpu().tolist())
                batches.append(list(tensor.shape))
        if len(scores) != len(coordinates) or not all(math.isfinite(value) for value in scores):
            raise ValueError("Incomplete YOTO crop prediction")
        return {"scores": scores, "model_calls": len(batches),
                "crop_forwards": len(scores), "batch_input_shapes": batches}


def image_mean(scores: list[float]) -> float:
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Final image score requires finite crop predictions")
    # Frozen author inference starts with a scalar tensor and adds each
    # float32 crop prediction in order before dividing by the crop count.
    total = np.float32(0.0)
    for score in scores:
        total = np.float32(total + np.float32(score))
    return float(np.float32(total / np.float32(len(scores))))
