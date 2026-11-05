
"""Image preprocessing for the original YOTO-NR model."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def preprocess_image(
    dataset: str,
    distorted_path: Path,
    reference_path: Path | None = None,
) -> np.ndarray:
    import cv2

    name = dataset.upper()
    if name not in {"KONIQ", "LIVE", "CSIQ"}:
        raise ValueError("Expected KONIQ, LIVE or CSIQ")

    distorted = cv2.imread(str(distorted_path), cv2.IMREAD_COLOR)
    if distorted is None:
        raise ValueError(f"Cannot decode image: {distorted_path}")

    if name in {"LIVE", "CSIQ"}:
        if reference_path is None:
            raise ValueError(f"{name} reader requires a reference image")

        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise ValueError(f"Cannot decode reference: {reference_path}")

        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        reference = np.transpose(
            np.asarray(reference).astype("float32") / 255,
            (2, 0, 1),
        )
        reference = np.ascontiguousarray(
            (reference - 0.5) / 0.5,
            dtype=np.float32,
        )
        if reference.ndim != 3 or reference.shape[0] != 3:
            raise ValueError("Reference image shape mismatch")

    distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)

    if name == "KONIQ":
        distorted = cv2.resize(
            distorted,
            (512, 384),
            interpolation=cv2.INTER_CUBIC,
        )

    image = np.asarray(distorted).astype("float32") / 255
    image = np.transpose(image, (2, 0, 1))

    normalized = np.ascontiguousarray(
        (image - 0.5) / 0.5,
        dtype=np.float32,
    )

    if normalized.ndim != 3 or normalized.shape[0] != 3:
        raise ValueError("YOTO image shape mismatch")

    if normalized.shape[1] <= 224 or normalized.shape[2] <= 224:
        raise ValueError("Image must be larger than 224x224")

    return normalized
