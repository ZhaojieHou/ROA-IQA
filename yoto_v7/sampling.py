
"""Native20 random crops and fixed-seed scrambled Sobol crops for YOTO-NR."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import qmc


NATIVE_CROPS = 20
CROP = 224
SOBOL_SEED = 20260816
MAX_BUDGET = 19


def _native_seed(dataset: str, session: str, phase: str,
                 image_id: str, repeat_id: int) -> list[int]:
    """Reproduce the original deterministic Native20 random stream."""
    if repeat_id not in (0, 1) or not all(
        (dataset, session, phase, image_id)
    ):
        raise ValueError("Invalid Native20 sampling parameters")

    payload = (
        f"YOTO_V7_NATIVE20_V1|{dataset}|{session}|{phase}|"
        f"{image_id}|{repeat_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()
    return [
        int.from_bytes(digest[i:i + 4], "little")
        for i in (0, 4, 8, 12)
    ]


def native_coordinates(height: int, width: int, dataset: str,
                       session: str, phase: str, image_id: str,
                       repeat_id: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Return Native20 crop coordinates as (top, left)."""
    if height <= CROP or width <= CROP:
        raise ValueError("Image must be larger than 224x224")

    seed = _native_seed(
        dataset, session, phase, image_id, repeat_id
    )
    rng = np.random.RandomState(seed)

    positions = [
        (
            int(rng.randint(0, height - CROP)),
            int(rng.randint(0, width - CROP))
        )
        for _ in range(NATIVE_CROPS)
    ]
    return positions, seed


def sobol_coordinates(height: int, width: int,
                      count: int = MAX_BUDGET) -> list[tuple[int, int]]:
    """Return fixed-seed scrambled Sobol coordinates as (top, left)."""
    if height <= CROP or width <= CROP:
        raise ValueError("Image must be larger than 224x224")
    if not 1 <= count <= MAX_BUDGET:
        raise ValueError("Invalid ROA budget")

    points = qmc.Sobol(
        d=2,
        scramble=True,
        seed=SOBOL_SEED
    ).random_base2(
        math.ceil(math.log2(max(count, 2)))
    )[:count]

    lengths = np.array([
        width - CROP + 1,
        height - CROP + 1
    ])

    xy = np.minimum(
        (points * lengths).astype(np.int64),
        lengths - 1
    )

    return [
        (int(y), int(x))
        for x, y in xy
    ]
