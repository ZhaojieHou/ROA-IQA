
"""YOTO Native20 and byte-frozen V7 fixed-Sobol coordinate streams."""

from __future__ import annotations

import hashlib

import numpy as np


NATIVE_CROPS = 20
CANDIDATE_MAX = 19
CROP = 224
SOBOL_SEED = 20260816


def _native_seed(dataset: str, session: str, phase: str, image_id: str, repeat_id: int) -> list[int]:
    if repeat_id not in (0, 1) or not all((dataset, session, phase, image_id)):
        raise ValueError("Incomplete Native20 independent stream identity")
    payload = f"YOTO_V7_NATIVE20_V1|{dataset}|{session}|{phase}|{image_id}|{repeat_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # RandomState accepts an array of uint32 MT19937 seed words. Retaining
    # 128 bits avoids a 32-bit birthday collision across tens of thousands
    # of independently identified A/T/image/session streams.
    return [int.from_bytes(digest[index:index + 4], "little") for index in (0, 4, 8, 12)]


def native_coordinates(height: int, width: int, dataset: str, session: str,
                       phase: str, image_id: str, repeat_id: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Return (top,left) with the author's exclusive-upper-bound randint law."""
    if height <= CROP or width <= CROP:
        raise ValueError("Author Native20 cannot sample an image <= 224 in either dimension")
    seed = _native_seed(dataset, session, phase, image_id, repeat_id)
    rng = np.random.RandomState(seed)
    positions = [(int(rng.randint(0, height - CROP)), int(rng.randint(0, width - CROP)))
                 for _ in range(NATIVE_CROPS)]
    return positions, seed


def sobol_coordinates(height: int, width: int, count: int = CANDIDATE_MAX) -> list[tuple[int, int]]:
    """Use the original frozen acquisition implementation; convert (x,y) to (top,left)."""
    from roa_v7.frozen.roa_core import acquisition

    if height <= CROP or width <= CROP or not 1 <= count <= CANDIDATE_MAX:
        raise ValueError("YOTO source image or candidate prefix outside frozen domain")
    xy = acquisition.sobol_coordinates(width, height, CROP, CROP, count)
    positions = [(int(y), int(x)) for x, y in xy]
    if len(positions) != count or any(
        not (0 <= top <= height - CROP and 0 <= left <= width - CROP)
        for top, left in positions
    ):
        raise ValueError("Frozen Sobol positions outside physical crop support")
    return positions


def sobol_unit_points(count: int = CANDIDATE_MAX) -> list[list[float]]:
    """Audit-only raw unit points from the same frozen SciPy call, never a second sampler."""
    import math
    from scipy.stats import qmc

    if not 1 <= count <= CANDIDATE_MAX:
        raise ValueError("Invalid audit Sobol prefix")
    points = qmc.Sobol(2, scramble=True, seed=SOBOL_SEED).random_base2(
        math.ceil(math.log2(max(count, 2)))
    )[:count]
    return points.tolist()
