"""V7 decision over image-level A, T, C and registered inference costs only."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.stats import t as student_t

from .contracts import CostRegistry
from .frozen.hyper_local_reference_adapter import risk as frozen_risk
from .frozen.v7_core import Validation, propose_safe, reverse_cost_validate


M_SELECTION = 1024
M_VALIDATION = 4096
JACKKNIFE_BLOCKS = 32
SELECTION_LAMBDA = 0.0
VALIDATION_LAMBDA = 0.25
ALPHA_EACH = 0.05


@dataclass(frozen=True)
class PredictionTable:
    image_ids: np.ndarray
    group_ids: np.ndarray
    partition: np.ndarray
    block_id: np.ndarray
    native_a: np.ndarray
    native_t: np.ndarray
    candidates: Mapping[int, np.ndarray]


def select_safe(native_a: np.ndarray, native_t: np.ndarray,
                candidates: Mapping[int, np.ndarray], costs: CostRegistry) -> dict:
    """Frozen Selection-stage proposal before Validation acquisition begins."""
    a = np.asarray(native_a)
    t = np.asarray(native_t)
    if (a.shape != (M_SELECTION,) or t.shape != (M_SELECTION,) or
            not np.isfinite(a).all() or not np.isfinite(t).all()):
        raise ValueError("FROZEN_SELECTION_INPUT_SHAPE_OR_FINITE_MISMATCH")
    contract = costs.frozen_contract()
    rows: list[dict] = []

    def selection_risks(b: int) -> tuple[float, float]:
        if b not in candidates:
            raise ValueError("MISSING_REQUIRED_SELECTION_CANDIDATE_B%d" % b)
        c = np.asarray(candidates[b])
        if c.shape != a.shape or not np.isfinite(c).all():
            raise ValueError("INVALID_SELECTION_CANDIDATE_B%d" % b)
        r = frozen_risk(a, t, c, SELECTION_LAMBDA)
        if not np.isfinite(r).all():
            raise ValueError("NONFINITE_SELECTION_RISK")
        rows.append({"B": b, "Spearman_point": float(r[0]),
                     "Pearson_point": float(r[1]), "both_nonpositive": bool(np.all(r <= 0))})
        return float(r[0]), float(r[1])

    safe = propose_safe(contract, selection_risks)
    return {"B_safe": safe, "selection_tested": rows}


def _inputs(table: PredictionTable) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(table.image_ids).astype(str)
    groups = np.asarray(table.group_ids).astype(str)
    parts = np.asarray(table.partition).astype(str)
    blocks = np.asarray(table.block_id, dtype=int)
    a = np.asarray(table.native_a)
    t = np.asarray(table.native_t)
    n = M_SELECTION + M_VALIDATION
    if any(x.shape != (n,) for x in (ids, groups, parts, blocks, a, t)):
        raise ValueError("V7_SOURCE_VECTOR_SHAPE_MISMATCH")
    if len(set(ids)) != n or len(set(groups)) != n or not np.isfinite(a).all() or not np.isfinite(t).all():
        raise ValueError("DUPLICATE_ID_OR_NONFINITE_NATIVE_PREDICTION")
    si = np.flatnonzero(parts == "SEL")
    vi = np.flatnonzero(parts == "VAL")
    if len(si) != M_SELECTION or len(vi) != M_VALIDATION:
        raise ValueError("FROZEN_SELECTION_VALIDATION_COUNT_MISMATCH")
    if set(groups[si]).intersection(groups[vi]) or not np.all(blocks[si] == -1):
        raise ValueError("GROUP_LEAKAGE_OR_SELECTION_BLOCK_MISMATCH")
    if set(map(int, blocks[vi])) != set(range(JACKKNIFE_BLOCKS)):
        raise ValueError("FROZEN_32_BLOCK_IDENTITY_MISMATCH")
    return a, t, si, vi, blocks[vi]


def calibrate(table: PredictionTable, costs: CostRegistry) -> dict:
    """Run the byte-frozen risk and controller on final scalar predictions.

    A caller may omit higher candidate vectors on validation images when they
    were never acquired. Every tested budget must contain finite predictions.
    """
    a, t, si, vi, blocks = _inputs(table)
    contract = costs.frozen_contract()
    ordered = contract.ordered_candidates()
    candidates = {b: np.asarray(v) for b, v in table.candidates.items()}
    if not set(candidates) <= set(ordered):
        raise ValueError("UNREGISTERED_CANDIDATE_PREDICTION")
    if any(v.shape != a.shape for v in candidates.values()):
        raise ValueError("CANDIDATE_VECTOR_SHAPE_MISMATCH")
    validation_rows: list[dict] = []
    deleted_rows: list[dict] = []
    validation_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def vector(b: int, positions: np.ndarray) -> np.ndarray:
        if b not in candidates:
            raise ValueError("MISSING_REQUIRED_CANDIDATE_B%d" % b)
        values = candidates[b][positions]
        if not np.isfinite(values).all():
            raise ValueError("NONFINITE_REQUIRED_CANDIDATE_B%d" % b)
        return values

    selection = select_safe(a[si], t[si], {b: vector(b, si) for b in candidates}, costs)
    safe = selection["B_safe"]
    selection_rows = selection["selection_tested"]

    def validate(b: int) -> Validation:
        c = vector(b, vi)
        point = frozen_risk(a[vi], t[vi], c, VALIDATION_LAMBDA)
        deleted = np.array([frozen_risk(a[vi][blocks != k], t[vi][blocks != k],
                                        c[blocks != k], VALIDATION_LAMBDA)
                            for k in range(JACKKNIFE_BLOCKS)])
        mean = deleted.mean(axis=0)
        se = np.sqrt((31 / 32) * np.sum((deleted - mean) ** 2, axis=0))
        upper = point + float(student_t.ppf(.95, 31)) * se
        if (not np.isfinite(point).all() or not np.isfinite(se).all() or
                not np.isfinite(upper).all() or not (se > 0).all()):
            raise ValueError("DEGENERATE_OR_NONFINITE_JACKKNIFE")
        validation_cache[b] = point, se, upper
        for k, risk in enumerate(deleted):
            deleted_rows.append({"B": b, "block_id": k,
                                 "Spearman_deleted_risk": float(risk[0]),
                                 "Pearson_deleted_risk": float(risk[1])})
        return Validation(passed=bool((upper <= 0).all()), spearman_ucb=float(upper[0]),
                          pearson_ucb=float(upper[1]), spearman_point_risk=float(point[0]),
                          pearson_point_risk=float(point[1]))

    decision = reverse_cost_validate(contract, safe, validate)
    for row in decision["tested"]:
        b = row["budget"]
        point, se, upper = validation_cache[b]
        validation_rows.append({"B": b, "Spearman_point": float(point[0]),
                                "Pearson_point": float(point[1]), "Spearman_SE": float(se[0]),
                                "Pearson_SE": float(se[1]), "Spearman_UCB": float(upper[0]),
                                "Pearson_UCB": float(upper[1]), "both_pass": row["both_pass"],
                                "first_failure": b == decision["first_failed_budget"],
                                "critical_t": float(student_t.ppf(.95, 31)),
                                "one_sided_alpha_each": ALPHA_EACH})
    return {"B_safe": safe, "B_V7": decision["final_budget"], "decision": decision,
            "selection_tested": selection_rows, "validation_ACTUALLY_TESTED": validation_rows,
            "delete_block_ACTUALLY_TESTED": deleted_rows}
