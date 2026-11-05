"""Predeclared structural YOTO-NR crop cost order for frozen V7 decisions."""

from __future__ import annotations

from roa_v7.contracts import CostRecord, CostRegistry


def registry(crop_batch_size: int) -> CostRegistry:
    if crop_batch_size not in (1, 8):
        raise ValueError("Formal crop batches require proven batch1 or batch8 parity")
    unit = "YOTO_NR_REAL_CROP_INPUTS_STRUCTURAL"
    identity = f"V7_YOTO_N20_B1_TO19_MEASURED_FORWARDS_SEPARATE_BATCH{crop_batch_size}"

    def record(budget: int) -> CostRecord:
        return CostRecord(
            budget=budget, decision_cost=float(budget), decision_unit=unit,
            forward_flops=None, latency_ms=None,
            model_calls=(budget + crop_batch_size - 1) // crop_batch_size,
            verified_prefix_reuse=True, measurement_identity=identity,
        )

    result = CostRegistry(
        native=record(20),
        candidates={budget: record(budget) for budget in range(1, 20)},
        candidate_resources={budget: frozenset(range(1, budget + 1)) for budget in range(1, 20)},
        resource_costs={budget: 1.0 for budget in range(1, 20)},
        nested_candidates=True,
    )
    if result.frozen_contract().ordered_candidates() != tuple(range(1, 20)):
        raise ValueError("Frozen V7 YOTO candidate order differs")
    return result
