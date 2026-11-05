"""Versioned downstream and cost contracts for final image-level predictions."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Hashable, Mapping, Protocol, Sequence

from .frozen.v7_core import BudgetContract
from . import SCHEMA as CONTRACT_VERSION


@dataclass(frozen=True)
class CostRecord:
    budget: int
    decision_cost: float
    decision_unit: str
    forward_flops: float | None
    latency_ms: float | None
    model_calls: int
    verified_prefix_reuse: bool
    measurement_identity: str

    def __post_init__(self) -> None:
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("INVALID_BUDGET")
        if not math.isfinite(self.decision_cost) or self.decision_cost <= 0:
            raise ValueError("INVALID_DECISION_COST")
        if not self.decision_unit or not self.measurement_identity:
            raise ValueError("COST_UNIT_OR_MEASUREMENT_IDENTITY_MISSING")
        if self.forward_flops is not None and (not math.isfinite(self.forward_flops) or self.forward_flops <= 0):
            raise ValueError("INVALID_FORWARD_FLOPS")
        if self.latency_ms is not None and (not math.isfinite(self.latency_ms) or self.latency_ms <= 0):
            raise ValueError("INVALID_LATENCY")
        if type(self.model_calls) is not int or self.model_calls < 1:
            raise ValueError("INVALID_MODEL_CALL_COUNT")


@dataclass(frozen=True)
class CostRegistry:
    native: CostRecord
    candidates: Mapping[int, CostRecord]
    candidate_resources: Mapping[int, frozenset[Hashable]]
    resource_costs: Mapping[Hashable, float]
    nested_candidates: bool

    def frozen_contract(self) -> BudgetContract:
        if not self.candidates or set(self.candidates) != set(self.candidate_resources):
            raise ValueError("CANDIDATE_REGISTRY_INCOMPLETE")
        if any(b != rec.budget or b >= self.native.budget or
               rec.decision_unit != self.native.decision_unit for b, rec in self.candidates.items()):
            raise ValueError("COST_BUDGET_OR_UNIT_MISMATCH")
        if self.nested_candidates and not all(r.verified_prefix_reuse for r in self.candidates.values()):
            raise ValueError("NESTED_COST_REUSE_NOT_VERIFIED")
        return BudgetContract(
            native_label=self.native.budget,
            native_cost=self.native.decision_cost,
            candidate_costs={b: r.decision_cost for b, r in self.candidates.items()},
            candidate_resources=self.candidate_resources,
            resource_costs=self.resource_costs,
            nested_candidates=self.nested_candidates,
        )


class DownstreamAdapter(Protocol):
    """Every prediction returned here is the frozen model's final image score."""

    def get_contract_version(self) -> str: ...
    def get_native_budget(self) -> int: ...
    def get_supported_budgets(self) -> list[int]: ...
    def predict_native(self, image: Any, repeat_id: int, rng_context: Any) -> float: ...
    def predict_candidate(self, image: Any, observations: Sequence[Any]) -> float: ...
    def get_inference_cost(self, budget: int) -> CostRecord: ...


def validate_adapter(adapter: DownstreamAdapter) -> tuple[int, ...]:
    if adapter.get_contract_version() != CONTRACT_VERSION:
        raise ValueError("UNSUPPORTED_DOWNSTREAM_ADAPTER_CONTRACT_VERSION")
    native = adapter.get_native_budget()
    budgets = adapter.get_supported_budgets()
    if type(native) is not int or native < 2 or not isinstance(budgets, list):
        raise ValueError("INVALID_NATIVE_OR_SUPPORTED_BUDGETS")
    if len(budgets) != len(set(budgets)) or any(type(b) is not int or not 1 <= b < native for b in budgets):
        raise ValueError("INVALID_SUPPORTED_BUDGET_DOMAIN")
    if not budgets:
        raise ValueError("NO_CHEAPER_LEGAL_BUDGET")
    for b in budgets + [native]:
        if adapter.get_inference_cost(b).budget != b:
            raise ValueError("ADAPTER_COST_BUDGET_MISMATCH")
    return tuple(sorted(budgets))
