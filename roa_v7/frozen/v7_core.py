"""Model-agnostic reverse-cost adjacent validation for frozen scalar predictors.

This controls the order of *existing* validation tests. It does not define or
alter the model's risk, confidence bound, data split, or candidate sampler.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Hashable, Mapping


Budget = Hashable


@dataclass(frozen=True)
class BudgetContract:
    native_label: Budget
    native_cost: float
    candidate_costs: Mapping[Budget, float]
    candidate_resources: Mapping[Budget, frozenset[Hashable]]
    resource_costs: Mapping[Hashable, float]
    nested_candidates: bool

    def ordered_candidates(self) -> tuple[Budget, ...]:
        if not math.isfinite(self.native_cost) or self.native_cost <= 0:
            raise ValueError("INVALID_NATIVE_COST")
        if (not self.candidate_costs or self.native_label in self.candidate_costs or
                set(self.candidate_costs) != set(self.candidate_resources)):
            raise ValueError("CANDIDATE_COST_OR_RESOURCE_REGISTRATION_MISSING")
        values = [float(v) for v in self.candidate_costs.values()]
        if (any(not math.isfinite(v) or v <= 0 or v >= self.native_cost for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("INVALID_OR_TIED_CANDIDATE_COST")
        universe = set().union(*self.candidate_resources.values())
        if not universe or not universe <= set(self.resource_costs):
            raise ValueError("RESOURCE_COST_REGISTRATION_MISSING")
        if any(not math.isfinite(float(self.resource_costs[r])) or self.resource_costs[r] <= 0 for r in universe):
            raise ValueError("INVALID_RESOURCE_COST")
        ordered = tuple(sorted(self.candidate_costs, key=lambda b: float(self.candidate_costs[b])))
        if self.nested_candidates:
            previous: frozenset[Hashable] = frozenset()
            for budget in ordered:
                current = self.candidate_resources[budget]
                if not previous <= current:
                    raise ValueError("FALSE_NESTED_PREFIX_DECLARATION")
                previous = current
        return ordered


@dataclass(frozen=True)
class Validation:
    passed: bool
    spearman_ucb: float
    pearson_ucb: float
    spearman_point_risk: float
    pearson_point_risk: float


def propose_safe(contract: BudgetContract,
                 selection_risks: Callable[[Budget], tuple[float, float]]) -> Budget:
    """Use predeclared actual cost order; both frozen selection gaps must pass."""
    for budget in contract.ordered_candidates():
        spearman, pearson = selection_risks(budget)
        if not all(math.isfinite(v) for v in (spearman, pearson)):
            raise ValueError("NONFINITE_SELECTION_RISK")
        if spearman <= 0 and pearson <= 0:
            return budget
    return contract.native_label


def reverse_cost_validate(contract: BudgetContract, safe_budget: Budget,
                          validate: Callable[[Budget], Validation]) -> dict:
    """Test safe, then adjacent cheaper budgets; first failure stops forever."""
    ordered = contract.ordered_candidates()
    if safe_budget == contract.native_label:
        return {"safe_budget": safe_budget, "final_budget": contract.native_label,
                "first_failed_budget": None, "tested": [], "validation_test_count": 0,
                "additional_acquisition_resources": 0, "additional_acquisition_cost": 0.0,
                "nested_candidates": contract.nested_candidates,
                "stop_reason": "NO_SAFE_CANDIDATE_PROPOSED"}
    if safe_budget not in ordered:
        raise ValueError("SAFE_BUDGET_NOT_REGISTERED")
    index = ordered.index(safe_budget)
    descending = (safe_budget,) + tuple(reversed(ordered[:index]))
    selected = contract.native_label
    first_failed = None
    acquired: set[Hashable] = set()
    total_new_resources = 0
    total_acquisition_cost = 0.0
    trace = []
    for step, budget in enumerate(descending):
        resources = set(contract.candidate_resources[budget])
        extra = resources - acquired
        added_cost = sum(float(contract.resource_costs[item]) for item in extra)
        acquired.update(resources)
        total_new_resources += len(extra)
        total_acquisition_cost += added_cost
        result = validate(budget)
        if not isinstance(result, Validation) or any(not math.isfinite(v) for v in
            (result.spearman_ucb, result.pearson_ucb,
             result.spearman_point_risk, result.pearson_point_risk)):
            raise ValueError("INVALID_VALIDATION_RESULT")
        if result.passed != (result.spearman_ucb <= 0 and result.pearson_ucb <= 0):
            raise ValueError("VALIDATION_PASS_FLAG_INCONSISTENT_WITH_BOTH_UCBS")
        trace.append({"step": step, "budget": budget, "registered_cost": float(contract.candidate_costs[budget]),
                      "spearman_UCB": result.spearman_ucb, "pearson_UCB": result.pearson_ucb,
                      "spearman_point_risk": result.spearman_point_risk,
                      "pearson_point_risk": result.pearson_point_risk,
                      "both_pass": result.passed,
                      "new_resource_count": len(extra), "new_acquisition_cost": added_cost})
        if not result.passed:
            first_failed = budget
            break
        selected = budget
    return {"safe_budget": safe_budget, "final_budget": selected,
            "first_failed_budget": first_failed, "tested": trace,
            "validation_test_count": len(trace),
            "additional_acquisition_resources": total_new_resources,
            "additional_acquisition_cost": total_acquisition_cost,
            "nested_candidates": contract.nested_candidates,
            "stop_reason": "FIRST_FAILURE" if first_failed is not None else "LOWEST_REGISTERED_COST_PASSED"}
