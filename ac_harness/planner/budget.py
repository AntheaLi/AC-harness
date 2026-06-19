# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Budget gates for experiment planning (§8).

A coarse, human-meaningful cost model. Real GPU-hour estimates can be
wired in later, but for v1 we map experiment_type + size to a
discrete tier so the planner can filter by `--budget {small,medium,large}`.
"""
from __future__ import annotations

from typing import Literal

Budget = Literal["small", "medium", "large"]

_TIER = {"small": 1, "medium": 2, "large": 3}

# (experiment_type, scope) → cost tier (1..3)
_COST_TABLE: dict[str, int] = {
    "kernel_microbench": 1,
    "serving_bench": 1,
    "import_external_result": 1,
    "quality_eval": 2,
    "small_training_ablation": 2,
}


def cost_tier(experiment_type: str) -> int:
    return _COST_TABLE.get(experiment_type, 3)


def cost_label(experiment_type: str) -> str:
    return {1: "low", 2: "medium", 3: "high"}[cost_tier(experiment_type)]


def fits_budget(experiment_type: str, budget: Budget) -> bool:
    return cost_tier(experiment_type) <= _TIER[budget]


def cost_dict(experiment_type: str) -> dict[str, object]:
    """A small structured cost record attached to ExperimentPlan."""
    return {
        "tier": cost_tier(experiment_type),
        "label": cost_label(experiment_type),
    }
