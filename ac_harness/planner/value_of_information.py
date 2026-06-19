# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Value-of-information heuristic (§8).

Heuristic, not Bayesian. The planner ranks candidate experiments by:

    value = P(result_changes_decision) * importance(decision) / cost

For v1 we encode this as a handful of named rules. The shape of
`score_experiment_candidate` and the returned dict is what the planner
uses; the rules themselves can evolve as we collect data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .budget import cost_tier


@dataclass
class ScoredCandidate:
    """An internal candidate experiment with a heuristic score."""

    candidate_ids: list[str]
    experiment_type: Literal[
        "kernel_microbench",
        "serving_bench",
        "small_training_ablation",
        "quality_eval",
        "import_external_result",
    ]
    decision_unblocked: str
    uncertainty_target: str
    score: float
    rationale: str
    config_hint: dict[str, object]


def score_experiment_candidate(
    *,
    experiment_type: str,
    p_changes_decision: float,
    importance: float,
) -> float:
    """Compute the raw VoI score. `importance` is roughly the magnitude of
    the decision being unblocked (1.0 = routine, 2.0 = a Pareto-flipping
    decision). `p_changes_decision` is in [0, 1].
    """
    p_changes_decision = max(0.0, min(1.0, p_changes_decision))
    importance = max(0.0, importance)
    tier = cost_tier(experiment_type)
    return (p_changes_decision * importance) / tier
