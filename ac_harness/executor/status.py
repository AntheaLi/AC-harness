# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""Status transitions on ExperimentPlan."""
from __future__ import annotations

from ..schemas import ExperimentPlan, ExperimentStatus
from ..store import EvidenceStore

_ALLOWED: dict[ExperimentStatus, set[ExperimentStatus]] = {
    "planned": {"approved", "running", "skipped"},
    "approved": {"running", "skipped"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
    "skipped": set(),
}


def set_status(
    store: EvidenceStore, plan_id: str, new_status: ExperimentStatus
) -> ExperimentPlan:
    plan = store.get_experiment_plan(plan_id)
    if plan is None:
        raise KeyError(f"no ExperimentPlan with id {plan_id}")
    if new_status not in _ALLOWED.get(plan.status, set()) and new_status != plan.status:
        raise ValueError(
            f"illegal status transition {plan.status} -> {new_status} on {plan_id}"
        )
    plan.status = new_status
    store.insert_experiment_plan(plan)
    return plan
