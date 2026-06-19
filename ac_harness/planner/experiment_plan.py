# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Experiment planner (§8).

Reads current store state (via DecisionState skeleton), heuristically ranks
candidate experiments by expected decision value, and emits ranked
`ExperimentPlan` records.

Acceptance behavior the planner must satisfy (§8 acceptance test):
  - missing throughput data        → prioritize kernel/serving bench
  - missing quality on risky delta → prioritize small_training_ablation / quality_eval
  - all key buckets covered        → recommend a fit/update/report (modeled as
                                     `import_external_result` with config
                                     hint `mode=fit_and_report`)

The planner intentionally does not predict architecture quality or
throughput; it merely identifies which measurement would shift the
decision and at what cost.
"""
from __future__ import annotations

import uuid
from typing import Iterable

from ..decision import build_decision_state
from ..schemas import ExperimentPlan
from ..store import EvidenceStore
from ..store.provenance import make_provenance
from .budget import Budget, cost_dict, fits_budget
from .value_of_information import ScoredCandidate, score_experiment_candidate


def _new_plan_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def plan_next_experiments(
    store: EvidenceStore,
    *,
    budget: Budget = "small",
    max_plans: int = 5,
    persist: bool = True,
) -> list[ExperimentPlan]:
    """Build ranked ExperimentPlan records reflecting what to measure next."""
    ds = build_decision_state(store)
    cands = ds.candidate_ids
    missing = ds.uncertainty_summary["missing_buckets_by_candidate"]
    risky = set(ds.uncertainty_summary["risky_unmeasured_candidates"])
    pred_only = set(ds.uncertainty_summary["prediction_only_candidates"])

    scored: list[ScoredCandidate] = []

    # Rule 1: any candidate missing the throughput OR kernel_bandwidth
    # bucket → kernel_microbench or serving_bench.
    for cid in cands:
        miss = set(missing.get(cid, []))
        if "kernel_bandwidth" in miss or "throughput" in miss:
            # Pair with baseline if available for a meaningful delta.
            pair = [cid]
            for other in cands:
                if other != cid and other not in pair:
                    pair.append(other)
                    break
            imp = 2.0 if cid in pred_only else 1.5
            p = 0.7 if cid in pred_only else 0.55
            scored.append(
                ScoredCandidate(
                    candidate_ids=pair,
                    experiment_type="kernel_microbench",
                    decision_unblocked=(
                        f"whether {cid} delivers a serving-useful throughput delta"
                    ),
                    uncertainty_target="decode KV bandwidth correction",
                    score=score_experiment_candidate(
                        experiment_type="kernel_microbench",
                        p_changes_decision=p,
                        importance=imp,
                    ),
                    rationale=(
                        f"{cid} has no observed kernel/throughput data yet; a "
                        "decode-KV microbench is the cheapest way to learn whether the "
                        "predicted gain holds on this runtime."
                    ),
                    config_hint={"workload": "decode_kv", "shapes": "default_grid"},
                )
            )
            # A complementary serving bench at higher tier.
            scored.append(
                ScoredCandidate(
                    candidate_ids=pair,
                    experiment_type="serving_bench",
                    decision_unblocked=(
                        f"whether {cid} sustains throughput in a realistic serving mix"
                    ),
                    uncertainty_target="end-to-end serving throughput",
                    score=score_experiment_candidate(
                        experiment_type="serving_bench",
                        p_changes_decision=0.45,
                        importance=imp,
                    ),
                    rationale=(
                        f"Validates kernel result against a realistic mix for {cid}."
                    ),
                    config_hint={"workload": "long_chat"},
                )
            )

    # Rule 2: risky/research delta without quality measurements → quality eval
    # or small training ablation.
    for cid in cands:
        miss = set(missing.get(cid, []))
        if cid in risky and ("quality_loss" in miss or "quality_task" in miss):
            pair = [cid]
            for other in cands:
                if other != cid and other not in pair:
                    pair.append(other)
                    break
            # Risky-and-unmeasured-quality is the spec's "prioritize small
            # ablation" case. Importance is scored high (4.0) because a quality
            # regression here invalidates the whole delta, not just calibrates it.
            scored.append(
                ScoredCandidate(
                    candidate_ids=pair,
                    experiment_type="small_training_ablation",
                    decision_unblocked=(
                        f"whether {cid}'s delta is quality-safe at small scale"
                    ),
                    uncertainty_target="quality residual for the changed fields",
                    score=score_experiment_candidate(
                        experiment_type="small_training_ablation",
                        p_changes_decision=0.8,
                        importance=4.0,
                    ),
                    rationale=(
                        f"{cid} is tagged risky/research by AC-Core but has no "
                        "observed quality signal. A short continued-pretraining "
                        "ablation is the lowest-cost way to discover regressions."
                    ),
                    config_hint={"tokens": "small", "seeds": 3},
                )
            )
            scored.append(
                ScoredCandidate(
                    candidate_ids=pair,
                    experiment_type="quality_eval",
                    decision_unblocked=(
                        f"whether existing checkpoints for {cid} pass quality eval"
                    ),
                    uncertainty_target="downstream task scores",
                    score=score_experiment_candidate(
                        experiment_type="quality_eval",
                        p_changes_decision=0.6,
                        importance=3.0,
                    ),
                    rationale=(
                        "Cheaper than retraining; surfaces obvious failures."
                    ),
                    config_hint={"suites": ["needle", "mqar"]},
                )
            )

    # Rule 3: every candidate has all key buckets → propose fit/report.
    if cands and all(not missing.get(cid) for cid in cands):
        scored.append(
            ScoredCandidate(
                candidate_ids=list(cands),
                experiment_type="import_external_result",
                decision_unblocked=(
                    "whether current evidence supports updating fitted laws and "
                    "publishing a decision report"
                ),
                uncertainty_target="fit stability / calibration freshness",
                score=score_experiment_candidate(
                    experiment_type="import_external_result",
                    p_changes_decision=0.6,
                    importance=1.5,
                ),
                rationale=(
                    "All key measurement buckets are populated. Next step is to "
                    "refit calibration/residuals and regenerate the decision "
                    "report rather than collect more raw data."
                ),
                config_hint={"mode": "fit_and_report"},
            )
        )

    # Sort by score desc; break ties by lower cost (lower tier).
    scored.sort(
        key=lambda s: (-s.score, _tier_int(s.experiment_type), s.candidate_ids)
    )

    # Apply budget filter.
    scored = [s for s in scored if fits_budget(s.experiment_type, budget)]

    # Materialize ExperimentPlan records.
    plans: list[ExperimentPlan] = []
    for rank, s in enumerate(scored[:max_plans], start=1):
        plan = ExperimentPlan(
            id=_new_plan_id(s.experiment_type),
            name=f"{s.experiment_type}: {' vs '.join(s.candidate_ids)}",
            experiment_type=s.experiment_type,
            candidate_ids=s.candidate_ids,
            decision_unblocked=s.decision_unblocked,
            uncertainty_target=s.uncertainty_target,
            estimated_cost=cost_dict(s.experiment_type),
            required_resources={"gpu_count": 1 if s.experiment_type != "small_training_ablation" else 4},
            config=dict(s.config_hint),
            rationale=s.rationale,
            provenance=make_provenance(
                command="planner.plan_next_experiments",
                extra={"rank": rank, "score": s.score, "budget": budget},
            ),
        )
        plans.append(plan)
        if persist:
            store.insert_experiment_plan(plan)
    return plans


def _tier_int(experiment_type: str) -> int:
    from .budget import cost_tier as _ct

    return _ct(experiment_type)


def render_plan_markdown(plans: Iterable[ExperimentPlan]) -> str:
    """Render a ranked list of plans as Markdown matching the §8.3 shape."""
    lines = ["# Next experiments\n"]
    for i, p in enumerate(plans, start=1):
        cost_label = p.estimated_cost.get("label", "?")
        lines.append(
            f"{i}. **{p.name}**  \n"
            f"   *Decision unblocked:* {p.decision_unblocked}.  \n"
            f"   *Uncertainty target:* {p.uncertainty_target}.  \n"
            f"   *Cost:* {cost_label}.  \n"
            f"   *Why now:* {p.rationale or '—'}\n"
        )
    if not lines:
        lines.append("_No experiment recommendations at this budget._\n")
    return "\n".join(lines)
