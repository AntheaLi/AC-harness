# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Decision-state builder (§13 skeleton).

This module summarizes what the evidence store currently contains and
distills it into a DecisionState record. The skeleton in Phase 3 covers:

  - candidate_ids in scope
  - observed_summary: per-candidate, per-metric counts and missing buckets
  - uncertainty_summary: which metrics have NO observed measurements yet

The fuller frontier / disagreement logic ships in Phase 8. Keeping this
file lean now means the planner (Phase 4) can read a stable
DecisionState object without depending on the frontier code we haven't
written.

This module DOES NOT decide which architecture is best. It surfaces
what's been measured and what hasn't.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..schemas import (
    ACCorePrediction,
    Candidate,
    DecisionState,
    Measurement,
)
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance

# Metric buckets we treat as "key" coverage. A candidate that lacks one of
# these is a candidate the planner should consider measuring.
KEY_METRIC_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("throughput", ("throughput_tps", "decode_throughput_tps", "serving_throughput_tps")),
    ("kernel_bandwidth", ("decode_kv_bandwidth_gbps", "kv_read_bw_gbps")),
    ("quality_loss", ("val_loss", "validation_loss")),
    ("quality_task", ("needle_accuracy", "mqar_accuracy", "downstream_score")),
)


def _bucket_for_metric(metric_name: str) -> str | None:
    for bucket, names in KEY_METRIC_BUCKETS:
        if metric_name in names:
            return bucket
    return None


def build_decision_state(
    store: EvidenceStore,
    *,
    name: str = "current",
    candidate_ids: list[str] | None = None,
) -> DecisionState:
    """Build a DecisionState reflecting current store contents.

    Args:
        store: open EvidenceStore.
        name: human label for the decision state record.
        candidate_ids: optional restriction. If None, includes all candidates.
    """
    all_cands: list[Candidate] = store.list_candidates()
    if candidate_ids is not None:
        all_cands = [c for c in all_cands if c.id in set(candidate_ids)]
    cand_ids = [c.id for c in all_cands]

    preds: list[ACCorePrediction] = [
        p for p in store.list_predictions() if p.candidate_id in cand_ids
    ]
    measurements: list[Measurement] = [
        m for m in store.query_measurements() if m.candidate_id in cand_ids
    ]

    observed_summary = _observed_summary(all_cands, measurements)
    uncertainty_summary = _uncertainty_summary(all_cands, preds, measurements)

    ds = DecisionState(
        id=new_id("ds"),
        name=name,
        candidate_ids=cand_ids,
        observed_summary=observed_summary,
        uncertainty_summary=uncertainty_summary,
        # frontier + disagreements come from Phase 8.
        current_frontier=[],
        disagreements=[],
        recommended_next_experiments=[],
        human_decisions_required=[],
        notes=None,
        provenance=make_provenance(command="decision.build_decision_state"),
    )
    return ds


def _observed_summary(
    cands: list[Candidate], measurements: list[Measurement]
) -> dict[str, Any]:
    """Per-candidate measurement counts and bucket coverage."""
    by_cand: dict[str, dict[str, Any]] = {}
    for c in cands:
        by_cand[c.id] = {
            "n_measurements": 0,
            "metric_counts": {},
            "buckets_present": set(),
        }
    for m in measurements:
        entry = by_cand.setdefault(
            m.candidate_id, {"n_measurements": 0, "metric_counts": {}, "buckets_present": set()}
        )
        entry["n_measurements"] += 1
        entry["metric_counts"][m.metric_name] = (
            entry["metric_counts"].get(m.metric_name, 0) + 1
        )
        bucket = _bucket_for_metric(m.metric_name)
        if bucket:
            entry["buckets_present"].add(bucket)

    # Serialize sets → sorted lists for JSON.
    for cid, entry in by_cand.items():
        entry["buckets_present"] = sorted(entry["buckets_present"])

    return {
        "candidates": by_cand,
        "total_measurements": len(measurements),
        "candidates_with_any_measurement": sum(
            1 for e in by_cand.values() if e["n_measurements"] > 0
        ),
    }


def _uncertainty_summary(
    cands: list[Candidate],
    preds: list[ACCorePrediction],
    measurements: list[Measurement],
) -> dict[str, Any]:
    """Which key buckets are missing for which candidates, and whether the
    candidate has a prediction but no observed measurement of any kind
    ('prediction-only' candidates).
    """
    measured_buckets_by_cand: dict[str, set[str]] = defaultdict(set)
    for m in measurements:
        bucket = _bucket_for_metric(m.metric_name)
        if bucket:
            measured_buckets_by_cand[m.candidate_id].add(bucket)

    preds_by_cand: dict[str, list[ACCorePrediction]] = defaultdict(list)
    for p in preds:
        preds_by_cand[p.candidate_id].append(p)

    missing: dict[str, list[str]] = {}
    prediction_only: list[str] = []
    risky_unmeasured: list[str] = []

    all_bucket_names = [b for b, _ in KEY_METRIC_BUCKETS]
    for c in cands:
        present = measured_buckets_by_cand.get(c.id, set())
        missing[c.id] = [b for b in all_bucket_names if b not in present]
        if not present and preds_by_cand.get(c.id):
            prediction_only.append(c.id)
        # "risky" candidates with no quality measurement are top priority for
        # the planner.
        risks = {p.risk_label for p in preds_by_cand.get(c.id, []) if p.risk_label}
        if (
            ("risky" in risks or "research" in risks)
            and "quality_loss" not in present
            and "quality_task" not in present
        ):
            risky_unmeasured.append(c.id)

    return {
        "missing_buckets_by_candidate": missing,
        "prediction_only_candidates": sorted(prediction_only),
        "risky_unmeasured_candidates": sorted(risky_unmeasured),
        "key_buckets": list(all_bucket_names),
    }
