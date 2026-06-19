# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Frontier computation (§13.1).

Reports the current candidate frontier from observed evidence when it
exists, otherwise from AC-Core predictions. It also surfaces the
overlap and mismatch between observed and predicted views.

The frontier is intentionally *informational*. It does not declare a
winner. Quality metrics (where higher is better) and throughput metrics
(where higher is better) are combined into a Pareto frontier; if no
quality has been observed, only the throughput axis is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..schemas import ACCorePrediction, Candidate, Measurement
from ..store import EvidenceStore


# What we treat as a throughput-axis metric, in priority order.
_THROUGHPUT_METRIC_PRIORITY = (
    "serving_throughput_tps",
    "throughput_tps",
    "decode_throughput_tps",
    "decode_kv_bandwidth_gbps",
)

# Quality-axis metrics — higher is better. val_loss is inverted on use.
_QUALITY_METRIC_PRIORITY_HIGHER_BETTER = (
    "needle_accuracy",
    "mqar_accuracy",
    "downstream_score",
    "copy_accuracy",
)
_QUALITY_METRIC_LOWER_BETTER = ("val_loss", "validation_loss")


@dataclass
class FrontierReport:
    kind: str  # "observed" | "prediction_only" | "mixed"
    observed_supported: list[str]
    prediction_only_supported: list[str]
    observed_failures: list[str]
    surprises: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "observed_supported": list(self.observed_supported),
            "prediction_only_supported": list(self.prediction_only_supported),
            "observed_failures": list(self.observed_failures),
            "surprises": list(self.surprises),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pick_throughput_metric(rows: Iterable[Measurement]) -> str | None:
    names = {r.metric_name for r in rows}
    for m in _THROUGHPUT_METRIC_PRIORITY:
        if m in names:
            return m
    return None


def _pick_quality_metric(rows: Iterable[Measurement]) -> tuple[str | None, bool]:
    names = {r.metric_name for r in rows}
    for m in _QUALITY_METRIC_PRIORITY_HIGHER_BETTER:
        if m in names:
            return m, True
    for m in _QUALITY_METRIC_LOWER_BETTER:
        if m in names:
            return m, False
    return None, True


def _mean_metric(rows: Iterable[Measurement], metric: str) -> float | None:
    vals = [r.metric_value for r in rows if r.metric_name == metric]
    return sum(vals) / len(vals) if vals else None


def _pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Maximize both axes. Returns the non-dominated subset of ids."""
    items = list(points.items())
    front: list[str] = []
    for cid, (x, y) in items:
        dominated = False
        for other_cid, (ox, oy) in items:
            if other_cid == cid:
                continue
            if (ox >= x and oy >= y) and (ox > x or oy > y):
                dominated = True
                break
        if not dominated:
            front.append(cid)
    return front


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def compute_frontier(store: EvidenceStore) -> FrontierReport:
    cands: list[Candidate] = store.list_candidates()
    cand_ids = {c.id for c in cands}
    measurements: list[Measurement] = [
        m for m in store.query_measurements() if m.candidate_id in cand_ids
    ]
    predictions: list[ACCorePrediction] = [
        p for p in store.list_predictions() if p.candidate_id in cand_ids
    ]

    if not measurements and not predictions:
        return FrontierReport("observed", [], [], [], [])

    if not measurements:
        # Prediction-only frontier.
        obs_pts = _predicted_points(predictions)
        front = _pareto_front(obs_pts) if obs_pts else sorted(cand_ids)
        return FrontierReport("prediction_only", [], sorted(front), [], [])

    # Observed-supported frontier.
    t_metric = _pick_throughput_metric(measurements)
    q_metric, q_higher_better = _pick_quality_metric(measurements)

    observed_points: dict[str, tuple[float, float]] = {}
    observed_failures: list[str] = []
    for cid in sorted(cand_ids):
        cand_rows = [m for m in measurements if m.candidate_id == cid]
        if not cand_rows:
            continue
        t_val = _mean_metric(cand_rows, t_metric) if t_metric else None
        if q_metric:
            q_raw = _mean_metric(cand_rows, q_metric)
            if q_raw is None:
                q_val = None
            else:
                q_val = q_raw if q_higher_better else -q_raw
        else:
            q_val = None

        if t_val is None and q_val is None:
            continue
        # If we only have one axis, score the other as 0 so Pareto reduces to
        # ranking on the available axis.
        observed_points[cid] = (t_val if t_val is not None else 0.0,
                                 q_val if q_val is not None else 0.0)
        if t_val is not None and t_val == 0.0:
            observed_failures.append(cid)

    observed_supported = _pareto_front(observed_points) if observed_points else []

    # Prediction-only supported: predicted Pareto winners that aren't yet on
    # the observed Pareto.
    pred_points = _predicted_points(predictions)
    pred_pareto = _pareto_front(pred_points) if pred_points else []
    pred_only_supported = [
        cid for cid in pred_pareto if cid not in set(observed_supported)
    ]

    # Surprises: predicted-winner candidates with observed measurements that
    # come in worse than baseline / prediction.
    surprises = _detect_surprises(
        cand_ids, measurements, predictions, t_metric=t_metric
    )

    return FrontierReport(
        kind="mixed" if pred_only_supported else "observed",
        observed_supported=sorted(observed_supported),
        prediction_only_supported=sorted(pred_only_supported),
        observed_failures=sorted(set(observed_failures)),
        surprises=surprises,
    )


def _predicted_points(
    predictions: list[ACCorePrediction],
) -> dict[str, tuple[float, float]]:
    """Map candidate_id → (throughput-axis, quality-axis) from predictions."""
    out: dict[str, tuple[float, float]] = {}
    for p in predictions:
        m = p.predicted_metrics or {}
        t = (
            m.get("serving_throughput_tps")
            or m.get("throughput_tps")
            or 0.0
        )
        q = m.get("quality_score") or m.get("needle_accuracy") or 0.0
        if t or q:
            out[p.candidate_id] = (float(t), float(q))
    return out


def _detect_surprises(
    cand_ids: set[str],
    measurements: list[Measurement],
    predictions: list[ACCorePrediction],
    *,
    t_metric: str | None,
) -> list[dict[str, object]]:
    """Find cases where AC-Core predicted a throughput gain that didn't
    materialize observed.
    """
    if t_metric is None:
        return []
    pred_t = {p.candidate_id: (p.predicted_metrics or {}).get("throughput_tps") or
              (p.predicted_metrics or {}).get("serving_throughput_tps")
              for p in predictions}

    # Treat the baseline as the candidate with the lowest predicted throughput
    # (good-enough heuristic for v1).
    candidates_with_pred = [c for c in cand_ids if pred_t.get(c)]
    if not candidates_with_pred:
        return []
    baseline_id = min(candidates_with_pred, key=lambda c: pred_t[c])
    baseline_obs = _mean_metric(
        [m for m in measurements if m.candidate_id == baseline_id], t_metric
    )
    if baseline_obs is None:
        return []

    out: list[dict[str, object]] = []
    for cid in candidates_with_pred:
        if cid == baseline_id:
            continue
        cand_obs = _mean_metric(
            [m for m in measurements if m.candidate_id == cid], t_metric
        )
        if cand_obs is None:
            continue
        predicted_gain = (pred_t[cid] or 0) > (pred_t[baseline_id] or 0) * 1.05
        observed_gain = cand_obs > baseline_obs * 1.02
        if predicted_gain and not observed_gain:
            out.append({
                "candidate_id": cid,
                "type": "predicted_serving_gain_not_observed",
                "predicted_throughput": pred_t[cid],
                "observed_throughput": cand_obs,
                "baseline_observed": baseline_obs,
                "metric": t_metric,
            })
    return out
