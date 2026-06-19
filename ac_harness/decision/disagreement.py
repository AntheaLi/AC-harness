# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Disagreement detection (§13.2).

Returns a list of dicts describing places where observed evidence
contradicts AC-Core's prior, or where two evaluators disagree with
each other on the same candidate.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..schemas import ACCorePrediction, Measurement
from ..store import EvidenceStore


def detect_disagreements(store: EvidenceStore) -> list[dict[str, Any]]:
    preds: list[ACCorePrediction] = store.list_predictions()
    measurements: list[Measurement] = store.query_measurements()

    by_cand_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in measurements:
        by_cand_metric[m.candidate_id][m.metric_name].append(m.metric_value)

    out: list[dict[str, Any]] = []

    # 1. AC-Core predicted serving gain but benchmark showed no gain.
    out.extend(_predicted_serving_gain_not_observed(preds, by_cand_metric))

    # 2. AC-Core predicted low quality risk but ablation showed high val_loss delta.
    out.extend(_low_risk_but_quality_regression(preds, by_cand_metric))

    # 3. Perplexity improved but long-context recall worsened.
    out.extend(_ppl_vs_long_context(by_cand_metric))

    # 4. MoE throughput improved but load imbalance increased.
    out.extend(_moe_throughput_vs_imbalance(by_cand_metric))

    # 5. State/hybrid serving improved but recall failed.
    out.extend(_state_serving_vs_recall(preds, by_cand_metric))

    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #


def _predicted_serving_gain_not_observed(
    preds: list[ACCorePrediction],
    by: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pred_by_cand = {p.candidate_id: p for p in preds}
    if not pred_by_cand:
        return out
    # Baseline = the candidate with the lowest predicted throughput.
    pred_t = {
        cid: (p.predicted_metrics or {}).get("throughput_tps")
        or (p.predicted_metrics or {}).get("serving_throughput_tps")
        for cid, p in pred_by_cand.items()
    }
    pred_t = {k: v for k, v in pred_t.items() if v}
    if not pred_t:
        return out
    baseline_id = min(pred_t, key=lambda k: pred_t[k])
    baseline_obs = _mean(by.get(baseline_id, {}).get("throughput_tps", []) or
                         by.get(baseline_id, {}).get("serving_throughput_tps", []))
    if baseline_obs == 0:
        return out
    for cid, predicted in pred_t.items():
        if cid == baseline_id:
            continue
        cand_obs = _mean(by.get(cid, {}).get("throughput_tps", []) or
                         by.get(cid, {}).get("serving_throughput_tps", []))
        if cand_obs == 0:
            continue
        predicted_gain = predicted > pred_t[baseline_id] * 1.05
        observed_gain = cand_obs > baseline_obs * 1.02
        if predicted_gain and not observed_gain:
            out.append({
                "rule": "predicted_serving_gain_not_observed",
                "candidate_id": cid,
                "baseline_id": baseline_id,
                "predicted_throughput": predicted,
                "observed_throughput": cand_obs,
                "baseline_observed_throughput": baseline_obs,
            })
    return out


def _low_risk_but_quality_regression(
    preds: list[ACCorePrediction],
    by: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    safe_cands = {p.candidate_id for p in preds if (p.risk_label or "") in {"safe", ""}}
    if not by:
        return out
    val_losses = {
        cid: _mean(metrics.get("val_loss", []) or metrics.get("validation_loss", []))
        for cid, metrics in by.items()
        if metrics.get("val_loss") or metrics.get("validation_loss")
    }
    if not val_losses:
        return out
    # Choose the candidate with the smallest val_loss as the "baseline" of
    # quality for comparison.
    baseline_val = min(val_losses.values())
    threshold = baseline_val * 1.05  # 5% loss regression flagged
    for cid in safe_cands:
        v = val_losses.get(cid)
        if v is None:
            continue
        if v > threshold:
            out.append({
                "rule": "low_risk_but_quality_regression",
                "candidate_id": cid,
                "observed_val_loss": v,
                "baseline_val_loss": baseline_val,
            })
    return out


def _ppl_vs_long_context(
    by: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """For each candidate that has BOTH val_loss/perplexity and needle_accuracy,
    flag if perplexity dropped (good) but needle accuracy dropped (bad) vs the
    best observed val_loss candidate.
    """
    val_losses = {cid: _mean(m.get("val_loss", []) or m.get("perplexity", []))
                  for cid, m in by.items()
                  if m.get("val_loss") or m.get("perplexity")}
    needle = {cid: _mean(m.get("needle_accuracy", []))
              for cid, m in by.items() if m.get("needle_accuracy")}
    common = set(val_losses) & set(needle)
    if len(common) < 2:
        return []
    best_loss_cand = min(common, key=lambda c: val_losses[c])
    best_loss = val_losses[best_loss_cand]
    best_needle = needle[best_loss_cand]
    out: list[dict[str, Any]] = []
    for cid in common:
        if cid == best_loss_cand:
            continue
        if val_losses[cid] < best_loss * 1.0 and needle[cid] < best_needle * 0.9:
            out.append({
                "rule": "ppl_improved_but_long_context_worsened",
                "candidate_id": cid,
                "val_loss": val_losses[cid],
                "needle_accuracy": needle[cid],
                "compared_to": best_loss_cand,
            })
    return out


def _moe_throughput_vs_imbalance(
    by: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cid, metrics in by.items():
        if not metrics.get("load_imbalance"):
            continue
        if not (metrics.get("throughput_tps") or metrics.get("serving_throughput_tps")):
            continue
        imbalance = _mean(metrics["load_imbalance"])
        tps = _mean(metrics.get("throughput_tps", []) or metrics.get("serving_throughput_tps", []))
        # Crude rule: flag if any imbalance > 0.2 alongside throughput data.
        if imbalance > 0.2 and tps > 0:
            out.append({
                "rule": "moe_throughput_improved_but_load_imbalance_high",
                "candidate_id": cid,
                "load_imbalance": imbalance,
                "throughput_tps": tps,
            })
    return out


def _state_serving_vs_recall(
    preds: list[ACCorePrediction],
    by: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    state_cands = {p.candidate_id for p in preds
                   if "state" in (p.predicted_bottlenecks or [])
                   or "state_scan" in (p.predicted_bottlenecks or [])}
    for cid in state_cands:
        m = by.get(cid, {})
        tps = _mean(m.get("serving_throughput_tps", []) or m.get("throughput_tps", []))
        recall = _mean(m.get("needle_accuracy", []) or m.get("mqar_accuracy", []))
        if tps > 0 and recall > 0 and recall < 0.5:
            out.append({
                "rule": "state_serving_improved_but_recall_failed",
                "candidate_id": cid,
                "throughput": tps,
                "recall": recall,
            })
    return out
