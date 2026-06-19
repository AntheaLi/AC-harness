# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Quality residual fitter (§12.2).

Fits a residual law on top of AC-Core's compile-time quality prior:

    observed_quality = predicted_quality + residual(changed_fields)

For v1 the residual is a ridge regression on a small feature vector
extracted from `Candidate.changed_fields` (numeric values only). If the
prediction has no `quality_score`, the residual is fit against raw
observed value.

Output: `FittedCalibration` rows with `target="quality_residual"` plus
a JSON payload for AC-Core consumption.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from ..schemas import ACCorePrediction, Candidate, FittedCalibration, Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance
from .calibration_fit import _ridge_fit
from .uncertainty import bootstrap_std


_QUALITY_METRICS_HIGHER_BETTER = ("needle_accuracy", "mqar_accuracy", "downstream_score", "copy_accuracy")
_QUALITY_METRICS_LOWER_BETTER = ("val_loss", "validation_loss")


def _featurize(cand: Candidate, feature_keys: list[str]) -> np.ndarray:
    out: list[float] = []
    cf = cand.changed_fields or {}
    for k in feature_keys:
        v = cf.get(k)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return np.array(out, dtype=float)


def _collect_feature_keys(candidates: Iterable[Candidate]) -> list[str]:
    keys: set[str] = set()
    for c in candidates:
        for k, v in (c.changed_fields or {}).items():
            try:
                float(v)
            except (TypeError, ValueError):
                continue
            keys.add(k)
    return sorted(keys)


def fit_quality_residual(
    store: EvidenceStore,
    *,
    metric_name: str | None = None,
    min_points: int = 3,
) -> list[FittedCalibration]:
    """Fit residuals per quality metric present in the store."""
    candidates: list[Candidate] = store.list_candidates()
    cand_by_id = {c.id: c for c in candidates}
    preds: list[ACCorePrediction] = store.list_predictions()
    pred_q_by_cand: dict[str, float] = {}
    for p in preds:
        q = (p.predicted_metrics or {}).get("quality_score")
        if q is not None:
            pred_q_by_cand[p.candidate_id] = float(q)

    rows: list[Measurement] = store.query_measurements(measurement_type="eval")
    if metric_name:
        rows = [r for r in rows if r.metric_name == metric_name]
    by_metric: dict[str, list[Measurement]] = defaultdict(list)
    for r in rows:
        if r.metric_name in _QUALITY_METRICS_HIGHER_BETTER + _QUALITY_METRICS_LOWER_BETTER:
            by_metric[r.metric_name].append(r)

    feature_keys = _collect_feature_keys(candidates)
    out: list[FittedCalibration] = []

    for metric, ms in by_metric.items():
        higher_better = metric in _QUALITY_METRICS_HIGHER_BETTER
        Xs, ys, ids_used = [], [], []
        for m in ms:
            c = cand_by_id.get(m.candidate_id)
            if c is None:
                continue
            feats = _featurize(c, feature_keys)
            pred_q = pred_q_by_cand.get(m.candidate_id, 0.0)
            obs = m.metric_value
            if higher_better:
                residual = obs - pred_q
            else:
                residual = pred_q - obs if pred_q else -obs
            Xs.append(feats)
            ys.append(residual)
            ids_used.append(m.candidate_id)

        if len(ys) < min_points:
            fit = FittedCalibration(
                id=new_id("fit"),
                name=f"quality_residual_{metric}",
                target="quality_residual",
                input_measurement_query={
                    "measurement_type": "eval",
                    "metric_name": metric,
                },
                n_measurements=len(ys),
                functional_form=f"ridge(residual ~ {feature_keys})",
                coefficients={},
                uncertainty={},
                valid_for={"metric_name": metric, "feature_keys": feature_keys},
                status="insufficient_data",
                needed_measurements=[
                    f"{metric} eval rows for {min_points - len(ys)} more candidates"
                ],
                provenance=make_provenance(command="fitter.residual_fit"),
            )
            store.insert_fitted_calibration(fit)
            out.append(fit)
            continue

        X = np.stack(Xs) if Xs else np.zeros((0, len(feature_keys)))
        y = np.array(ys, dtype=float)

        # Add intercept column.
        X_aug = np.column_stack([X, np.ones(len(X))])
        coef = _ridge_fit(X_aug, y, alpha=0.1)
        coefs_dict = {k: float(coef[i]) for i, k in enumerate(feature_keys)}
        coefs_dict["__intercept__"] = float(coef[-1])

        sigma = bootstrap_std(_ridge_fit, X_aug, y, seed=0)
        sigma_dict = {k: float(sigma[i]) for i, k in enumerate(feature_keys)}
        sigma_dict["__intercept__"] = float(sigma[-1])

        # Holdout MAE via leave-one-out.
        if len(y) >= 4:
            errs = []
            for i in range(len(y)):
                X_train = np.delete(X_aug, i, axis=0)
                y_train = np.delete(y, i)
                c = _ridge_fit(X_train, y_train, alpha=0.1)
                pred = X_aug[i] @ c
                errs.append(abs(pred - y[i]))
            holdout = {"loo_mae": float(np.mean(errs))}
        else:
            holdout = None

        fit = FittedCalibration(
            id=new_id("fit"),
            name=f"quality_residual_{metric}",
            target="quality_residual",
            input_measurement_query={
                "measurement_type": "eval",
                "metric_name": metric,
            },
            n_measurements=len(y),
            functional_form=f"ridge(residual ~ {feature_keys} + intercept)",
            coefficients=coefs_dict,
            uncertainty=sigma_dict,
            holdout_error=holdout,
            valid_for={
                "metric_name": metric,
                "feature_keys": feature_keys,
                "candidate_ids": ids_used,
            },
            status="draft",
            provenance=make_provenance(command="fitter.residual_fit"),
        )
        store.insert_fitted_calibration(fit)
        out.append(fit)
    return out


def export_quality_residuals_json(
    store: EvidenceStore, *, out_dir: str
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    fits = [
        f for f in store.list_fitted_calibrations()
        if f.target == "quality_residual" and f.status in {"draft", "validated"}
    ]
    paths: list[str] = []
    for f in fits:
        path = os.path.join(out_dir, f"{f.name}.json")
        with open(path, "w") as fp:
            json.dump(json.loads(f.model_dump_json()), fp, indent=2, sort_keys=True)
        paths.append(path)
    return paths
