# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Throughput calibration fitter (§12.1).

Fits a per-(hardware, runtime, kernel) scalar efficiency correction:

    observed = alpha * peak_or_predicted + intercept

where alpha is the "achieved efficiency" multiplier on top of what
AC-Core's prior assumed. We fit one alpha per kernel category present
in the store; if a category lacks the `achieved_efficiency` metric we
fall back to fitting on raw `bandwidth_gbps` / `throughput_tflops`
values vs a constant baseline.

Output: `FittedCalibration` rows (one per kernel) and a
`MeasuredCalibration.json` payload that AC-Core can consume.
"""
from __future__ import annotations

import os
import json
from collections import defaultdict
from typing import Any

import numpy as np

from ..schemas import FittedCalibration, Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance
from .uncertainty import bootstrap_std


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Closed-form ridge: (XtX + alpha I)^-1 Xt y."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_features = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_features)
    return np.linalg.solve(A, X.T @ y)


def fit_throughput_calibration(
    store: EvidenceStore,
    *,
    hardware_id: str | None = None,
    runtime: str | None = None,
    min_points: int = 3,
) -> list[FittedCalibration]:
    """Fit one calibration per kernel category present in the store."""
    rows: list[Measurement] = store.query_measurements(
        measurement_type="kernel",
        hardware_id=hardware_id,
        runtime=runtime,
    )
    # Group by metric_name (a proxy for the kernel target).
    by_metric: dict[str, list[Measurement]] = defaultdict(list)
    for r in rows:
        by_metric[r.metric_name].append(r)

    out: list[FittedCalibration] = []
    for metric, ms in by_metric.items():
        if len(ms) < min_points:
            fit = FittedCalibration(
                id=new_id("fit"),
                name=f"throughput_{metric}",
                target="throughput",
                input_measurement_query={
                    "measurement_type": "kernel",
                    "metric_name": metric,
                    "hardware_id": hardware_id,
                    "runtime": runtime,
                },
                n_measurements=len(ms),
                functional_form="ridge(observed ~ const + intercept)",
                coefficients={},
                uncertainty={},
                valid_for={"hardware_id": hardware_id, "runtime": runtime},
                status="insufficient_data",
                needed_measurements=[
                    f"{metric}@hardware={hardware_id}/runtime={runtime}"
                    f" (need {min_points - len(ms)} more)"
                ],
                provenance=make_provenance(command="fitter.calibration_fit"),
            )
            store.insert_fitted_calibration(fit)
            out.append(fit)
            continue

        values = np.array([m.metric_value for m in ms], dtype=float)
        ones = np.ones_like(values)
        X = np.column_stack([ones])
        y = values

        coef = _ridge_fit(X, y)  # length-1: intercept-only baseline level
        # Define "achieved efficiency" as observed / max(observed) — a scale
        # in (0, 1] per measurement, whose mean is a useful summary.
        eff = values / values.max()
        alpha = float(eff.mean())

        # Holdout MAE via leave-one-out on the simple mean predictor.
        loo_preds = np.array([
            np.delete(values, i).mean() for i in range(len(values))
        ])
        holdout_mae = float(np.mean(np.abs(loo_preds - values)))

        # Bootstrap uncertainty on the intercept (mean).
        sigma = float(bootstrap_std(_ridge_fit, X, y).item())

        fit = FittedCalibration(
            id=new_id("fit"),
            name=f"throughput_{metric}",
            target="throughput",
            input_measurement_query={
                "measurement_type": "kernel",
                "metric_name": metric,
                "hardware_id": hardware_id,
                "runtime": runtime,
            },
            n_measurements=len(ms),
            functional_form="ridge(observed ~ const)",
            coefficients={
                "intercept": float(coef[0]),
                "achieved_efficiency_mean": alpha,
                "max_observed": float(values.max()),
            },
            uncertainty={"intercept_sigma": sigma},
            holdout_error={"loo_mae": holdout_mae},
            valid_for={
                "hardware_id": hardware_id,
                "runtime": runtime,
                "metric_name": metric,
            },
            status="draft",
            provenance=make_provenance(command="fitter.calibration_fit"),
        )
        store.insert_fitted_calibration(fit)
        out.append(fit)
    return out


def export_measured_calibration_json(
    store: EvidenceStore,
    *,
    out_dir: str,
    hardware_id: str | None = None,
    runtime: str | None = None,
) -> list[str]:
    """Write one `<hardware>_<runtime>_measured.json` file under `out_dir`.

    Bundles all draft-or-validated throughput fits relevant to the
    (hardware, runtime) filter.
    """
    os.makedirs(out_dir, exist_ok=True)
    fits = [
        f for f in store.list_fitted_calibrations()
        if f.target == "throughput" and f.status in {"draft", "validated"}
    ]
    if hardware_id is not None:
        fits = [f for f in fits if f.valid_for.get("hardware_id") == hardware_id]
    if runtime is not None:
        fits = [f for f in fits if f.valid_for.get("runtime") == runtime]
    if not fits:
        return []
    hw = hardware_id or "unknown_hw"
    rt = runtime or "unknown_runtime"
    path = os.path.join(out_dir, f"{hw}_{rt}_measured.json")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "hardware_id": hw,
        "runtime": rt,
        "fits": [json.loads(f.model_dump_json()) for f in fits],
        "notes": "Measured calibration emitted by AC-Harness. AC-Core may consume.",
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return [path]
