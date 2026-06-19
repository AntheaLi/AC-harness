# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Interaction fitter (§12.3).

Fits non-additive cross-terms only when we have enough data. Returns a
single FittedCalibration; if data is insufficient, the row is emitted
with `status="insufficient_data"` and a `needed_measurements` list
describing what to collect next.

The fitter NEVER silently fits an unstable law.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np

from ..schemas import Candidate, FittedCalibration, Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance
from .calibration_fit import _ridge_fit
from .uncertainty import bootstrap_std


def _build_interaction_matrix(
    candidates: list[Candidate],
    rows: list[Measurement],
    feature_a: str,
    feature_b: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X with [a, b, a*b, 1], y, candidate_ids_used)."""
    cand_by_id = {c.id: c for c in candidates}
    X_list, y_list, ids = [], [], []
    for m in rows:
        c = cand_by_id.get(m.candidate_id)
        if c is None:
            continue
        cf = c.changed_fields or {}
        try:
            a = float(cf.get(feature_a))
            b = float(cf.get(feature_b))
        except (TypeError, ValueError):
            continue
        X_list.append([a, b, a * b, 1.0])
        y_list.append(m.metric_value)
        ids.append(c.id)
    if not X_list:
        return np.zeros((0, 4)), np.zeros(0), []
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float), ids


def fit_interaction(
    store: EvidenceStore,
    *,
    metric_name: str,
    feature_a: str,
    feature_b: str,
    min_points: int = 5,
) -> FittedCalibration:
    candidates: list[Candidate] = store.list_candidates()
    rows: list[Measurement] = store.query_measurements(metric_name=metric_name)

    X, y, ids = _build_interaction_matrix(
        candidates, rows, feature_a, feature_b
    )

    name = f"interaction_{feature_a}_x_{feature_b}_on_{metric_name}"
    if len(y) < min_points:
        fit = FittedCalibration(
            id=new_id("fit"),
            name=name,
            target="interaction",
            input_measurement_query={
                "metric_name": metric_name,
                "feature_a": feature_a,
                "feature_b": feature_b,
            },
            n_measurements=len(y),
            functional_form=f"ridge({metric_name} ~ {feature_a} + {feature_b} + {feature_a}*{feature_b} + 1)",
            coefficients={},
            uncertainty={},
            valid_for={"feature_a": feature_a, "feature_b": feature_b},
            status="insufficient_data",
            needed_measurements=[
                f"{min_points - len(y)} more candidates with both "
                f"{feature_a} and {feature_b} set, measuring {metric_name}"
            ],
            provenance=make_provenance(command="fitter.interaction_fit"),
        )
        store.insert_fitted_calibration(fit)
        return fit

    # Need at least 2 distinct values per feature; otherwise the
    # interaction term is unidentifiable. Refuse to fit silently.
    if len({float(row[0]) for row in X}) < 2 or len({float(row[1]) for row in X}) < 2:
        fit = FittedCalibration(
            id=new_id("fit"),
            name=name,
            target="interaction",
            input_measurement_query={
                "metric_name": metric_name,
                "feature_a": feature_a,
                "feature_b": feature_b,
            },
            n_measurements=len(y),
            functional_form="ridge(unidentifiable)",
            coefficients={},
            uncertainty={},
            valid_for={"feature_a": feature_a, "feature_b": feature_b},
            status="insufficient_data",
            needed_measurements=[
                f"need at least 2 distinct values for both {feature_a} and {feature_b}"
            ],
            provenance=make_provenance(command="fitter.interaction_fit"),
        )
        store.insert_fitted_calibration(fit)
        return fit

    coef = _ridge_fit(X, y, alpha=0.1)
    sigma = bootstrap_std(_ridge_fit, X, y, seed=0)
    names = [feature_a, feature_b, f"{feature_a}*{feature_b}", "intercept"]
    coefs = {names[i]: float(coef[i]) for i in range(4)}
    uncert = {names[i]: float(sigma[i]) for i in range(4)}

    fit = FittedCalibration(
        id=new_id("fit"),
        name=name,
        target="interaction",
        input_measurement_query={
            "metric_name": metric_name,
            "feature_a": feature_a,
            "feature_b": feature_b,
        },
        n_measurements=len(y),
        functional_form=f"ridge({metric_name} ~ {feature_a} + {feature_b} + {feature_a}*{feature_b} + 1)",
        coefficients=coefs,
        uncertainty=uncert,
        valid_for={
            "feature_a": feature_a,
            "feature_b": feature_b,
            "candidate_ids": ids,
        },
        status="draft",
        provenance=make_provenance(command="fitter.interaction_fit"),
    )
    store.insert_fitted_calibration(fit)
    return fit


def export_interaction_terms_json(
    store: EvidenceStore, *, out_dir: str
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    fits = [
        f for f in store.list_fitted_calibrations()
        if f.target == "interaction" and f.status in {"draft", "validated"}
    ]
    paths: list[str] = []
    for f in fits:
        path = os.path.join(out_dir, f"{f.name}.json")
        with open(path, "w") as fp:
            json.dump(json.loads(f.model_dump_json()), fp, indent=2, sort_keys=True)
        paths.append(path)
    return paths
