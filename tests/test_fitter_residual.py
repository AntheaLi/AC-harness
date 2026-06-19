"""Quality residual + interaction fitter tests (§12.2, §12.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ac_harness.fitter import (
    export_interaction_terms_json,
    export_quality_residuals_json,
    fit_interaction,
    fit_quality_residual,
)
from ac_harness.ingest import import_ac_core_run
from ac_harness.schemas import Candidate, Measurement
from ac_harness.store import EvidenceStore, new_id


REPO_ROOT = Path(__file__).resolve().parents[1]
AC = REPO_ROOT / "examples" / "ac_core_outputs" / "llama_h100_long_chat"


@pytest.fixture()
def store(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    import_ac_core_run(str(AC), s)
    yield s
    s.close()


def _add_val_loss(s: EvidenceStore, cid: str, val: float):
    s.insert_measurement(
        Measurement(
            id=new_id("meas"),
            candidate_id=cid,
            measurement_type="eval",
            metric_name="val_loss",
            metric_value=val,
        )
    )


def test_quality_residual_insufficient_data(store):
    # Only one quality measurement → insufficient.
    _add_val_loss(store, "llama_gqa8", 2.1)
    fits = fit_quality_residual(store, min_points=3)
    assert fits[0].status == "insufficient_data"
    assert fits[0].needed_measurements


def test_quality_residual_fits_with_enough_points(store):
    _add_val_loss(store, "llama_baseline", 2.00)
    _add_val_loss(store, "llama_gqa8", 2.10)
    _add_val_loss(store, "llama_gqa4", 2.20)
    _add_val_loss(store, "llama_gqa8_kvfp8", 2.15)
    fits = fit_quality_residual(store)
    fit = fits[0]
    assert fit.status == "draft"
    assert fit.n_measurements == 4
    assert "__intercept__" in fit.coefficients
    assert "__intercept__" in fit.uncertainty


def test_interaction_insufficient_data(store):
    fit = fit_interaction(
        store,
        metric_name="val_loss",
        feature_a="n_kv_heads",
        feature_b="d_model",
    )
    assert fit.status == "insufficient_data"


def test_interaction_unidentifiable_when_one_feature_constant(store):
    # Add measurements where only n_kv_heads varies but d_model is constant.
    # (d_model isn't set on the AC-Core fixture candidates → all 0.0)
    for cid, v in [("llama_baseline", 1.0), ("llama_gqa8", 0.9),
                    ("llama_gqa4", 0.8), ("llama_gqa8_kvfp8", 0.85),
                    ("llama_gqa8_kvfp8", 0.86)]:
        s = store
        s.insert_measurement(
            Measurement(
                id=new_id("meas"),
                candidate_id=cid,
                measurement_type="kernel",
                metric_name="custom_metric",
                metric_value=v,
            )
        )
    fit = fit_interaction(
        store,
        metric_name="custom_metric",
        feature_a="n_kv_heads",
        feature_b="d_model",
    )
    assert fit.status == "insufficient_data"


def test_interaction_fits_with_distinct_values(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    for cid, a, b in [
        ("c1", 1.0, 1.0),
        ("c2", 1.0, 2.0),
        ("c3", 2.0, 1.0),
        ("c4", 2.0, 2.0),
        ("c5", 3.0, 3.0),
        ("c6", 3.0, 1.0),
    ]:
        s.insert_candidate(
            Candidate(
                id=cid,
                source="manual",
                changed_fields={"alpha": a, "beta": b},
            )
        )
        s.insert_measurement(
            Measurement(
                id=new_id("meas"),
                candidate_id=cid,
                measurement_type="kernel",
                metric_name="target",
                metric_value=a + 2 * b + 0.5 * a * b + 0.01,
            )
        )
    fit = fit_interaction(
        s, metric_name="target", feature_a="alpha", feature_b="beta", min_points=4
    )
    assert fit.status == "draft"
    assert "alpha" in fit.coefficients
    assert "alpha*beta" in fit.coefficients
    s.close()


def test_export_quality_residuals_and_interaction(store, tmp_path):
    for cid, v in [
        ("llama_baseline", 2.0),
        ("llama_gqa8", 2.1),
        ("llama_gqa4", 2.2),
        ("llama_gqa8_kvfp8", 2.15),
    ]:
        _add_val_loss(store, cid, v)
    fit_quality_residual(store)
    out = tmp_path / "feedback" / "quality_residuals"
    paths = export_quality_residuals_json(store, out_dir=str(out))
    assert paths
    assert all(p.endswith(".json") for p in paths)

    # Interaction file emits even when only insufficient_data fits exist if a
    # draft fit hasn't been produced — confirm export skips those.
    fit_interaction(store, metric_name="val_loss", feature_a="n_kv_heads", feature_b="d_model")
    out2 = tmp_path / "feedback" / "interactions"
    paths2 = export_interaction_terms_json(store, out_dir=str(out2))
    # Should be empty since the interaction fit was insufficient_data.
    assert paths2 == []
