"""EvidenceStore tests (§6)."""
from __future__ import annotations

import os
import tempfile

import pytest

from ac_harness.schemas import (
    ACCorePrediction,
    Candidate,
    DecisionState,
    ExperimentPlan,
    FittedCalibration,
    HumanDecision,
    Measurement,
)
from ac_harness.store import EvidenceStore, new_id


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "store.sqlite"
    s = EvidenceStore(str(db))
    yield s
    s.close()


def _make_candidate(id="cand1", baseline="baseline"):
    return Candidate(
        id=id,
        source="ac_core",
        baseline_id=baseline,
        changed_fields={"n_kv_heads": 8},
    )


def test_insert_and_get_candidate(store):
    cid = store.insert_candidate(_make_candidate())
    assert cid == "cand1"
    back = store.get_candidate("cand1")
    assert back is not None
    assert back.changed_fields == {"n_kv_heads": 8}
    # provenance was auto-populated
    assert back.provenance.get("created_at")


def test_insert_prediction_and_measurement(store):
    store.insert_candidate(_make_candidate())
    pred = ACCorePrediction(
        id="pred1",
        candidate_id="cand1",
        hardware_id="h100_sxm",
        runtime="vllm",
        workload_id="long_chat",
        predicted_metrics={"throughput_tps": 1200.0},
        predicted_bottlenecks=["decode_kv"],
    )
    store.insert_prediction(pred)

    plan = ExperimentPlan(
        id="plan1",
        name="decode_kv",
        experiment_type="kernel_microbench",
        candidate_ids=["cand1"],
        decision_unblocked="serving usefulness of GQA-8",
        uncertainty_target="decode KV bandwidth",
    )
    store.insert_experiment_plan(plan)

    m = Measurement(
        id=new_id("meas"),
        candidate_id="cand1",
        experiment_id="plan1",
        measurement_type="kernel",
        metric_name="decode_kv_bandwidth_gbps",
        metric_value=820.5,
        metric_unit="GB/s",
        hardware_id="h100_sxm",
        runtime="vllm",
    )
    store.insert_measurement(m)

    rows = store.query_measurements(candidate_id="cand1")
    assert len(rows) == 1
    assert rows[0].metric_value == 820.5

    rows = store.query_measurements(metric_name="decode_kv_bandwidth_gbps")
    assert len(rows) == 1


def test_validate_detects_orphan_prediction(store):
    # insert a prediction whose candidate doesn't exist
    pred = ACCorePrediction(
        id="pred_orphan",
        candidate_id="ghost",
        hardware_id="h100",
        runtime="vllm",
        workload_id="x",
    )
    store.insert_prediction(pred)
    errs = store.validate()
    assert any("ghost" in e for e in errs)


def test_validate_detects_orphan_measurement(store):
    m = Measurement(
        id="meas_orphan",
        candidate_id="ghost",
        measurement_type="kernel",
        metric_name="x",
        metric_value=0.0,
    )
    store.insert_measurement(m)
    errs = store.validate()
    assert any("ghost" in e for e in errs)


def test_validate_detects_orphan_experiment_ref(store):
    store.insert_candidate(_make_candidate())
    m = Measurement(
        id="meas_orphan_exp",
        candidate_id="cand1",
        experiment_id="ghost_plan",
        measurement_type="kernel",
        metric_name="x",
        metric_value=0.0,
    )
    store.insert_measurement(m)
    errs = store.validate()
    assert any("ghost_plan" in e for e in errs)


def test_export_jsonl_roundtrips(store, tmp_path):
    store.insert_candidate(_make_candidate())
    store.insert_prediction(
        ACCorePrediction(
            id="pred1",
            candidate_id="cand1",
            hardware_id="h100",
            runtime="vllm",
            workload_id="x",
        )
    )
    out = tmp_path / "export"
    store.export_jsonl(str(out))
    assert (out / "candidates.jsonl").exists()
    assert (out / "predictions.jsonl").exists()
    with open(out / "candidates.jsonl") as f:
        line = f.readline()
    assert "cand1" in line


def test_fitted_calibration_and_decision_state_roundtrip(store):
    fit = FittedCalibration(
        id="fit1",
        name="throughput",
        target="throughput",
        n_measurements=42,
        functional_form="affine",
        coefficients={"a": 1.0},
        uncertainty={"sigma": 0.05},
        status="draft",
    )
    store.insert_fitted_calibration(fit)
    assert store.get_fitted_calibration("fit1").n_measurements == 42

    ds = DecisionState(id="ds1", name="initial", candidate_ids=["cand1"])
    store.insert_decision_state(ds)
    assert store.get_decision_state("ds1").name == "initial"


def test_human_decision_record(store):
    hd = HumanDecision(
        id="h1",
        decision_type="approve_expensive_run",
        prompt="run 5h MoE all-to-all?",
        options=["approve", "skip"],
    )
    store.insert_human_decision(hd)
    # no public getter required; verify via direct SQL count
    cur = store._conn.execute("SELECT COUNT(*) FROM human_decisions")
    assert cur.fetchone()[0] == 1
