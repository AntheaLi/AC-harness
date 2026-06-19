"""Schema round-trip + tolerance tests (§5)."""
from __future__ import annotations

import json

import pytest

from ac_harness.schemas import (
    ACCorePrediction,
    ALL_SCHEMAS,
    Candidate,
    DecisionState,
    ExperimentPlan,
    FittedCalibration,
    HumanDecision,
    Measurement,
)


def test_all_schemas_have_schema_version():
    for cls in ALL_SCHEMAS:
        # construct with minimum fields by inspecting required ones
        assert "schema_version" in cls.model_fields
        assert cls.model_fields["schema_version"].default == 1


def test_candidate_round_trip():
    c = Candidate(
        id="cand1",
        source="ac_core",
        changed_fields={"n_kv_heads": 8},
        candidate_metadata={"family": "llama"},
        ac_core_prediction_id="pred1",
    )
    raw = c.model_dump_json()
    back = Candidate.model_validate_json(raw)
    assert back.id == "cand1"
    assert back.source == "ac_core"
    assert back.changed_fields == {"n_kv_heads": 8}


def test_candidate_unknown_field_tolerated():
    raw = json.dumps({
        "schema_version": 1,
        "id": "cand1",
        "source": "ac_core",
        "changed_fields": {},
        "candidate_metadata": {},
        "future_field_from_ac_core": {"x": 1},
    })
    back = Candidate.model_validate_json(raw)
    assert back.id == "cand1"
    # unknown fields preserved via model_extra
    assert back.model_extra is not None
    assert back.model_extra.get("future_field_from_ac_core") == {"x": 1}


def test_ac_core_prediction_round_trip():
    p = ACCorePrediction(
        id="pred1",
        candidate_id="cand1",
        hardware_id="h100_sxm",
        runtime="vllm",
        workload_id="long_chat",
        predicted_metrics={"throughput_tps": 1200.0},
        predicted_bottlenecks=["decode_kv"],
        risk_label="safe",
    )
    raw = p.model_dump_json()
    back = ACCorePrediction.model_validate_json(raw)
    assert back.predicted_metrics["throughput_tps"] == 1200.0
    assert back.predicted_bottlenecks == ["decode_kv"]


def test_experiment_plan_status_enum():
    plan = ExperimentPlan(
        id="plan1",
        name="decode_kv microbench",
        experiment_type="kernel_microbench",
        candidate_ids=["baseline", "gqa8"],
        decision_unblocked="whether reduced KV heads is serving-useful",
        uncertainty_target="decode KV bandwidth correction",
        estimated_cost={"gpu_hours": 0.5},
    )
    assert plan.status == "planned"
    with pytest.raises(Exception):
        ExperimentPlan(
            id="plan2",
            name="bad",
            experiment_type="not_a_valid_type",  # type: ignore[arg-type]
            candidate_ids=["x"],
            decision_unblocked="-",
            uncertainty_target="-",
        )


def test_measurement_round_trip():
    m = Measurement(
        id="meas1",
        candidate_id="cand1",
        experiment_id="plan1",
        measurement_type="kernel",
        metric_name="decode_kv_bandwidth_gbps",
        metric_value=850.3,
        metric_unit="GB/s",
        hardware_id="h100_sxm",
        runtime="vllm",
        workload_id="long_chat",
    )
    raw = m.model_dump_json()
    back = Measurement.model_validate_json(raw)
    assert back.metric_value == 850.3
    assert back.metric_unit == "GB/s"


def test_fitted_calibration_insufficient_data_path():
    fit = FittedCalibration(
        id="fit1",
        name="moe_x_ep",
        target="interaction",
        n_measurements=2,
        functional_form="ridge(degree=1)",
        coefficients={},
        uncertainty={},
        status="insufficient_data",
        needed_measurements=["moe_all_to_all@ep=8", "moe_all_to_all@ep=16"],
    )
    assert fit.status == "insufficient_data"
    assert "moe_all_to_all@ep=8" in fit.needed_measurements


def test_decision_state_defaults():
    ds = DecisionState(id="ds1", name="initial", candidate_ids=["a", "b"])
    assert ds.current_frontier == []
    assert ds.disagreements == []
    assert ds.recommended_next_experiments == []


def test_human_decision_record():
    hd = HumanDecision(
        id="h1",
        decision_type="approve_expensive_run",
        prompt="Run 5h MoE all-to-all bench at EP=16?",
        options=["approve", "skip", "downscope"],
    )
    assert hd.selected_option is None
    hd.selected_option = "approve"
    hd.resolved_at = "2026-06-16T00:00:00Z"
    assert hd.selected_option == "approve"
