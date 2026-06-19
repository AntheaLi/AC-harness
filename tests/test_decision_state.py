"""Decision-state, frontier, disagreement, and report tests (§13)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ac_harness.decision import (
    build_decision_state,
    compute_frontier,
    detect_disagreements,
    generate_report,
)
from ac_harness.executor import import_results
from ac_harness.ingest import import_ac_core_run
from ac_harness.schemas import Measurement
from ac_harness.store import EvidenceStore, new_id


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "ac_core_outputs" / "llama_h100_long_chat"


@pytest.fixture()
def store_with_fixture(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    import_ac_core_run(str(FIXTURE), s)
    yield s
    s.close()


def test_initial_state_has_no_measurements(store_with_fixture):
    ds = build_decision_state(store_with_fixture)
    assert "llama_baseline" in ds.candidate_ids
    assert ds.observed_summary["total_measurements"] == 0
    # No buckets measured → every candidate is "prediction_only".
    expected = {"llama_baseline", "llama_gqa8", "llama_gqa4", "llama_gqa8_kvfp8"}
    assert set(ds.uncertainty_summary["prediction_only_candidates"]) >= expected


def test_state_after_kernel_measurement(store_with_fixture):
    s = store_with_fixture
    s.insert_measurement(
        Measurement(
            id=new_id("meas"),
            candidate_id="llama_gqa8",
            measurement_type="kernel",
            metric_name="decode_kv_bandwidth_gbps",
            metric_value=820.0,
            metric_unit="GB/s",
            hardware_id="h100_sxm",
            runtime="vllm",
        )
    )
    ds = build_decision_state(s)
    summary = ds.observed_summary["candidates"]["llama_gqa8"]
    assert summary["n_measurements"] == 1
    assert "kernel_bandwidth" in summary["buckets_present"]
    # gqa8 still has missing quality buckets
    missing = ds.uncertainty_summary["missing_buckets_by_candidate"]["llama_gqa8"]
    assert "quality_loss" in missing
    assert "throughput" in missing


def test_risky_unmeasured_flag(store_with_fixture):
    ds = build_decision_state(store_with_fixture)
    # gqa4 was tagged risky in fixture; gqa8_kvfp8 was tagged research.
    risky = set(ds.uncertainty_summary["risky_unmeasured_candidates"])
    assert {"llama_gqa4", "llama_gqa8_kvfp8"} <= risky


def test_candidate_filter(store_with_fixture):
    ds = build_decision_state(
        store_with_fixture, candidate_ids=["llama_gqa8", "llama_baseline"]
    )
    assert set(ds.candidate_ids) == {"llama_gqa8", "llama_baseline"}


# --------------------------------------------------------------------------- #
# Phase 8 — frontier / disagreement / report
# --------------------------------------------------------------------------- #


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_DECODE_RESULTS = REPO_ROOT / "examples" / "imported_results" / "decode_kv_fake.json"


def test_frontier_prediction_only_when_no_measurements(store_with_fixture):
    report = compute_frontier(store_with_fixture)
    assert report.kind == "prediction_only"
    # gqa8_kvfp8 has the highest predicted throughput in the fixture.
    assert "llama_gqa8_kvfp8" in report.prediction_only_supported


def test_frontier_observed_after_results_imported(store_with_fixture):
    import_results(store_with_fixture, result_path=str(FAKE_DECODE_RESULTS))
    report = compute_frontier(store_with_fixture)
    assert report.kind in {"observed", "mixed"}
    # Highest observed throughput in fake results is llama_gqa4.
    assert "llama_gqa4" in report.observed_supported


def test_disagreement_predicted_serving_gain_not_observed(tmp_path):
    """Construct a store where AC-Core predicted a big throughput win that the
    measurements contradict.
    """
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    import_ac_core_run(str(REPO_ROOT / "examples" / "ac_core_outputs" / "llama_h100_long_chat"), s)
    # Baseline observed = 1000, gqa8 observed = 990 (no gain) despite predicted 1320.
    for cid, val in [("llama_baseline", 1000.0), ("llama_gqa8", 990.0)]:
        s.insert_measurement(
            Measurement(
                id=new_id("meas"),
                candidate_id=cid,
                measurement_type="kernel",
                metric_name="throughput_tps",
                metric_value=val,
                hardware_id="h100_sxm",
                runtime="vllm",
            )
        )
    issues = detect_disagreements(s)
    rules = {i["rule"] for i in issues}
    assert "predicted_serving_gain_not_observed" in rules
    s.close()


def test_report_smoke_writes_markdown(store_with_fixture, tmp_path):
    import_results(store_with_fixture, result_path=str(FAKE_DECODE_RESULTS))
    out = tmp_path / "report.md"
    md = generate_report(store_with_fixture, out_path=str(out))
    assert out.exists()
    text = out.read_text()
    # Every required section header is present.
    for sec in [
        "## 1. Research question",
        "## 2. Candidate set",
        "## 3. Current evidence table",
        "## 4. Current supported frontier",
        "## 5. Uncertain candidates",
        "## 6. Disagreements / surprises",
        "## 7. Fitted calibration / residuals available",
        "## 8. Recommended next experiments",
        "## 9. Human decision points",
        "## 10. Export files for AC-Core",
    ]:
        assert sec in text, sec
    assert "llama_baseline" in text
    # The report must not declare a winner (only the disclaimer mentions
    # "optimal architecture" as the thing it doesn't claim).
    forbidden = [
        "this is the optimal",
        "recommended architecture is",
        "winning candidate is",
    ]
    for phrase in forbidden:
        assert phrase not in text.lower(), phrase
