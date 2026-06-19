"""Throughput calibration fitter tests (§12.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ac_harness.executor import import_results
from ac_harness.fitter import (
    export_measured_calibration_json,
    fit_throughput_calibration,
)
from ac_harness.ingest import import_ac_core_run
from ac_harness.store import EvidenceStore


REPO_ROOT = Path(__file__).resolve().parents[1]
AC = REPO_ROOT / "examples" / "ac_core_outputs" / "llama_h100_long_chat"
DECODE = REPO_ROOT / "examples" / "imported_results" / "decode_kv_fake.json"


@pytest.fixture()
def store_with_kernels(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    import_ac_core_run(str(AC), s)
    import_results(s, result_path=str(DECODE))
    yield s
    s.close()


def test_throughput_calibration_fit_smoke(store_with_kernels):
    fits = fit_throughput_calibration(store_with_kernels)
    # We measured ≥3 kernel rows per metric in the fixture (3 candidates).
    by_name = {f.name: f for f in fits}
    assert "throughput_decode_kv_bandwidth_gbps" in by_name
    fit = by_name["throughput_decode_kv_bandwidth_gbps"]
    assert fit.status == "draft"
    assert fit.coefficients["intercept"] > 0
    assert 0.0 <= fit.coefficients["achieved_efficiency_mean"] <= 1.0
    assert fit.holdout_error and fit.holdout_error.get("loo_mae") is not None
    assert fit.uncertainty["intercept_sigma"] >= 0.0


def test_insufficient_data_returns_status(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    import_ac_core_run(str(AC), s)
    fits = fit_throughput_calibration(s, min_points=10)
    # We have at most 3 candidates' worth of kernel data → insufficient.
    statuses = {f.status for f in fits}
    # When no kernel rows exist yet, fits list is empty — both outcomes acceptable.
    assert statuses in ({"insufficient_data"}, set())
    s.close()


def test_export_measured_calibration_json(store_with_kernels, tmp_path):
    fit_throughput_calibration(store_with_kernels, hardware_id="h100_sxm", runtime="vllm")
    out_dir = tmp_path / "feedback"
    paths = export_measured_calibration_json(
        store_with_kernels,
        out_dir=str(out_dir),
        hardware_id="h100_sxm",
        runtime="vllm",
    )
    assert paths
    payload = json.loads(Path(paths[0]).read_text())
    assert payload["hardware_id"] == "h100_sxm"
    assert payload["runtime"] == "vllm"
    assert payload["fits"]
