"""End-to-end §16 workflow smoke test.

Runs the CLI commands programmatically against the fixture, asserts each
step produces the expected files, and confirms the final
DecisionStateReport.md references the imported result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ac_harness.cli import app


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_AC = REPO_ROOT / "examples" / "ac_core_outputs" / "llama_h100_long_chat"
FIXTURE_RESULT = REPO_ROOT / "examples" / "imported_results" / "decode_kv_fake.json"


@pytest.fixture()
def runner():
    return CliRunner()


def _run(runner, args: list[str]):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def test_section_16_end_to_end(runner, tmp_path):
    store = tmp_path / "demo.sqlite"
    next_md = tmp_path / "demo" / "next_experiment.md"
    plans_dir = next_md.parent / "plans"
    materialized = tmp_path / "demo" / "materialized"
    report = tmp_path / "demo" / "DecisionStateReport.md"

    # 1. init
    _run(runner, ["init", "--store", str(store)])
    assert store.exists()

    # 2. ingest AC-Core
    _run(runner, [
        "ingest-ac-core",
        "--input", str(FIXTURE_AC),
        "--store", str(store),
    ])

    # 3. plan-next (small budget)
    _run(runner, [
        "plan-next",
        "--store", str(store),
        "--budget", "small",
        "--out", str(next_md),
    ])
    assert next_md.exists()
    plan_files = list(plans_dir.glob("*.json"))
    assert plan_files, "plan-next did not emit any per-plan JSON files"

    # 4. materialize the first kernel plan in dry_run mode
    _run(runner, [
        "materialize",
        "--plan", str(plan_files[0]),
        "--mode", "dry_run",
        "--out", str(materialized),
    ])
    for f in ("plan.json", "run.py", "run.sh"):
        assert (materialized / f).exists(), f

    # 5. import results
    _run(runner, [
        "import-results",
        "--plan-id", plan_files[0].stem,
        "--input", str(FIXTURE_RESULT),
        "--store", str(store),
    ])

    # 6. fit calibration
    cal_out = tmp_path / "demo" / "feedback" / "calibration"
    _run(runner, [
        "fit-calibration",
        "--target", "throughput",
        "--store", str(store),
        "--out", str(cal_out),
        "--hardware-id", "h100_sxm",
        "--runtime", "vllm",
    ])
    files = list(cal_out.glob("*.json"))
    assert files

    # 7. decision report
    _run(runner, [
        "decision-report",
        "--store", str(store),
        "--out", str(report),
    ])
    text = report.read_text()
    assert "AC-Harness Decision State Report" in text
    assert "llama_baseline" in text
    # The imported result should have populated the evidence table.
    assert "decode_kv_bandwidth_gbps" in text or "throughput_tps" in text

    # 8. export-ac-core-feedback
    feedback = tmp_path / "demo" / "feedback"
    _run(runner, [
        "export-ac-core-feedback",
        "--store", str(store),
        "--out", str(feedback),
    ])
    assert (feedback / "calibration").exists()
    assert (feedback / "quality_residuals").exists()
    assert (feedback / "interaction_terms").exists()
