"""Executor materialize + import tests (§10)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ac_harness.executor import emit_slurm, import_results, materialize, set_status
from ac_harness.ingest import import_ac_core_run
from ac_harness.planner import plan_next_experiments
from ac_harness.schemas import ExperimentPlan
from ac_harness.store import EvidenceStore


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_AC = REPO_ROOT / "examples" / "ac_core_outputs" / "llama_h100_long_chat"
FIXTURE_RESULT = REPO_ROOT / "examples" / "imported_results" / "decode_kv_fake.json"


@pytest.fixture()
def populated(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    import_ac_core_run(str(FIXTURE_AC), s)
    plans = plan_next_experiments(s, budget="small")
    yield s, plans, tmp_path
    s.close()


def test_materialize_creates_runnable_artifacts(populated):
    s, plans, tmp_path = populated
    plan = plans[0]
    out = tmp_path / "materialized"
    files = materialize(plan, str(out))
    names = {os.path.basename(p) for p in files}
    assert names == {"plan.json", "run.py", "run.sh"}
    assert os.access(os.path.join(out, "run.sh"), os.X_OK)
    # run.py is a valid python file (parses)
    with open(os.path.join(out, "run.py")) as f:
        compile(f.read(), "run.py", "exec")
    # plan.json is the full ExperimentPlan
    plan_payload = json.loads((out / "plan.json").read_text())
    assert plan_payload["id"] == plan.id
    assert plan_payload["experiment_type"] == plan.experiment_type


def test_emit_slurm_writes_sbatch(populated):
    _, plans, tmp_path = populated
    plan = plans[0]
    out = tmp_path / "materialized"
    path = emit_slurm(plan, str(out))
    text = Path(path).read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "#SBATCH" in text
    assert plan.id in text


def test_import_results_from_fixture(populated):
    s, plans, _ = populated
    plan = plans[0]
    # Pretend this plan is the decode_kv plan the result file was generated for.
    n_before = len(s.query_measurements())
    ids = import_results(s, result_path=str(FIXTURE_RESULT), plan_id=plan.id)
    assert len(ids) >= 9  # 3 candidates × 3 metrics
    rows = s.query_measurements(experiment_id=plan.id)
    assert rows
    # Each row carries the source file in provenance.
    assert all(
        m.provenance.get("source_path", "").endswith("decode_kv_fake.json")
        for m in rows
    )
    # Plan moved to completed.
    refreshed = s.get_experiment_plan(plan.id)
    assert refreshed.status == "completed"
    n_after = len(s.query_measurements())
    assert n_after - n_before >= 9


def test_import_results_unknown_shape_errors(populated, tmp_path):
    s, _, _ = populated
    bad = tmp_path / "bad.json"
    bad.write_text('{"unexpected": true}')
    with pytest.raises(ImportError):
        import_results(s, result_path=str(bad))


def test_status_transition_rules(populated):
    s, plans, _ = populated
    plan = plans[0]
    set_status(s, plan.id, "approved")
    refreshed = s.get_experiment_plan(plan.id)
    assert refreshed.status == "approved"
    with pytest.raises(ValueError):
        # cannot go approved -> completed
        set_status(s, plan.id, "completed")
