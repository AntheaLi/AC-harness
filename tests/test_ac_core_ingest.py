"""AC-Core ingestor tests (§7)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from ac_harness.ingest import import_ac_core_run
from ac_harness.store import EvidenceStore


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "ac_core_outputs" / "llama_h100_long_chat"


@pytest.fixture()
def store(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    yield s
    s.close()


def test_import_creates_candidates_and_predictions(store):
    cand_ids = import_ac_core_run(str(FIXTURE), store)
    assert "llama_baseline" in cand_ids
    assert "llama_gqa8" in cand_ids
    assert "llama_gqa8_kvfp8" in cand_ids

    candidates = store.list_candidates()
    ids = {c.id for c in candidates}
    assert {"llama_baseline", "llama_gqa8", "llama_gqa4", "llama_gqa8_kvfp8"} <= ids

    preds = store.list_predictions()
    assert len(preds) == 4
    pred_by_cand = {p.candidate_id: p for p in preds}
    assert pred_by_cand["llama_gqa4"].risk_label == "risky"
    assert (
        pred_by_cand["llama_gqa8"].predicted_metrics["throughput_tps"] == 1320.0
    )


def test_provenance_points_back_to_source(store):
    import_ac_core_run(str(FIXTURE), store)
    c = store.get_candidate("llama_gqa8")
    assert c is not None
    # Provenance carries the source path to CandidateSet.json or DeltaReport.json.
    src = c.provenance.get("source_path", "")
    assert "ac_core_outputs/llama_h100_long_chat" in src
    assert src.endswith(".json")


def test_unknown_fields_are_tolerated(store):
    import_ac_core_run(str(FIXTURE), store)
    c = store.get_candidate("llama_gqa8_kvfp8")
    assert c is not None
    # The raw payload contained a "compiler_speculative_field" — the candidate
    # should ingest without error. We don't require we preserve it on
    # Candidate model_extra (it lives in candidate_metadata only if AC-Core
    # placed it there), only that the import didn't fail.
    assert c.changed_fields.get("n_kv_heads") == 8


def test_baseline_id_is_propagated_from_delta_report(store, tmp_path):
    # Build a minimal fixture with ONLY a DeltaReport (no CandidateSet).
    src = tmp_path / "fake_ac_core"
    src.mkdir()
    payload = {
        "schema_version": 1,
        "baseline_id": "fake_baseline",
        "deltas": [
            {"id": "d1", "delta": {"d_model": 4096}},
            {"id": "d2", "delta": {"d_model": 8192}},
        ],
    }
    (src / "DeltaReport.json").write_text(json.dumps(payload))

    cand_ids = import_ac_core_run(str(src), store)
    assert set(cand_ids) == {"d1", "d2"}
    d1 = store.get_candidate("d1")
    assert d1 is not None
    assert d1.baseline_id == "fake_baseline"
    assert d1.changed_fields == {"d_model": 4096}


def test_import_does_not_mutate_ac_core_files(store, tmp_path):
    # Copy fixture so we can hash the contents before/after import.
    work = tmp_path / "ac_core_copy"
    shutil.copytree(FIXTURE, work)

    pre = {p.name: p.read_bytes() for p in work.glob("*.json")}
    import_ac_core_run(str(work), store)
    post = {p.name: p.read_bytes() for p in work.glob("*.json")}
    assert pre == post


def test_missing_required_fields_in_prediction_are_skipped(store, tmp_path):
    src = tmp_path / "fake_ac_core"
    src.mkdir()
    cs = {
        "candidates": [{"id": "x", "changed_fields": {"a": 1}}],
    }
    pp = {
        "points": [
            {"candidate_id": "x"},  # missing hardware_id / runtime / workload_id
            {
                "candidate_id": "x",
                "hardware_id": "h100",
                "runtime": "vllm",
                "workload_id": "wl",
            },
        ],
    }
    (src / "CandidateSet.json").write_text(json.dumps(cs))
    (src / "PredictedPareto.json").write_text(json.dumps(pp))
    import_ac_core_run(str(src), store)
    preds = store.list_predictions()
    assert len(preds) == 1  # the malformed one was skipped
