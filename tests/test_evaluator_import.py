"""Evaluator plan_mode + import_mode tests (§11)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ac_harness.evaluator import EVALUATORS, copy_task, lm_eval_adapter, mqar, needle, perplexity
from ac_harness.schemas import Candidate
from ac_harness.store import EvidenceStore


@pytest.fixture()
def store(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    s.insert_candidate(Candidate(id="cA", source="ac_core"))
    s.insert_candidate(Candidate(id="cB", source="ac_core"))
    yield s
    s.close()


def test_perplexity_plan_and_import(store):
    plan = perplexity.plan_mode(
        candidate_ids=["cA", "cB"], dataset_path="/data/val.jsonl"
    )
    assert plan["evaluator"] == "perplexity"
    assert "val_loss" in plan["metrics"]
    ids = perplexity.import_mode(
        store,
        result_payload={
            "results": [
                {"candidate_id": "cA", "val_loss": 2.13, "perplexity": 8.4, "step": 1000, "seed": 0},
                {"candidate_id": "cB", "val_loss": 2.05, "step": 1000, "seed": 0},
            ]
        },
        source_path="/fake/val.json",
    )
    # cA → 2 metrics (val_loss + perplexity), cB → 1 (val_loss only)
    assert len(ids) == 3
    rows = store.query_measurements(metric_name="val_loss")
    assert {r.candidate_id for r in rows} == {"cA", "cB"}
    assert all(r.measurement_type == "eval" for r in rows)
    # Provenance source preserved
    assert rows[0].provenance.get("source_path") == "/fake/val.json"


def test_needle_records_context_and_depth(store):
    ids = needle.import_mode(
        store,
        result_payload={
            "results": [
                {"candidate_id": "cA", "needle_accuracy": 0.91, "context_length": 16384, "depth": 0.5},
                {"candidate_id": "cB", "needle_accuracy": 0.42, "context_length": 65536, "depth": 0.75},
            ]
        },
    )
    assert len(ids) == 2
    rows = store.query_measurements(metric_name="needle_accuracy")
    extras = {(r.candidate_id, r.extra.get("context_length"), r.extra.get("depth")) for r in rows}
    assert ("cA", 16384, 0.5) in extras
    assert ("cB", 65536, 0.75) in extras


def test_mqar_records_K_and_seq_length(store):
    ids = mqar.import_mode(
        store,
        result_payload={
            "results": [
                {"candidate_id": "cA", "mqar_accuracy": 0.88, "K": 32, "seq_length": 4096},
            ]
        },
    )
    assert ids
    rows = store.query_measurements(metric_name="mqar_accuracy")
    assert rows[0].extra["K"] == 32


def test_copy_task_records_seq_length(store):
    ids = copy_task.import_mode(
        store,
        result_payload={
            "results": [
                {"candidate_id": "cA", "copy_accuracy": 0.99, "seq_length": 1024},
            ]
        },
    )
    assert ids
    assert store.query_measurements(metric_name="copy_accuracy")[0].extra["seq_length"] == 1024


def test_lm_eval_adapter_expands_tasks(store):
    ids = lm_eval_adapter.import_mode(
        store,
        result_payload={
            "results": [
                {
                    "candidate_id": "cA",
                    "task_scores": {"arc_challenge": 0.55, "hellaswag": 0.72},
                }
            ]
        },
    )
    assert len(ids) == 2
    names = {m.metric_name for m in store.query_measurements(candidate_id="cA")}
    assert "lmeval_arc_challenge" in names
    assert "lmeval_hellaswag" in names


def test_evaluators_registry_lookup():
    assert set(EVALUATORS) == {"perplexity", "needle", "mqar", "copy_task", "lm_eval"}
    # Every evaluator exposes plan_mode + import_mode callables.
    for name, mod in EVALUATORS.items():
        assert callable(getattr(mod, "plan_mode"))
        assert callable(getattr(mod, "import_mode"))
