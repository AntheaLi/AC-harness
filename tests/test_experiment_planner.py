"""Experiment planner tests (§8 acceptance behavior)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ac_harness.ingest import import_ac_core_run
from ac_harness.planner import plan_next_experiments, render_plan_markdown
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


def test_missing_throughput_prioritizes_kernel_or_serving(store_with_fixture):
    plans = plan_next_experiments(store_with_fixture, budget="small")
    assert plans, "planner returned no plans"
    top_types = {p.experiment_type for p in plans[:3]}
    assert top_types & {"kernel_microbench", "serving_bench"}, top_types


def test_risky_unmeasured_yields_quality_or_ablation(store_with_fixture):
    plans = plan_next_experiments(store_with_fixture, budget="medium")
    types = {p.experiment_type for p in plans}
    # gqa4 / gqa8_kvfp8 are risky/research with no quality data → should appear.
    assert (
        "small_training_ablation" in types or "quality_eval" in types
    ), types


def test_all_measured_recommends_fit_report(tmp_path):
    s = EvidenceStore(str(tmp_path / "store.sqlite"))
    import_ac_core_run(str(FIXTURE), s)
    # Backfill measurements covering every key bucket for every candidate so
    # `missing_buckets_by_candidate[*]` becomes empty.
    metrics = [
        ("throughput_tps", "tok/s"),
        ("decode_kv_bandwidth_gbps", "GB/s"),
        ("val_loss", None),
        ("needle_accuracy", None),
    ]
    for cand in s.list_candidates():
        for name, unit in metrics:
            s.insert_measurement(
                Measurement(
                    id=new_id("meas"),
                    candidate_id=cand.id,
                    measurement_type="kernel" if "gbps" in name else "eval",
                    metric_name=name,
                    metric_value=1.0,
                    metric_unit=unit,
                    hardware_id="h100_sxm",
                    runtime="vllm",
                    workload_id="long_chat",
                )
            )
    plans = plan_next_experiments(s, budget="small")
    assert plans
    top = plans[0]
    assert top.experiment_type == "import_external_result"
    assert top.config.get("mode") == "fit_and_report"
    s.close()


def test_budget_filters_expensive_plans(store_with_fixture):
    small = {p.experiment_type for p in plan_next_experiments(store_with_fixture, budget="small")}
    # small budget should not propose tier-2 experiments
    assert "small_training_ablation" not in small
    assert "quality_eval" not in small


def test_render_plan_markdown_smoke(store_with_fixture):
    plans = plan_next_experiments(store_with_fixture, budget="medium")
    md = render_plan_markdown(plans)
    assert "Next experiments" in md
    assert plans[0].name in md
