"""Benchmark plan generator tests (§9)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ac_harness.benchmarks import (
    KERNEL_CATEGORIES,
    SERVING_CATEGORIES,
    build_kernel_plan,
    build_serving_plan,
    default_metrics,
    default_shapes,
    write_kernel_plan,
    write_serving_plan,
)


def test_kernel_plan_shape_matches_spec_9_3(tmp_path):
    plan = build_kernel_plan(
        benchmark_type="attention_decode",
        candidate_ids=["llama_baseline", "llama_gqa8"],
        hardware_id="h100_sxm",
        runtime="vllm",
    )
    # Required keys from §9.3
    for key in (
        "schema_version",
        "benchmark_type",
        "candidate_ids",
        "hardware_id",
        "runtime",
        "shapes",
        "metrics",
        "notes",
    ):
        assert key in plan

    assert plan["benchmark_type"] == "attention_decode"
    assert plan["candidate_ids"] == ["llama_baseline", "llama_gqa8"]
    assert plan["shapes"]
    assert "latency_ms" in plan["metrics"]

    out = tmp_path / "kernel_plan.json"
    write_kernel_plan(plan, str(out))
    back = json.loads(out.read_text())
    assert back == plan


def test_kernel_plan_unknown_type_rejected():
    with pytest.raises(ValueError):
        build_kernel_plan(
            benchmark_type="not_a_real_kernel",
            candidate_ids=["x"],
            hardware_id="h100",
            runtime="vllm",
        )


def test_serving_plan_shape(tmp_path):
    plan = build_serving_plan(
        benchmark_type="long_chat",
        candidate_ids=["llama_baseline", "llama_gqa8"],
        hardware_id="h100_sxm",
        runtime="vllm",
    )
    assert plan["benchmark_type"] == "long_chat"
    assert plan["mix"]["prompt_tokens"] == 4096
    assert "serving_throughput_tps" in plan["metrics"]

    out = tmp_path / "serving_plan.json"
    write_serving_plan(plan, str(out))
    back = json.loads(out.read_text())
    assert back == plan


def test_all_kernel_categories_have_defaults():
    # Every advertised kernel category should at least be plannable; some
    # may have empty default grids but the metric list must exist.
    for k in KERNEL_CATEGORIES:
        # default_shapes may be empty for now (e.g. state_scan tier-1 stub),
        # but the metric list should not.
        assert isinstance(default_shapes(k), list)
        assert default_metrics(k), f"missing default metrics for {k}"


def test_all_serving_categories_have_a_mix():
    for w in SERVING_CATEGORIES:
        plan = build_serving_plan(
            benchmark_type=w,
            candidate_ids=["x"],
            hardware_id="h100",
            runtime="vllm",
        )
        assert plan["mix"], w
