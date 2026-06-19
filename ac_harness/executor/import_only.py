# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Import-only executor: turn a result JSON file into Measurement rows.

Expected result JSON shape (one of):

    1. {"measurements": [ {<Measurement fields>}, ... ]}
    2. {"benchmark_type": ..., "results": [ {<flat>}, ... ]}
    3. {"results": [ {"candidate_id": ..., "metrics": {"name": val, ...}}, ... ]}

The shape is intentionally permissive because external runners differ.
If the file doesn't match any of these shapes, ImportError is raised
with a hint.

We never invent values; if a metric is missing from the source file it
stays missing in the store.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ..schemas import Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance


def import_results(
    store: EvidenceStore,
    *,
    result_path: str,
    plan_id: str | None = None,
) -> list[str]:
    if not os.path.exists(result_path):
        raise FileNotFoundError(result_path)
    with open(result_path) as f:
        payload = json.load(f)

    rows: list[dict[str, Any]] = _normalize(payload)
    if not rows:
        raise ImportError(
            f"No measurements parsed from {result_path}; expected one of "
            "{measurements:[...]}, {results:[...]}, or per-candidate metrics dict."
        )

    plan = store.get_experiment_plan(plan_id) if plan_id else None
    inferred_type = _infer_type_from_payload(payload)
    common = {
        "hardware_id": payload.get("hardware_id"),
        "runtime": payload.get("runtime"),
        "workload_id": payload.get("workload_id") or payload.get("benchmark_type"),
        "experiment_id": plan_id,
    }

    inserted: list[str] = []
    for r in rows:
        m = Measurement(
            id=r.get("id") or new_id("meas"),
            candidate_id=r["candidate_id"],
            experiment_id=r.get("experiment_id") or common["experiment_id"],
            measurement_type=r.get("measurement_type") or inferred_type,
            metric_name=r["metric_name"],
            metric_value=float(r["metric_value"]),
            metric_unit=r.get("metric_unit"),
            hardware_id=r.get("hardware_id") or common["hardware_id"],
            runtime=r.get("runtime") or common["runtime"],
            workload_id=r.get("workload_id") or common["workload_id"],
            step=r.get("step"),
            seed=r.get("seed"),
            extra=r.get("extra", {}),
            provenance=make_provenance(
                source_path=result_path,
                command="executor.import_only.import_results",
                extra={"plan_id": plan_id},
            ),
        )
        store.insert_measurement(m)
        inserted.append(m.id)

    if plan is not None and plan.status in {"planned", "approved", "running"}:
        from .status import set_status

        # Move directly to completed on a successful import.
        if plan.status == "running":
            set_status(store, plan.id, "completed")
        elif plan.status in {"planned", "approved"}:
            set_status(store, plan.id, "running")
            set_status(store, plan.id, "completed")

    return inserted


def _infer_type_from_payload(payload: dict[str, Any]) -> str:
    bt = payload.get("benchmark_type") or payload.get("type")
    if bt is None:
        return "manual_import"
    bt = str(bt)
    if bt in {"long_chat", "short_chat", "rag_long_context", "code_agent",
              "decode_heavy", "prefill_heavy", "batch_offline"}:
        return "serving"
    if bt in {"perplexity", "needle", "mqar", "copy_task", "lm_eval"}:
        return "eval"
    if bt in {"small_training_ablation", "training"}:
        return "training"
    return "kernel"


def _normalize(payload: Any) -> list[dict[str, Any]]:
    # Shape 1: explicit measurements list of full Measurement-shaped dicts.
    if isinstance(payload, dict) and isinstance(payload.get("measurements"), list):
        return [dict(x) for x in payload["measurements"]]

    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        for r in payload["results"]:
            if not isinstance(r, dict):
                continue
            cid = r.get("candidate_id")
            if cid is None:
                continue
            # Shape 3: {"candidate_id": ..., "metrics": {...}}
            if isinstance(r.get("metrics"), dict):
                for name, val in r["metrics"].items():
                    if val is None:
                        continue
                    rows.append({
                        "candidate_id": cid,
                        "metric_name": name,
                        "metric_value": val,
                        "metric_unit": r.get("units", {}).get(name) if isinstance(r.get("units"), dict) else None,
                        "step": r.get("step"),
                        "seed": r.get("seed"),
                        "extra": r.get("extra", {}),
                    })
            # Shape 2: flat row already shaped as a Measurement.
            elif "metric_name" in r and "metric_value" in r:
                rows.append(dict(r))
    return rows
