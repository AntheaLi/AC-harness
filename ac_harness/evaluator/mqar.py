# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
MQAR (Multi-Query Associative Recall) evaluator (§11).
"""
from __future__ import annotations

from typing import Any

from ..schemas import Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance


METRIC_NAME = "mqar_accuracy"


def plan_mode(
    *,
    candidate_ids: list[str],
    K_values: list[int] | None = None,
    seq_lengths: list[int] | None = None,
    notes: str = "AC-Harness MQAR eval plan.",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluator": "mqar",
        "candidate_ids": list(candidate_ids),
        "K_values": list(K_values or [8, 32, 128]),
        "seq_lengths": list(seq_lengths or [1024, 4096, 16384]),
        "metrics": [METRIC_NAME],
        "notes": notes,
    }


def import_mode(
    store: EvidenceStore,
    *,
    result_payload: dict[str, Any],
    plan_id: str | None = None,
    source_path: str | None = None,
) -> list[str]:
    out: list[str] = []
    for r in result_payload.get("results", []):
        cid = r.get("candidate_id")
        if cid is None or "mqar_accuracy" not in r:
            continue
        m = Measurement(
            id=new_id("meas"),
            candidate_id=cid,
            experiment_id=plan_id,
            measurement_type="eval",
            metric_name=METRIC_NAME,
            metric_value=float(r["mqar_accuracy"]),
            extra={"K": r.get("K"), "seq_length": r.get("seq_length")},
            seed=r.get("seed"),
            provenance=make_provenance(
                source_path=source_path,
                command="evaluator.mqar.import_mode",
            ),
        )
        store.insert_measurement(m)
        out.append(m.id)
    return out
