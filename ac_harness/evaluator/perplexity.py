# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Perplexity / validation-loss evaluator (§11).

`plan_mode()` emits a tiny config dict that describes WHAT to evaluate
(dataset path, sequence length, batch size). `import_mode()` parses a
result file and writes Measurement rows with `measurement_type="eval"`
and `metric_name="val_loss"` (or "perplexity").

We never compute perplexity here. Real eval runs happen elsewhere and
report their results into the harness.
"""
from __future__ import annotations

from typing import Any

from ..schemas import Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance


METRIC_NAMES = ("val_loss", "perplexity")


def plan_mode(
    *,
    candidate_ids: list[str],
    dataset_path: str,
    seq_len: int = 4096,
    batch_size: int = 4,
    notes: str = "AC-Harness perplexity eval plan.",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluator": "perplexity",
        "candidate_ids": list(candidate_ids),
        "dataset_path": dataset_path,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "metrics": list(METRIC_NAMES),
        "notes": notes,
    }


def import_mode(
    store: EvidenceStore,
    *,
    result_payload: dict[str, Any],
    plan_id: str | None = None,
    source_path: str | None = None,
) -> list[str]:
    """Ingest a perplexity result payload of shape:
        {"results": [
            {"candidate_id": "...", "val_loss": 2.13, "perplexity": 8.4,
             "step": 1000, "seed": 0}
        ]}
    """
    out: list[str] = []
    for r in result_payload.get("results", []):
        cid = r.get("candidate_id")
        if cid is None:
            continue
        for name in METRIC_NAMES:
            if name not in r or r[name] is None:
                continue
            m = Measurement(
                id=new_id("meas"),
                candidate_id=cid,
                experiment_id=plan_id,
                measurement_type="eval",
                metric_name=name,
                metric_value=float(r[name]),
                step=r.get("step"),
                seed=r.get("seed"),
                extra={k: v for k, v in r.items() if k not in {"candidate_id", *METRIC_NAMES, "step", "seed"}},
                provenance=make_provenance(
                    source_path=source_path,
                    command="evaluator.perplexity.import_mode",
                ),
            )
            store.insert_measurement(m)
            out.append(m.id)
    return out
