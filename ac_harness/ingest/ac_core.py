# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
AC-Core ingestor (§7).

Reads AC-Core output files and creates Candidate + ACCorePrediction
records in the evidence store, without coupling the harness to AC-Core
internals.

Supported input files (under a directory):
    CandidateSet.json
    DeltaReport.json
    PredictedPareto.json
    CalibrationRequest.json

Behavior contract:
1. Create Candidate records.
2. Create ACCorePrediction records.
3. Preserve AC-Core IDs when present.
4. Store raw AC-Core payload path in provenance.
5. Gracefully handle unknown fields.
6. Never mutate AC-Core files.

The harness does NOT validate AC-Core's compiler logic. Unknown fields
are preserved via the pydantic `extra="allow"` setting on schemas.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ..schemas import ACCorePrediction, Candidate
from ..store import EvidenceStore
from ..store.provenance import make_provenance

KNOWN_FILES = (
    "CandidateSet.json",
    "DeltaReport.json",
    "PredictedPareto.json",
    "CalibrationRequest.json",
)


def _read_json(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _coerce_candidate(
    raw: dict[str, Any], source_path: str, default_baseline: str | None = None
) -> Candidate:
    """Map an AC-Core candidate dict to our Candidate schema, tolerant of
    field-name drift. AC-Core may use 'candidate_id' or 'id', 'delta' or
    'changed_fields', etc.
    """
    cid = raw.get("id") or raw.get("candidate_id")
    if not cid:
        # Generate a stable-ish id from the dict if AC-Core didn't provide one.
        cid = f"cand_{abs(hash(json.dumps(raw, sort_keys=True))) % (10**10)}"

    changed = (
        raw.get("changed_fields")
        or raw.get("delta")
        or raw.get("delta_fields")
        or {}
    )

    metadata = raw.get("candidate_metadata") or raw.get("metadata") or {}

    baseline = raw.get("baseline_id") or default_baseline
    arch_path = raw.get("architecture_config_path") or raw.get("config_path")
    pred_id = raw.get("ac_core_prediction_id") or raw.get("prediction_id")

    prov = make_provenance(
        source_path=source_path,
        command="ingest.ac_core.import_ac_core_run",
        extra={"raw": raw},
    )

    return Candidate(
        id=str(cid),
        source="ac_core",
        architecture_config_path=arch_path,
        baseline_id=baseline,
        changed_fields=changed if isinstance(changed, dict) else {"_raw": changed},
        candidate_metadata=metadata if isinstance(metadata, dict) else {},
        ac_core_prediction_id=str(pred_id) if pred_id else None,
        provenance=prov,
    )


def _coerce_prediction(
    raw: dict[str, Any], source_path: str
) -> ACCorePrediction | None:
    """Map an AC-Core predicted point to our ACCorePrediction schema.

    Returns None if the raw entry lacks the minimum required fields
    (candidate_id + hardware_id + runtime + workload_id) — those are
    structural to a meaningful prediction record.
    """
    pid = raw.get("id") or raw.get("prediction_id")
    candidate_id = raw.get("candidate_id") or raw.get("id")
    hardware_id = raw.get("hardware_id") or raw.get("hardware")
    runtime = raw.get("runtime")
    workload_id = raw.get("workload_id") or raw.get("workload")

    if not (candidate_id and hardware_id and runtime and workload_id):
        return None

    if not pid:
        pid = f"pred_{candidate_id}_{hardware_id}_{workload_id}"

    metrics = raw.get("predicted_metrics") or raw.get("metrics") or {}
    bottlenecks = raw.get("predicted_bottlenecks") or raw.get("bottlenecks") or []
    risk = raw.get("risk_label") or raw.get("risk")
    notes = raw.get("compiler_notes") or raw.get("notes")

    prov = make_provenance(
        source_path=source_path,
        command="ingest.ac_core.import_ac_core_run",
        extra={"raw": raw},
    )

    return ACCorePrediction(
        id=str(pid),
        candidate_id=str(candidate_id),
        hardware_id=str(hardware_id),
        runtime=str(runtime),
        workload_id=str(workload_id),
        predicted_metrics=metrics if isinstance(metrics, dict) else {},
        predicted_bottlenecks=list(bottlenecks) if isinstance(bottlenecks, list) else [],
        risk_label=str(risk) if risk else None,
        compiler_notes=str(notes) if notes else None,
        provenance=prov,
    )


def import_ac_core_run(path: str, store: EvidenceStore) -> list[str]:
    """Import AC-Core candidates and predictions into the evidence store.

    Returns the list of candidate IDs inserted (or already present).
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not a directory: {path}")

    # Track which candidates we've created so predictions can wire to them.
    candidate_ids: list[str] = []

    # ---- 1. CandidateSet.json -----------------------------------------------
    cs_path = os.path.join(path, "CandidateSet.json")
    cs_raw = _read_json(cs_path)
    default_baseline = None

    if cs_raw is not None:
        default_baseline = cs_raw.get("baseline_id") or cs_raw.get("baseline")
        for raw in cs_raw.get("candidates", cs_raw if isinstance(cs_raw, list) else []):
            cand = _coerce_candidate(raw, cs_path, default_baseline=default_baseline)
            store.insert_candidate(cand)
            candidate_ids.append(cand.id)

    # ---- 2. DeltaReport.json (acts like a CandidateSet of deltas) -----------
    dr_path = os.path.join(path, "DeltaReport.json")
    dr_raw = _read_json(dr_path)
    if dr_raw is not None:
        baseline_id = dr_raw.get("baseline_id") or default_baseline
        for raw in dr_raw.get("deltas", []):
            raw = dict(raw)
            # A "delta" maps to a Candidate with baseline_id set.
            raw.setdefault("baseline_id", baseline_id)
            cand = _coerce_candidate(raw, dr_path, default_baseline=baseline_id)
            store.insert_candidate(cand)
            candidate_ids.append(cand.id)

    # ---- 3. PredictedPareto.json --------------------------------------------
    pp_path = os.path.join(path, "PredictedPareto.json")
    pp_raw = _read_json(pp_path)
    if pp_raw is not None:
        points = pp_raw.get("points") or pp_raw.get("frontier") or []
        for raw in points:
            # Some payloads include a candidate sub-dict; if so, ingest it too.
            if "candidate" in raw and isinstance(raw["candidate"], dict):
                cand = _coerce_candidate(
                    raw["candidate"], pp_path, default_baseline=default_baseline
                )
                store.insert_candidate(cand)
                candidate_ids.append(cand.id)
                # Make sure the prediction wires to this candidate id.
                raw = dict(raw)
                raw.setdefault("candidate_id", cand.id)
            pred = _coerce_prediction(raw, pp_path)
            if pred is not None:
                store.insert_prediction(pred)

    # ---- 4. CalibrationRequest.json -----------------------------------------
    # We don't have a dedicated schema for "calibration request" in v1; we
    # log its provenance via a NO-OP read so users can confirm presence in
    # downstream tooling. Future work: add a CalibrationRequest schema.
    cr_path = os.path.join(path, "CalibrationRequest.json")
    _ = _read_json(cr_path)  # touched, not stored

    # Dedup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for cid in candidate_ids:
        if cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out
