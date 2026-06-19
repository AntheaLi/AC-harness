# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
SQLite-backed evidence store.

Append-only registry of harness records. Each entity gets one table; the
payload column is JSON (`text`) so we keep schema drift cheap and avoid
needing migrations for additive fields. A handful of indexable columns
(candidate_id, experiment_id, metric_name, etc.) are mirrored at the
row level for cheap filtering.

This is intentionally simple. The spec (§6) explicitly says: do not
build a massive knowledge graph in v1. A minimal append-only registry
with provenance is the right target.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from ..schemas import (
    ACCorePrediction,
    Candidate,
    DecisionState,
    ExperimentPlan,
    FittedCalibration,
    HumanDecision,
    Measurement,
)
from .provenance import make_provenance


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    baseline_id TEXT,
    ac_core_prediction_id TEXT,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    hardware_id TEXT NOT NULL,
    runtime TEXT NOT NULL,
    workload_id TEXT NOT NULL,
    risk_label TEXT,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS experiment_plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    status TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    experiment_id TEXT,
    measurement_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT,
    hardware_id TEXT,
    runtime TEXT,
    workload_id TEXT,
    step INTEGER,
    seed INTEGER,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fitted_calibrations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    target TEXT NOT NULL,
    status TEXT NOT NULL,
    n_measurements INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_states (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_decisions (
    id TEXT PRIMARY KEY,
    decision_type TEXT NOT NULL,
    selected_option TEXT,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_candidate ON predictions(candidate_id);
CREATE INDEX IF NOT EXISTS idx_measurements_candidate ON measurements(candidate_id);
CREATE INDEX IF NOT EXISTS idx_measurements_experiment ON measurements(experiment_id);
CREATE INDEX IF NOT EXISTS idx_measurements_metric ON measurements(metric_name);
CREATE INDEX IF NOT EXISTS idx_measurements_type ON measurements(measurement_type);
"""


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class EvidenceStore:
    """Append-only SQLite store with JSONL export and a `validate()` integrity
    check. Designed for single-process harness runs; not a server.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ----- context management ------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ----- inserts ----------------------------------------------------------

    def insert_candidate(self, candidate: Candidate) -> str:
        if not candidate.provenance:
            candidate.provenance = make_provenance(command="insert_candidate")
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO candidates
                   (id, source, baseline_id, ac_core_prediction_id,
                    schema_version, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.id,
                    candidate.source,
                    candidate.baseline_id,
                    candidate.ac_core_prediction_id,
                    candidate.schema_version,
                    candidate.created_at,
                    candidate.model_dump_json(),
                ),
            )
        return candidate.id

    def insert_prediction(self, prediction: ACCorePrediction) -> str:
        if not prediction.provenance:
            prediction.provenance = make_provenance(command="insert_prediction")
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO predictions
                   (id, candidate_id, hardware_id, runtime, workload_id,
                    risk_label, schema_version, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction.id,
                    prediction.candidate_id,
                    prediction.hardware_id,
                    prediction.runtime,
                    prediction.workload_id,
                    prediction.risk_label,
                    prediction.schema_version,
                    prediction.created_at,
                    prediction.model_dump_json(),
                ),
            )
        return prediction.id

    def insert_experiment_plan(self, plan: ExperimentPlan) -> str:
        if not plan.provenance:
            plan.provenance = make_provenance(command="insert_experiment_plan")
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO experiment_plans
                   (id, name, experiment_type, status,
                    schema_version, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan.id,
                    plan.name,
                    plan.experiment_type,
                    plan.status,
                    plan.schema_version,
                    plan.created_at,
                    plan.model_dump_json(),
                ),
            )
        return plan.id

    def insert_measurement(self, measurement: Measurement) -> str:
        if not measurement.provenance:
            measurement.provenance = make_provenance(command="insert_measurement")
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO measurements
                   (id, candidate_id, experiment_id, measurement_type,
                    metric_name, metric_value, metric_unit,
                    hardware_id, runtime, workload_id, step, seed,
                    schema_version, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    measurement.id,
                    measurement.candidate_id,
                    measurement.experiment_id,
                    measurement.measurement_type,
                    measurement.metric_name,
                    measurement.metric_value,
                    measurement.metric_unit,
                    measurement.hardware_id,
                    measurement.runtime,
                    measurement.workload_id,
                    measurement.step,
                    measurement.seed,
                    measurement.schema_version,
                    measurement.created_at,
                    measurement.model_dump_json(),
                ),
            )
        return measurement.id

    def insert_fitted_calibration(self, fit: FittedCalibration) -> str:
        if not fit.provenance:
            fit.provenance = make_provenance(command="insert_fitted_calibration")
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO fitted_calibrations
                   (id, name, target, status, n_measurements,
                    schema_version, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fit.id,
                    fit.name,
                    fit.target,
                    fit.status,
                    fit.n_measurements,
                    fit.schema_version,
                    fit.created_at,
                    fit.model_dump_json(),
                ),
            )
        return fit.id

    def insert_decision_state(self, ds: DecisionState) -> str:
        if not ds.provenance:
            ds.provenance = make_provenance(command="insert_decision_state")
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO decision_states
                   (id, name, schema_version, created_at, payload)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    ds.id,
                    ds.name,
                    ds.schema_version,
                    ds.created_at,
                    ds.model_dump_json(),
                ),
            )
        return ds.id

    def insert_human_decision(self, hd: HumanDecision) -> str:
        if not hd.provenance:
            hd.provenance = make_provenance(command="insert_human_decision")
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO human_decisions
                   (id, decision_type, selected_option,
                    schema_version, created_at, resolved_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    hd.id,
                    hd.decision_type,
                    hd.selected_option,
                    hd.schema_version,
                    hd.created_at,
                    hd.resolved_at,
                    hd.model_dump_json(),
                ),
            )
        return hd.id

    # ----- reads ------------------------------------------------------------

    def get_candidate(self, id: str) -> Candidate | None:
        row = self._conn.execute(
            "SELECT payload FROM candidates WHERE id = ?", (id,)
        ).fetchone()
        return Candidate.model_validate_json(row["payload"]) if row else None

    def get_prediction(self, id: str) -> ACCorePrediction | None:
        row = self._conn.execute(
            "SELECT payload FROM predictions WHERE id = ?", (id,)
        ).fetchone()
        return ACCorePrediction.model_validate_json(row["payload"]) if row else None

    def get_experiment_plan(self, id: str) -> ExperimentPlan | None:
        row = self._conn.execute(
            "SELECT payload FROM experiment_plans WHERE id = ?", (id,)
        ).fetchone()
        return ExperimentPlan.model_validate_json(row["payload"]) if row else None

    def get_fitted_calibration(self, id: str) -> FittedCalibration | None:
        row = self._conn.execute(
            "SELECT payload FROM fitted_calibrations WHERE id = ?", (id,)
        ).fetchone()
        return FittedCalibration.model_validate_json(row["payload"]) if row else None

    def get_decision_state(self, id: str) -> DecisionState | None:
        row = self._conn.execute(
            "SELECT payload FROM decision_states WHERE id = ?", (id,)
        ).fetchone()
        return DecisionState.model_validate_json(row["payload"]) if row else None

    def list_candidates(self) -> list[Candidate]:
        rows = self._conn.execute("SELECT payload FROM candidates").fetchall()
        return [Candidate.model_validate_json(r["payload"]) for r in rows]

    def list_predictions(self) -> list[ACCorePrediction]:
        rows = self._conn.execute("SELECT payload FROM predictions").fetchall()
        return [ACCorePrediction.model_validate_json(r["payload"]) for r in rows]

    def list_experiment_plans(self) -> list[ExperimentPlan]:
        rows = self._conn.execute("SELECT payload FROM experiment_plans").fetchall()
        return [ExperimentPlan.model_validate_json(r["payload"]) for r in rows]

    def list_fitted_calibrations(self) -> list[FittedCalibration]:
        rows = self._conn.execute("SELECT payload FROM fitted_calibrations").fetchall()
        return [FittedCalibration.model_validate_json(r["payload"]) for r in rows]

    def list_decision_states(self) -> list[DecisionState]:
        rows = self._conn.execute(
            "SELECT payload FROM decision_states ORDER BY created_at"
        ).fetchall()
        return [DecisionState.model_validate_json(r["payload"]) for r in rows]

    def query_measurements(
        self,
        *,
        candidate_id: str | None = None,
        experiment_id: str | None = None,
        measurement_type: str | None = None,
        metric_name: str | None = None,
        hardware_id: str | None = None,
        runtime: str | None = None,
        workload_id: str | None = None,
    ) -> list[Measurement]:
        clauses: list[str] = []
        params: list[Any] = []
        for col, val in (
            ("candidate_id", candidate_id),
            ("experiment_id", experiment_id),
            ("measurement_type", measurement_type),
            ("metric_name", metric_name),
            ("hardware_id", hardware_id),
            ("runtime", runtime),
            ("workload_id", workload_id),
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        sql = "SELECT payload FROM measurements"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(sql, params).fetchall()
        return [Measurement.model_validate_json(r["payload"]) for r in rows]

    # ----- export & validate ------------------------------------------------

    def export_jsonl(self, output_dir: str) -> None:
        """Export every table as a JSONL file (one record per line)."""
        os.makedirs(output_dir, exist_ok=True)
        tables = {
            "candidates.jsonl": self.list_candidates(),
            "predictions.jsonl": self.list_predictions(),
            "experiment_plans.jsonl": self.list_experiment_plans(),
            "measurements.jsonl": self.query_measurements(),
            "fitted_calibrations.jsonl": self.list_fitted_calibrations(),
            "decision_states.jsonl": self.list_decision_states(),
        }
        for fname, rows in tables.items():
            with open(os.path.join(output_dir, fname), "w") as f:
                for r in rows:
                    f.write(r.model_dump_json() + "\n")

    def validate(self) -> list[str]:
        """Return a list of integrity errors. Empty list means store is OK."""
        errors: list[str] = []

        # predictions reference real candidates
        rows = self._conn.execute(
            """SELECT p.id, p.candidate_id
               FROM predictions p
               LEFT JOIN candidates c ON c.id = p.candidate_id
               WHERE c.id IS NULL"""
        ).fetchall()
        for r in rows:
            errors.append(
                f"prediction {r['id']} references missing candidate {r['candidate_id']}"
            )

        # measurements reference real candidates
        rows = self._conn.execute(
            """SELECT m.id, m.candidate_id
               FROM measurements m
               LEFT JOIN candidates c ON c.id = m.candidate_id
               WHERE c.id IS NULL"""
        ).fetchall()
        for r in rows:
            errors.append(
                f"measurement {r['id']} references missing candidate {r['candidate_id']}"
            )

        # measurements with experiment_id reference real plans
        rows = self._conn.execute(
            """SELECT m.id, m.experiment_id
               FROM measurements m
               LEFT JOIN experiment_plans p ON p.id = m.experiment_id
               WHERE m.experiment_id IS NOT NULL AND p.id IS NULL"""
        ).fetchall()
        for r in rows:
            errors.append(
                f"measurement {r['id']} references missing experiment_plan {r['experiment_id']}"
            )

        return errors


def new_id(prefix: str) -> str:
    """Helper for ad-hoc record IDs."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
