# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Core schemas for AC-Harness.

All schemas are pydantic v2 BaseModels, versioned via `schema_version`,
and JSON round-trippable. Each top-level entity also carries a `provenance`
dict (or a dedicated provenance field) capturing the source of the record.

The spec (§5) defines the canonical structure of each model. Field
additions are allowed but must be additive and preserve schema_version
semantics: a non-breaking change keeps the version; a breaking change
bumps it.

These schemas describe OBSERVED EVIDENCE, not compiler outputs.
ACCorePrediction is read-only from the harness's perspective — it is
imported from AC-Core, never generated here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Base(BaseModel):
    """Common config: allow unknown fields without failing (tolerance for
    AC-Core schema drift) but expose them via model_extra."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=False,
    )


# --------------------------------------------------------------------------- #
# §5.1 Candidate
# --------------------------------------------------------------------------- #

class Candidate(_Base):
    schema_version: int = 1
    id: str
    source: Literal["ac_core", "manual", "external"]
    architecture_config_path: Optional[str] = None
    baseline_id: Optional[str] = None
    changed_fields: dict[str, Any] = Field(default_factory=dict)
    candidate_metadata: dict[str, Any] = Field(default_factory=dict)
    ac_core_prediction_id: Optional[str] = None

    # Provenance is harness-side bookkeeping, not part of the AC-Core contract.
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)


# --------------------------------------------------------------------------- #
# §5.2 AC-Core prediction (imported, not generated)
# --------------------------------------------------------------------------- #

class ACCorePrediction(_Base):
    schema_version: int = 1
    id: str
    candidate_id: str
    hardware_id: str
    runtime: str
    workload_id: str
    predicted_metrics: dict[str, Any] = Field(default_factory=dict)
    predicted_bottlenecks: list[str] = Field(default_factory=list)
    risk_label: Optional[str] = None
    compiler_notes: Optional[str] = None

    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)


# --------------------------------------------------------------------------- #
# §5.3 Experiment plan
# --------------------------------------------------------------------------- #

ExperimentType = Literal[
    "kernel_microbench",
    "serving_bench",
    "small_training_ablation",
    "quality_eval",
    "import_external_result",
]

ExperimentStatus = Literal[
    "planned", "approved", "running", "completed", "failed", "skipped"
]


class ExperimentPlan(_Base):
    schema_version: int = 1
    id: str
    name: str
    experiment_type: ExperimentType
    candidate_ids: list[str]
    decision_unblocked: str
    uncertainty_target: str
    estimated_cost: dict[str, Any] = Field(default_factory=dict)
    required_resources: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = "planned"

    # Harness bookkeeping.
    rationale: Optional[str] = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)


# --------------------------------------------------------------------------- #
# §5.4 Measurement
# --------------------------------------------------------------------------- #

MeasurementType = Literal[
    "kernel",
    "serving",
    "training",
    "eval",
    "manual_import",
]


class Measurement(_Base):
    schema_version: int = 1
    id: str
    candidate_id: str
    experiment_id: Optional[str] = None
    measurement_type: MeasurementType
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None
    hardware_id: Optional[str] = None
    runtime: Optional[str] = None
    workload_id: Optional[str] = None
    step: Optional[int] = None
    seed: Optional[int] = None
    extra: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    created_at: str = Field(default_factory=_now_iso)


# --------------------------------------------------------------------------- #
# §5.5 Fitted calibration / residual / interaction
# --------------------------------------------------------------------------- #

FitTarget = Literal["throughput", "serving", "quality_residual", "interaction"]
FitStatus = Literal["draft", "validated", "stale", "insufficient_data"]


class FittedCalibration(_Base):
    schema_version: int = 1
    id: str
    name: str
    target: FitTarget
    input_measurement_query: dict[str, Any] = Field(default_factory=dict)
    n_measurements: int
    functional_form: str
    coefficients: dict[str, Any] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    holdout_error: Optional[dict[str, Any]] = None
    valid_for: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    status: FitStatus = "draft"

    # When status == "insufficient_data" the fitter should populate this.
    needed_measurements: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# §5.6 Decision state
# --------------------------------------------------------------------------- #

class DecisionState(_Base):
    schema_version: int = 1
    id: str
    name: str
    candidate_ids: list[str]
    observed_summary: dict[str, Any] = Field(default_factory=dict)
    current_frontier: list[str] = Field(default_factory=list)
    uncertainty_summary: dict[str, Any] = Field(default_factory=dict)
    disagreements: list[dict[str, Any]] = Field(default_factory=list)
    recommended_next_experiments: list[str] = Field(default_factory=list)
    human_decisions_required: list[dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None
    created_at: str = Field(default_factory=_now_iso)
    provenance: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# §14 Human decision record
# --------------------------------------------------------------------------- #

class HumanDecision(_Base):
    schema_version: int = 1
    id: str
    decision_type: str
    prompt: str
    options: list[str]
    selected_option: Optional[str] = None
    rationale: Optional[str] = None
    created_at: str = Field(default_factory=_now_iso)
    resolved_at: Optional[str] = None
    provenance: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Public registry of all schemas (used by store + docs generator)
# --------------------------------------------------------------------------- #

ALL_SCHEMAS: tuple[type[_Base], ...] = (
    Candidate,
    ACCorePrediction,
    ExperimentPlan,
    Measurement,
    FittedCalibration,
    DecisionState,
    HumanDecision,
)


__all__ = [
    "Candidate",
    "ACCorePrediction",
    "ExperimentPlan",
    "ExperimentType",
    "ExperimentStatus",
    "Measurement",
    "MeasurementType",
    "FittedCalibration",
    "FitTarget",
    "FitStatus",
    "DecisionState",
    "HumanDecision",
    "ALL_SCHEMAS",
]
