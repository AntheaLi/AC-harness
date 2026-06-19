# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
JSONL helpers.

The SQLite store is the source of truth; JSONL is a portable export
format. This module reads JSONL files back into pydantic models so an
exported run can be replayed into a fresh store.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

from ..schemas import (
    ACCorePrediction,
    Candidate,
    DecisionState,
    ExperimentPlan,
    FittedCalibration,
    Measurement,
)

M = TypeVar("M", bound=BaseModel)


def load_jsonl(path: str, cls: Type[M]) -> list[M]:
    out: list[M] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(cls.model_validate_json(line))
    return out


def dump_jsonl(path: str, rows: Iterable[BaseModel]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


# Convenience loaders for the standard export layout.
def load_candidates(dirpath: str) -> list[Candidate]:
    return load_jsonl(os.path.join(dirpath, "candidates.jsonl"), Candidate)


def load_predictions(dirpath: str) -> list[ACCorePrediction]:
    return load_jsonl(os.path.join(dirpath, "predictions.jsonl"), ACCorePrediction)


def load_measurements(dirpath: str) -> list[Measurement]:
    return load_jsonl(os.path.join(dirpath, "measurements.jsonl"), Measurement)


def load_experiment_plans(dirpath: str) -> list[ExperimentPlan]:
    return load_jsonl(os.path.join(dirpath, "experiment_plans.jsonl"), ExperimentPlan)


def load_fitted_calibrations(dirpath: str) -> list[FittedCalibration]:
    return load_jsonl(
        os.path.join(dirpath, "fitted_calibrations.jsonl"), FittedCalibration
    )


def load_decision_states(dirpath: str) -> list[DecisionState]:
    return load_jsonl(os.path.join(dirpath, "decision_states.jsonl"), DecisionState)
