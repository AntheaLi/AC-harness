# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Slurm template emitter.

v1 deliberately does not submit jobs. We emit a sbatch script next to
the materialized plan so an operator can run `sbatch run.slurm`.
"""
from __future__ import annotations

import os

from ..schemas import ExperimentPlan

_SBATCH_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name=ach_{plan_id}
#SBATCH --output=ach_{plan_id}.%j.out
#SBATCH --error=ach_{plan_id}.%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gpu_count}
#SBATCH --time=01:00:00

# AC-Harness sbatch template.
# Pair with the materialized run.py / plan.json.
set -euo pipefail
cd "$(dirname "$0")"
python run.py plan.json results.json
"""


def emit_slurm(plan: ExperimentPlan, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run.slurm")
    gpu_count = int(plan.required_resources.get("gpu_count", 1))
    with open(path, "w") as f:
        f.write(_SBATCH_TEMPLATE.format(plan_id=plan.id, gpu_count=gpu_count))
    os.chmod(path, 0o755)
    return path
