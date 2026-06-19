# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
AC-Harness CLI (§15).

Typer command group `ach`. Every command takes a `--store` path; commands
that produce output files take `--out`. `--dry-run` is supported where
applicable and is a no-op for read-only commands.

The CLI is a thin shim over the library API. All logic lives in the
`ac_harness/*` modules.
"""
from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import print as rprint

from .benchmarks import build_kernel_plan, write_kernel_plan
from .decision import generate_report
from .executor import emit_slurm, import_results, materialize
from .fitter import (
    export_interaction_terms_json,
    export_measured_calibration_json,
    export_quality_residuals_json,
    fit_quality_residual,
    fit_throughput_calibration,
)
from .ingest import import_ac_core_run
from .planner import plan_next_experiments, render_plan_markdown
from .store import EvidenceStore


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AC-Harness CLI. Observes, learns, and chooses the next experiment.",
)


# --------------------------------------------------------------------------- #
# ach init
# --------------------------------------------------------------------------- #


@app.command("init")
def cmd_init(
    store: str = typer.Option(..., "--store", help="Path to the SQLite store file."),
):
    """Initialize an empty evidence store at the given path."""
    os.makedirs(os.path.dirname(os.path.abspath(store)) or ".", exist_ok=True)
    s = EvidenceStore(store)
    s.close()
    rprint(f"[green]initialized[/green] {store}")


# --------------------------------------------------------------------------- #
# ach ingest-ac-core
# --------------------------------------------------------------------------- #


@app.command("ingest-ac-core")
def cmd_ingest_ac_core(
    input: str = typer.Option(..., "--input", help="Directory of AC-Core JSON outputs."),
    store: str = typer.Option(..., "--store"),
):
    """Import AC-Core candidate + prediction outputs into the store."""
    s = EvidenceStore(store)
    ids = import_ac_core_run(input, s)
    s.close()
    rprint(f"[green]ingested[/green] {len(ids)} candidates from {input}")


# --------------------------------------------------------------------------- #
# ach plan-next
# --------------------------------------------------------------------------- #


@app.command("plan-next")
def cmd_plan_next(
    store: str = typer.Option(..., "--store"),
    budget: str = typer.Option("small", "--budget", help="small | medium | large"),
    out: str = typer.Option(..., "--out", help="Markdown file to write."),
    max_plans: int = typer.Option(5, "--max-plans"),
    write_plan_files: bool = typer.Option(
        True, "--write-plan-files/--no-write-plan-files",
        help="Also write per-plan JSON files into <out_dir>/plans/.",
    ),
):
    """Rank and write the next experiment recommendations."""
    s = EvidenceStore(store)
    plans = plan_next_experiments(s, budget=budget, max_plans=max_plans)
    md = render_plan_markdown(plans)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    if write_plan_files:
        plans_dir = out_path.parent / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        for p in plans:
            # When the plan is a kernel benchmark, emit a §9.3-shaped plan JSON
            # so the materialize step has something concrete to run.
            if p.experiment_type == "kernel_microbench":
                workload = p.config.get("workload", "attention_decode")
                # Map workload hint → kernel category.
                bench = "attention_decode" if workload == "decode_kv" else "attention_decode"
                kp = build_kernel_plan(
                    benchmark_type=bench,
                    candidate_ids=p.candidate_ids,
                    hardware_id="h100_sxm",
                    runtime="vllm",
                    notes=f"Auto-emitted from ExperimentPlan {p.id}.",
                )
                write_kernel_plan(kp, str(plans_dir / f"{p.id}.json"))
            else:
                (plans_dir / f"{p.id}.json").write_text(p.model_dump_json(indent=2))

    s.close()
    rprint(f"[green]wrote[/green] {len(plans)} plans → {out}")


# --------------------------------------------------------------------------- #
# ach materialize
# --------------------------------------------------------------------------- #


@app.command("materialize")
def cmd_materialize(
    plan: str = typer.Option(..., "--plan", help="Path to an ExperimentPlan JSON file."),
    mode: str = typer.Option("dry_run", "--mode", help="dry_run | local_script | slurm_script"),
    out: str = typer.Option(..., "--out", help="Output directory."),
):
    """Materialize a benchmark plan into runnable scripts (no execution)."""
    import json as _json

    from .schemas import ExperimentPlan

    raw = _json.loads(Path(plan).read_text())
    # Accept either a full ExperimentPlan JSON or a benchmark plan JSON
    # produced by `plan-next`. If the file has `experiment_type`, treat it as
    # an ExperimentPlan; otherwise wrap it as a kernel_microbench.
    if "experiment_type" in raw:
        plan_obj = ExperimentPlan.model_validate(raw)
    else:
        plan_obj = ExperimentPlan(
            id=raw.get("benchmark_type", "kernel") + "_plan",
            name=f"{raw.get('benchmark_type', 'kernel')} benchmark",
            experiment_type="kernel_microbench",
            candidate_ids=raw.get("candidate_ids", []),
            decision_unblocked="—",
            uncertainty_target="—",
            estimated_cost={"tier": 1, "label": "low"},
            required_resources={"gpu_count": 1},
            config=raw,
        )
    files = materialize(plan_obj, out)
    if mode == "slurm_script":
        files.append(emit_slurm(plan_obj, out))
    rprint(f"[green]materialized[/green] {len(files)} file(s) → {out}")


# --------------------------------------------------------------------------- #
# ach import-results
# --------------------------------------------------------------------------- #


@app.command("import-results")
def cmd_import_results(
    plan_id: str = typer.Option(..., "--plan-id"),
    input: str = typer.Option(..., "--input"),
    store: str = typer.Option(..., "--store"),
):
    """Import result JSON into the store as Measurement rows."""
    s = EvidenceStore(store)
    ids = import_results(s, result_path=input, plan_id=plan_id)
    s.close()
    rprint(f"[green]imported[/green] {len(ids)} measurements from {input}")


# --------------------------------------------------------------------------- #
# ach fit-calibration
# --------------------------------------------------------------------------- #


@app.command("fit-calibration")
def cmd_fit_calibration(
    target: str = typer.Option("throughput", "--target"),
    store: str = typer.Option(..., "--store"),
    out: str = typer.Option(..., "--out"),
    hardware_id: str | None = typer.Option(None, "--hardware-id"),
    runtime: str | None = typer.Option(None, "--runtime"),
):
    """Fit throughput calibration and export MeasuredCalibration.json."""
    if target != "throughput":
        raise typer.BadParameter(f"target {target!r} not supported in v1")
    s = EvidenceStore(store)
    fits = fit_throughput_calibration(s, hardware_id=hardware_id, runtime=runtime)
    out_dir = Path(out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out.endswith(".json"):
        # Write a single bundled file at the exact path.
        from .fitter.calibration_fit import export_measured_calibration_json as _exp
        # Use the directory variant under the parent.
        paths = _exp(s, out_dir=str(out_dir.parent), hardware_id=hardware_id, runtime=runtime)
    else:
        paths = export_measured_calibration_json(
            s, out_dir=out, hardware_id=hardware_id, runtime=runtime,
        )
    s.close()
    rprint(f"[green]fit[/green] {len(fits)} calibration(s); wrote {len(paths)} file(s)")


# --------------------------------------------------------------------------- #
# ach fit-residual
# --------------------------------------------------------------------------- #


@app.command("fit-residual")
def cmd_fit_residual(
    target: str = typer.Option("quality", "--target"),
    store: str = typer.Option(..., "--store"),
    out: str = typer.Option(..., "--out"),
):
    """Fit quality residuals and export quality_residual JSON files."""
    if target != "quality":
        raise typer.BadParameter(f"target {target!r} not supported in v1")
    s = EvidenceStore(store)
    fits = fit_quality_residual(s)
    out_dir = Path(out).parent if out.endswith(".json") else Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = export_quality_residuals_json(s, out_dir=str(out_dir))
    s.close()
    rprint(f"[green]fit[/green] {len(fits)} residual(s); wrote {len(paths)} file(s)")


# --------------------------------------------------------------------------- #
# ach decision-report
# --------------------------------------------------------------------------- #


@app.command("decision-report")
def cmd_decision_report(
    store: str = typer.Option(..., "--store"),
    out: str = typer.Option(..., "--out"),
    question: str = typer.Option(
        "Which candidate architecture should we ship next?", "--question"
    ),
):
    """Generate the DecisionStateReport.md."""
    s = EvidenceStore(store)
    generate_report(s, research_question=question, out_path=out)
    s.close()
    rprint(f"[green]wrote[/green] decision state report → {out}")


# --------------------------------------------------------------------------- #
# ach export-ac-core-feedback
# --------------------------------------------------------------------------- #


@app.command("export-ac-core-feedback")
def cmd_export_feedback(
    store: str = typer.Option(..., "--store"),
    out: str = typer.Option(..., "--out"),
):
    """Write calibration/quality_residuals/interaction_terms files for AC-Core."""
    s = EvidenceStore(store)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_paths = export_measured_calibration_json(s, out_dir=str(out_dir / "calibration"))
    qr_paths = export_quality_residuals_json(s, out_dir=str(out_dir / "quality_residuals"))
    it_paths = export_interaction_terms_json(s, out_dir=str(out_dir / "interaction_terms"))
    s.close()
    rprint(
        f"[green]exported[/green] calibration={len(cal_paths)} "
        f"residuals={len(qr_paths)} interactions={len(it_paths)} → {out}"
    )


if __name__ == "__main__":
    app()
