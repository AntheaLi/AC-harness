# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
DecisionStateReport.md generator (§13.3).

Renders a human-readable Markdown report from the current store +
the latest DecisionState. The report is the central user-facing
output of AC-Harness.

It explicitly does NOT say "this is the optimal architecture." It says
what's supported, what's uncertain, and what to do next.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from ..planner import render_plan_markdown
from ..schemas import (
    ACCorePrediction,
    Candidate,
    DecisionState,
    FittedCalibration,
    Measurement,
)
from ..store import EvidenceStore
from .disagreement import detect_disagreements
from .frontier import compute_frontier
from .state import build_decision_state


SECTIONS_TOC = [
    "1. Research question / decision",
    "2. Candidate set",
    "3. Current evidence table",
    "4. Current supported frontier",
    "5. Uncertain candidates",
    "6. Disagreements / surprises",
    "7. Fitted calibration / residuals available",
    "8. Recommended next experiments",
    "9. Human decision points",
    "10. Export files for AC-Core",
]


def generate_report(
    store: EvidenceStore,
    *,
    research_question: str = "Which candidate architecture should we ship next?",
    out_path: str | None = None,
    decision_state: DecisionState | None = None,
    persist_decision_state: bool = True,
) -> str:
    ds = decision_state or build_decision_state(store)
    frontier = compute_frontier(store)
    disagreements = detect_disagreements(store)

    ds.current_frontier = list(frontier.observed_supported or frontier.prediction_only_supported)
    ds.disagreements = list(disagreements)

    candidates: list[Candidate] = store.list_candidates()
    predictions: list[ACCorePrediction] = store.list_predictions()
    measurements: list[Measurement] = store.query_measurements()
    fits: list[FittedCalibration] = store.list_fitted_calibrations()

    md = _render(
        research_question=research_question,
        candidates=candidates,
        predictions=predictions,
        measurements=measurements,
        ds=ds,
        frontier_kind=frontier.kind,
        observed_supported=frontier.observed_supported,
        prediction_only_supported=frontier.prediction_only_supported,
        observed_failures=frontier.observed_failures,
        surprises=frontier.surprises,
        disagreements=disagreements,
        fits=fits,
        store=store,
    )

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(md)

    if persist_decision_state:
        store.insert_decision_state(ds)
    return md


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


def _render(
    *,
    research_question: str,
    candidates: list[Candidate],
    predictions: list[ACCorePrediction],
    measurements: list[Measurement],
    ds: DecisionState,
    frontier_kind: str,
    observed_supported: list[str],
    prediction_only_supported: list[str],
    observed_failures: list[str],
    surprises: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
    fits: list[FittedCalibration],
    store: EvidenceStore,
) -> str:
    L: list[str] = []
    L.append("# AC-Harness Decision State Report\n")
    L.append("> AC-Harness does not declare an optimal architecture. It summarizes the\n"
             "> current evidence and tells us what to measure next.\n")

    # §1 — Research question
    L.append("## 1. Research question / decision\n")
    L.append(f"{research_question}\n")

    # §2 — Candidate set
    L.append("## 2. Candidate set\n")
    if not candidates:
        L.append("_No candidates ingested yet._\n")
    else:
        L.append("| Candidate | Source | Baseline | Changed fields | AC-Core prediction |")
        L.append("|---|---|---|---|---|")
        for c in candidates:
            cf = ", ".join(f"{k}={v}" for k, v in (c.changed_fields or {}).items()) or "—"
            L.append(
                f"| `{c.id}` | {c.source} | `{c.baseline_id or '—'}` | {cf} | "
                f"`{c.ac_core_prediction_id or '—'}` |"
            )
        L.append("")

    # §3 — Current evidence table
    L.append("## 3. Current evidence table\n")
    table = _evidence_table(candidates, measurements)
    L.append(table)

    # §4 — Current supported frontier
    L.append("## 4. Current supported frontier\n")
    if frontier_kind == "prediction_only":
        L.append("_Prediction-only frontier (no observed measurements yet)._\n")
    L.append(f"- **Kind:** `{frontier_kind}`")
    L.append(f"- **Observed-supported candidates:** {_fmt_list(observed_supported)}")
    L.append(f"- **Prediction-supported but unmeasured:** {_fmt_list(prediction_only_supported)}")
    L.append(f"- **Observed failures:** {_fmt_list(observed_failures)}")
    L.append(f"- **Wrong-order / surprises:** {len(surprises)} item(s)\n")
    for s in surprises:
        L.append(f"  - `{s.get('candidate_id')}`: {s.get('type')} — predicted "
                 f"{s.get('predicted_throughput')}, observed {s.get('observed_throughput')}")
    L.append("")

    # §5 — Uncertain candidates
    L.append("## 5. Uncertain candidates\n")
    missing = ds.uncertainty_summary.get("missing_buckets_by_candidate", {})
    uncertain = [cid for cid, miss in missing.items() if miss]
    if not uncertain:
        L.append("_All key measurement buckets are populated._\n")
    else:
        L.append("| Candidate | Missing buckets |")
        L.append("|---|---|")
        for cid in sorted(uncertain):
            L.append(f"| `{cid}` | {', '.join(missing[cid])} |")
        L.append("")
        ro = ds.uncertainty_summary.get("risky_unmeasured_candidates", [])
        if ro:
            L.append(f"**Risky deltas without quality signal:** "
                     f"{', '.join(f'`{c}`' for c in ro)}\n")

    # §6 — Disagreements / surprises
    L.append("## 6. Disagreements / surprises\n")
    if not disagreements:
        L.append("_No disagreements detected with current evidence._\n")
    else:
        L.append("| Rule | Candidate | Detail |")
        L.append("|---|---|---|")
        for d in disagreements:
            detail = ", ".join(f"{k}={v}" for k, v in d.items()
                                if k not in {"rule", "candidate_id"})
            L.append(f"| `{d.get('rule')}` | `{d.get('candidate_id')}` | {detail} |")
        L.append("")

    # §7 — Fitted calibrations / residuals
    L.append("## 7. Fitted calibration / residuals available\n")
    if not fits:
        L.append("_No fits available yet. Run `ach fit-calibration` / `ach fit-residual` when ready._\n")
    else:
        L.append("| Fit | Target | Status | n_measurements | Form |")
        L.append("|---|---|---|---|---|")
        for f in fits:
            L.append(
                f"| `{f.name}` | {f.target} | {f.status} | {f.n_measurements} | "
                f"`{f.functional_form}` |"
            )
        L.append("")

    # §8 — Recommended next experiments
    L.append("## 8. Recommended next experiments\n")
    plans = [p for p in store.list_experiment_plans() if p.status == "planned"]
    if not plans:
        L.append("_No planned experiments. Run `ach plan-next` to populate._\n")
    else:
        L.append(render_plan_markdown(plans))
        L.append("")

    # §9 — Human decision points
    L.append("## 9. Human decision points\n")
    hd = ds.human_decisions_required or []
    if not hd:
        L.append("_No human decisions currently pending. The harness will flag them when\n"
                 "evidence requires sign-off (e.g. expensive runs, contradictions of priors)._\n")
    else:
        for item in hd:
            L.append(f"- {item}")
        L.append("")

    # §10 — Export files for AC-Core
    L.append("## 10. Export files for AC-Core\n")
    L.append("Run `ach export-ac-core-feedback --store <store> --out <dir>` to emit:\n")
    L.append("- `calibration/<hardware>_<runtime>_measured.json`")
    L.append("- `quality_residuals/<law>.json`")
    L.append("- `interaction_terms/<fit>.json`\n")
    L.append("AC-Core remains runnable without these files.\n")

    return "\n".join(L) + "\n"


def _fmt_list(xs: list[str]) -> str:
    return ", ".join(f"`{x}`" for x in xs) if xs else "—"


def _evidence_table(
    candidates: list[Candidate], measurements: list[Measurement]
) -> str:
    # Pick a stable set of metric columns: top by frequency.
    counts: dict[str, int] = defaultdict(int)
    for m in measurements:
        counts[m.metric_name] += 1
    metric_cols = [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:6]
    if not metric_cols:
        return "_No measurements recorded yet._\n"

    by_cand_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for m in measurements:
        by_cand_metric[(m.candidate_id, m.metric_name)].append(m.metric_value)

    L: list[str] = []
    L.append("| Candidate | " + " | ".join(metric_cols) + " |")
    L.append("|---|" + "|".join("---" for _ in metric_cols) + "|")
    for c in candidates:
        row = [f"`{c.id}`"]
        for col in metric_cols:
            vals = by_cand_metric.get((c.id, col))
            if not vals:
                row.append("—")
            else:
                mean = sum(vals) / len(vals)
                row.append(f"{mean:.3g} (n={len(vals)})")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    return "\n".join(L)
