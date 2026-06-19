# AC-Core ↔ AC-Harness boundary

> AC-Core proposes and optimizes.
> AC-Harness observes, learns, and chooses the next experiment.

This document describes the contract between the two systems. If a feature crosses the line it should be moved to the right side; if it falls in a gray area it is listed below as a deliberate non-goal of v1.

## What AC-Harness must NOT implement

The following live in AC-Core. AC-Harness reads their outputs as opaque data and never re-derives them.

- Hardware-aware architecture optimizer
- Tile lattice compiler
- Quality prior used for compile-time ranking
- Baseline-aware delta generator
- Safe / risky / research / rejected taxonomy
- Kernel availability matrix used by the compiler
- Greenfield architecture search
- Predicted Pareto frontier generation as a compiler output

## What AC-Harness reads from AC-Core

Under a directory passed to `ach ingest-ac-core --input <dir>`:

| File | Used for |
|---|---|
| `CandidateSet.json` | Create `Candidate` rows |
| `DeltaReport.json` | Create `Candidate` rows with `baseline_id` propagated |
| `PredictedPareto.json` | Create `ACCorePrediction` rows (and embedded candidates) |
| `CalibrationRequest.json` | Touched but not stored in v1 |

Unknown fields in any of these are tolerated (pydantic `extra="allow"`); AC-Harness never mutates AC-Core files.

## What AC-Harness writes back

Optional feedback files emitted by `ach export-ac-core-feedback`:

| Path | Purpose |
|---|---|
| `calibration/<hw>_<runtime>_measured.json` | Measured throughput correction factors |
| `quality_residuals/<law>.json` | Observed quality residuals on top of AC-Core's prior |
| `interaction_terms/<fit>.json` | Non-additive cross-terms when data supports them |

Insufficient-data fits are deliberately NOT exported, so AC-Core never picks up unstable laws.

**AC-Core must remain runnable without these files.** The harness produces them as recommendations; AC-Core decides whether to consume them.

## Deliberate non-goals for v1

The following stay out of v1 — they may belong to AC-Harness later, but not now:

- Full production serving engine
- Full distributed training framework
- Full model zoo
- Multi-user auth
- Real-time web dashboard
- Automatic paper writing
- Full video/multimodal workflows
- Private-lab calibration assumptions
- Slurm job submission (we only emit sbatch templates)
- Real eval execution (we only ingest results)

## Every harness module carries a one-line header

```python
# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
```

If you find yourself wanting to predict throughput, predict quality, or generate a new architecture in this codebase, that work belongs in AC-Core — not here.
