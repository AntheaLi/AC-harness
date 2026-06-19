# Workflows

The §16 end-to-end workflow is the canonical v1 demo. It runs without GPUs.

## §16 demo

```bash
ach init --store runs/demo.sqlite

ach ingest-ac-core \
  --input examples/ac_core_outputs/llama_h100_long_chat/ \
  --store runs/demo.sqlite

ach plan-next \
  --store runs/demo.sqlite \
  --budget small \
  --out runs/demo/next_experiment.md

ach materialize \
  --plan runs/demo/plans/<plan_id>.json \
  --mode dry_run \
  --out runs/demo/materialized/

ach import-results \
  --plan-id <plan_id> \
  --input examples/imported_results/decode_kv_fake.json \
  --store runs/demo.sqlite

ach decision-report \
  --store runs/demo.sqlite \
  --out runs/demo/DecisionStateReport.md
```

Every command writes its output where you tell it to; rerun any step at will. The store is the single source of truth.

## What each command does

| Command | Inputs | Outputs |
|---|---|---|
| `init` | — | empty SQLite store |
| `ingest-ac-core` | AC-Core output dir | Candidate + ACCorePrediction rows in store |
| `plan-next` | store, `--budget` | `next_experiment.md` (ranked plan list) + per-plan JSON files |
| `materialize` | plan JSON, `--mode` | `plan.json`, `run.py`, `run.sh` (+ optional `run.slurm`) |
| `import-results` | result JSON, `--plan-id` | Measurement rows in store; plan moves to `completed` |
| `fit-calibration` | store, `--hardware-id`, `--runtime` | FittedCalibration rows + `<hw>_<runtime>_measured.json` |
| `fit-residual` | store | quality residual JSON files |
| `decision-report` | store | `DecisionStateReport.md` |
| `export-ac-core-feedback` | store | feedback file tree for AC-Core consumption |

## Iterative loop

Most real use looks like this:

1. `ingest-ac-core` → harness sees the candidate set
2. `plan-next` → harness recommends the cheapest decision-changing measurement
3. Run the experiment (or import existing results)
4. `import-results` → harness updates evidence
5. `decision-report` → see what's supported, what's uncertain, what to do next
6. Loop to step 2

`fit-calibration` / `fit-residual` are run when enough measurements accumulate; `export-ac-core-feedback` is run when you want to push observed corrections back to AC-Core.

## Decision points the harness flags for humans (§14)

- Before expensive runs
- When evaluator metrics disagree
- When fitted residuals are unstable
- When a prediction-only candidate is about to be treated as supported
- When imported results lack provenance
- When a calibration file would overwrite a validated calibration
- When a result contradicts the current compiler prior

These surface in section 9 of `DecisionStateReport.md`. v1 represents them as `HumanDecision` records but does not yet drive interactive prompts.
