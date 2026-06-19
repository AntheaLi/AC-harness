# Integration guide — plugging AC-Harness into an existing recipe

AC-Harness is designed to sit *alongside* a lab's existing training, eval,
and benchmarking stack. It does not train models, run kernels, or drive a
serving runtime. It owns four things:

1. The candidate set (which architectures are in play)
2. The evidence store (measurements from your runs)
3. The fitter (residuals on top of AC-Core's predictions)
4. The decision report + planner (what's blocking a decision; what to run next)

This guide walks through what a lab needs to do to plug it in.

## 0. The contract in one paragraph

A "Measurement" is a single (candidate, metric, value, step?) row. Your
existing recipe already emits these — as a CSV, a JSONL log, the JSON dump
of an eval harness, or the printout at the end of a benchmark. The
adapters in `ac_harness/adapters/` translate from those formats into the
canonical envelope below; `ach import-results` writes them to the store.
Once measurements land in the store, every other harness command — fit,
report, plan-next — operates on them without caring where they came from.

## 1. The canonical input envelope

Every adapter emits, and `ach import-results` consumes, JSON of this shape:

```json
{
  "benchmark_type": "eval | serving_bench | training | kernel_microbench",
  "hardware_id":   "h100_sxm",
  "runtime":       "vllm | lm_eval_harness | torchtitan | ...",
  "workload_id":   "<lab-defined>",
  "notes":         "<free text>",
  "results": [
    {
      "candidate_id": "<must match a CandidateSet entry>",
      "metrics":      {"<metric_name>": <float>, ...},
      "units":        {"<metric_name>": "<unit>", ...},
      "seed":         <int | null>,
      "provenance":   "<free text>"
    }
  ]
}
```

Field-by-field:

| Field | Required | Notes |
|---|---|---|
| `benchmark_type` | yes | One of the four buckets. Drives planner heuristics. |
| `hardware_id` | yes | Any string. Used to scope calibration fits. |
| `runtime` | yes | Any string. Used to scope calibration fits. |
| `workload_id` | yes | Lab-defined; lets you partition by serving mix. |
| `notes` | optional | Free-text. Renders in the decision report. |
| `results[].candidate_id` | yes | Must match an `id` in some ingested `CandidateSet.json`. |
| `results[].metrics` | yes | `{name: float}`. See metric naming below. |
| `results[].units` | optional | `{name: unit_string}`. For documentation only. |
| `results[].seed` | optional | For training callbacks, use `step`. The store treats each (candidate, seed) as a distinct sample. |
| `results[].provenance` | optional | Lineage string preserved verbatim in the report. |

## 2. Metric naming conventions

The harness's decision logic uses specific metric names to bucket coverage.
Adapters emit these names by default; if you're hand-rolling an envelope,
use them where the meaning matches:

| Bucket | Recognized metric names | Emitted by |
|---|---|---|
| `throughput` | `throughput_tps`, `decode_throughput_tps`, `serving_throughput_tps` | vLLM adapter |
| `kernel_bandwidth` | `decode_kv_bandwidth_gbps`, `kv_read_bw_gbps` | (write your own) |
| `quality_loss` | `val_loss`, `validation_loss` | training-callback adapter |
| `quality_task` | `downstream_score`, `needle_accuracy`, `mqar_accuracy` | lm-eval adapter (sets `downstream_score`), training-callback adapter |

Anything else (e.g. `lmeval_mmlu`, `ttft_ms_p99`, `request_throughput_rps`)
is stored and fittable, just doesn't count toward bucket coverage. That's
intentional — coverage is about whether the decision has enough signal,
not about how many numbers you wrote down.

## 3. Mapping a run to a candidate ID

A `candidate_id` is whatever lab-side string you want, as long as it
appears in some `CandidateSet.json` you've ingested. The convention this
guide assumes:

- One `CandidateSet.json` per research question (e.g. "should we ship a
  GQA-8 variant of our 8B base?")
- One candidate per architecture/recipe variant ("base", "gqa8", "swa4k")
- One training run per candidate; if you sweep seeds, encode them via
  `results[].seed` rather than minting new candidate IDs

You ingest the candidate set once at the start of a study:

```bash
ach ingest-ac-core --input /path/to/candidate_set_dir/ --store runs/study.sqlite
```

If you generated the candidates from AC-Core, point at the directory that
contains AC-Core's `CandidateSet.json` / `PredictedPareto.json` /
`DeltaReport.json` / `CalibrationRequest.json`. If you're hand-rolling,
the minimum viable file is at
`examples/real_published/llama3_family_h100/CandidateSet.json`.

## 4. The three reference adapters

All three live in `ac_harness/adapters/` and are runnable as
`python -m ac_harness.adapters.<name>`.

### `lm_eval` — lm-evaluation-harness → `eval` measurements

```bash
# 1) Run lm-eval-harness against your model
lm_eval --model hf \
        --model_args pretrained=meta-llama/Meta-Llama-3-8B \
        --tasks mmlu,arc_challenge,piqa,winogrande \
        --batch_size 4 \
        --output_path /tmp/lm_eval/llama3_8b/

# 2) Convert to the harness envelope
python -m ac_harness.adapters.lm_eval \
    --candidate-id llama3_8b \
    --input  /tmp/lm_eval/llama3_8b/results_*.json \
    --out    /tmp/llama3_8b_quality.json \
    --hardware-id h100_sxm

# 3) Ingest
ach import-results --plan-id quality_smoke \
    --input /tmp/llama3_8b_quality.json \
    --store runs/study.sqlite
```

Emits one `lmeval_<task>` metric per task plus a `downstream_score`
headline (picked from MMLU > ARC-C > HellaSwag > PIQA > Winogrande > CSQA
> BoolQ, whichever appears first) so the quality bucket fires.

### `vllm_serving` — vLLM `benchmark_serving.py` → `serving_bench` measurements

```bash
# 1) Run vLLM and its serving benchmark
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 1 &
python vllm/benchmarks/benchmark_serving.py \
    --backend vllm --model meta-llama/Meta-Llama-3-8B \
    --dataset-name sharegpt --num-prompts 500 --request-rate 8 \
    --save-result --result-filename /tmp/vllm_llama3_8b.json

# 2) Convert
python -m ac_harness.adapters.vllm_serving \
    --candidate-id llama3_8b \
    --input /tmp/vllm_llama3_8b.json \
    --out   /tmp/llama3_8b_throughput.json \
    --hardware-id h100_sxm

# 3) Ingest
ach import-results --plan-id throughput_smoke \
    --input /tmp/llama3_8b_throughput.json \
    --store runs/study.sqlite
```

Extracts `request_throughput`, `output_throughput` → `throughput_tps`,
TTFT / TBT / ITL medians and p99s. The `throughput_tps` field fires
the throughput bucket.

### `training_callback` — training-loop hook → `training` measurements

Two ways to use it.

**Option A — in-process buffer** dropped into your eval hook:

```python
from ac_harness.adapters.training_callback import TrainingMetricsBuffer

buf = TrainingMetricsBuffer(
    candidate_id="my_run_42",
    hardware_id="h100_sxm",
    runtime="torchtitan",
    workload_id="pretrain_8b",
    out_path="/runs/my_run_42.metrics.json",
)

# inside your training loop's eval hook:
buf.record(step=step, val_loss=val_loss_scalar, downstream_score=mmlu_acc)

# at the end of the run (or periodically):
buf.flush()
```

**Option B — convert an existing log file** the recipe already writes
(JSONL, JSON array, or CSV with a header row):

```bash
python -m ac_harness.adapters.training_callback \
    --candidate-id my_run_42 \
    --input  /runs/my_run_42.metrics.jsonl \
    --out    /runs/my_run_42.metrics.json \
    --hardware-id h100_sxm \
    --runtime torchtitan
```

Each step becomes a row with `seed=<step>`, so the residual fitter sees
the full eval curve rather than just a single average.

## 5. End-to-end story — a typical lab study

Imagine the question is: *"On H100, does a GQA-8 variant of our 8B base
beat the MHA baseline at long context?"* The candidate set has two
entries: `base_mha` and `gqa8`. You already train both with torchtitan,
already run lm-eval on both, already run vLLM benchmark_serving on both.

```bash
# Day 0 — boot the study
ach init --store runs/gqa_study.sqlite
ach ingest-ac-core --input candidate_sets/gqa_study/ --store runs/gqa_study.sqlite

# After each training run finishes, the recipe's post-hook calls:
python -m ac_harness.adapters.training_callback \
    --candidate-id base_mha --input  runs/base_mha/metrics.jsonl \
    --out runs/base_mha/metrics.json --hardware-id h100_sxm --runtime torchtitan
ach import-results --plan-id pretrain_base --input runs/base_mha/metrics.json \
    --store runs/gqa_study.sqlite

# After eval:
python -m ac_harness.adapters.lm_eval \
    --candidate-id base_mha --input results/base_mha/lm_eval_*.json \
    --out results/base_mha/quality.json --hardware-id h100_sxm
ach import-results --plan-id eval_base --input results/base_mha/quality.json \
    --store runs/gqa_study.sqlite

# After serving bench:
python -m ac_harness.adapters.vllm_serving \
    --candidate-id base_mha --input bench/base_mha/vllm.json \
    --out bench/base_mha/throughput.json --hardware-id h100_sxm
ach import-results --plan-id bench_base --input bench/base_mha/throughput.json \
    --store runs/gqa_study.sqlite

# (Repeat the three blocks above for the gqa8 candidate.)

# Day N — what does the harness say?
ach fit-calibration  --store runs/gqa_study.sqlite --target throughput --out runs/gqa_study/calibration/
ach fit-residual     --store runs/gqa_study.sqlite --out runs/gqa_study/residuals/
ach decision-report  --store runs/gqa_study.sqlite --out runs/gqa_study/DecisionStateReport.md
ach plan-next        --store runs/gqa_study.sqlite --budget small --out runs/gqa_study/next_experiment.md
ach export-ac-core-feedback --store runs/gqa_study.sqlite --out runs/gqa_study/feedback/
```

The decision report tells you whether you have enough evidence yet, where
the supported frontier sits, and what the disagreements with AC-Core's
predictions are. `plan-next` proposes the cheapest next experiment that
unblocks an unresolved decision. The feedback export hands back fitted
calibrations and residuals so AC-Core's next candidate generation
benefits from what your lab observed.

## 6. What's NOT in v1, and what to do instead

| Want | v1 status | Workaround |
|---|---|---|
| Slurm/SkyPilot submission from `ach materialize` | dry-run only — emits a stub `run.sh`/`run.py` | Replace the stub with your `sbatch` wrapper; ach still owns the plan JSON |
| TRT-LLM `gptManagerBenchmark` adapter | not bundled | Same shape as the vLLM adapter; copy `vllm_serving.py` and remap field names |
| Distributed training-loop hooks (multi-rank flush) | not bundled | Only rank 0 should call `buf.flush()`; gather from all ranks first |
| Quality-loss bucket from anything other than `val_loss` | name-bound | Either alias your metric to `val_loss` at import time, or extend `KEY_METRIC_BUCKETS` in `ac_harness/decision/state.py` |

If you find yourself reaching for any of these, both the adapter modules
and `decision/state.py` are small (~150–250 lines each) and meant to be
forked — there's no clever framework to fight.
