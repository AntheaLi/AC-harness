# Tier 1 example — real published numbers, no GPU required

A worked candidate set built entirely from public, citable benchmark numbers,
so you can exercise AC-Harness end-to-end with non-synthetic signal without
running any kernels yourself.

## Candidate set

Four members of the Llama 3 family on H100 SXM:

| Candidate ID | HF model | Notes |
|---|---|---|
| `llama3_8b` | `meta-llama/Meta-Llama-3-8B` | baseline; 8B, 8k context |
| `llama3_70b` | `meta-llama/Meta-Llama-3-70B` | size delta (8B → 70B) |
| `llama31_8b` | `meta-llama/Meta-Llama-3.1-8B` | context-extension delta (8k → 128k via Llama-3 RoPE scaling) |
| `llama31_70b` | `meta-llama/Meta-Llama-3.1-70B` | combined size + context delta |

This isn't a clean architectural-delta set (Llama 3 → 3.1 is a retrain, not a
graft), but it's good enough to make every stage of the harness do real work.

## What's in here

| File | What it is |
|---|---|
| `CandidateSet.json` | AC-Core-format candidate set (consumed by `ach ingest-ac-core`) |
| `quality_lm_eval.json` | Quality measurements transcribed from Meta's model cards |
| `throughput_vllm.json` | Approximate serving throughput from published vLLM benchmarks |
| `convert_lm_eval_output.py` | Shim that turns real lm-eval-harness output into the same shape, for when you want to replace the published numbers with your own runs |

### Source attribution

- **Quality** — Meta's official Llama 3 / 3.1 model cards (pretrained, 5-shot
  MMLU macro, 25-shot ARC-Challenge, 7-shot CommonSenseQA). Single published
  score per task, so n=1; the harness will report uncertainty accordingly.
  See https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md and
  https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md.
- **Throughput** — approximate order-of-magnitude numbers from the vLLM 0.6
  perf blog (https://blog.vllm.ai/2024/09/05/perf-update.html) and public
  cloud-provider Llama-3 H100 benchmark posts. These are intentionally rough;
  for exact figures run `vllm/benchmarks/benchmark_serving.py` yourself and
  import the JSON via the same `import-results` path.

The `provenance` field on every measurement row preserves the source so the
decision report cites it back.

## Running the harness against this fixture

```bash
# 1) Init a fresh store
ach init --store runs/real.sqlite

# 2) Ingest the candidate set
ach ingest-ac-core \
    --input examples/real_published/llama3_family_h100/ \
    --store runs/real.sqlite

# 3) Import quality + throughput
ach import-results --plan-id meta_model_card \
    --input examples/real_published/llama3_family_h100/quality_lm_eval.json \
    --store runs/real.sqlite

ach import-results --plan-id vllm_published \
    --input examples/real_published/llama3_family_h100/throughput_vllm.json \
    --store runs/real.sqlite

# 4) Fit a calibration on the throughput data
ach fit-calibration --store runs/real.sqlite \
    --target throughput --out runs/real/calibration/

# 5) Decision report
ach decision-report --store runs/real.sqlite \
    --out runs/real/DecisionStateReport.md
```

## Replacing the published numbers with your own runs

If you have a single GPU (a free Colab T4 is enough for the 1B–3B end of the
family), `convert_lm_eval_output.py` turns lm-eval-harness's JSON output into
the same shape this fixture uses. The harness will treat the imported runs
identically to the published numbers — same store, same fitter, same report.
