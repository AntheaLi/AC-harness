# AC-harness

![ACharness](assets/image1.png)


AC-Harness is a loop scaffold accompanying AC for AI automated architecture opitmization. It consumes AC-Core candidate sets and observed experiment results, and turn architecture hypotheses into measured evidence and tells us what to measure next.

> AC-Harness is the external research loop around hardware-aware architecture design. 

### What this does

```text
candidate set / research question
  → experiment plan
  → kernel / serving / small-training / eval execution or import
  → observed result store
  → fitted calibration or residual law
  → decision-state report
  → next experiment recommendation
```


## Install

```bash
pip install -e .
```

The harness has no Python-level dependency on AC-Core — it consumes AC-Core'
JSON outputs (`CandidateSet`, `PredictedPareto`, `DeltaReport`, `CalibrationRequest`)
and writes its own evidence. If you also want AC-Core for generating those
inputs, install it from the AC compiler repo (`v0/`):

```bash
pip install -e git+https://github.com/AntheaLi/AC.git  # gives you ac-compile / ac-delta-eval / ac-stress
```

Otherwise the bundled `examples/ac_core_outputs/llama_h100_long_chat/` is
enough to run the quick start below end-to-end.

## Quick start (matches §16 of the spec)

```bash
ach init --store runs/demo.sqlite
ach ingest-ac-core --input examples/ac_core_outputs/llama_h100_long_chat/ --store runs/demo.sqlite
ach plan-next --store runs/demo.sqlite --budget small --out runs/demo/next_experiment.md
ach materialize --plan runs/demo/plans/decode_kv_plan.json --mode dry_run --out runs/demo/materialized/
ach import-results --plan-id decode_kv_plan --input examples/imported_results/decode_kv_fake.json --store runs/demo.sqlite
ach decision-report --store runs/demo.sqlite --out runs/demo/DecisionStateReport.md
```

## examples

simple examples: `examples/real_published/llama3_family_h100/`

```bash
ach init --store runs/real.sqlite
ach ingest-ac-core --input examples/real_published/llama3_family_h100/ --store runs/real.sqlite
ach import-results --plan-id meta_model_card  --input examples/real_published/llama3_family_h100/quality_lm_eval.json --store runs/real.sqlite
ach import-results --plan-id vllm_published   --input examples/real_published/llama3_family_h100/throughput_vllm.json --store runs/real.sqlite
ach fit-calibration --store runs/real.sqlite --target throughput --out runs/real/calibration/
ach decision-report --store runs/real.sqlite --out runs/real/DecisionStateReport.md
```


## Plugging into an existing work loop

AC-Harness is thin layer that sits beside existing training, eval, and benchmarking stack. It doesn't train, doesn't run kernels — it ingests results from whatever tooling you already use and owns the candidate set, evidence store, fitter, and decision report.

Three reference adapters live in `ac_harness/adapters/`, each runnable as `python -m ac_harness.adapters.<name>`:

| Adapter | Converts | Emits |
|---|---|---|
| `lm_eval` | lm-evaluation-harness JSON output | per-task `lmeval_*` scores + a `downstream_score` headline |
| `vllm_serving` | vLLM `benchmark_serving.py` JSON | `throughput_tps`, TTFT / TBT / ITL medians and p99s |
| `training_callback` | In-process buffer or CSV/JSONL training log | per-step `val_loss`, custom metrics, optional `downstream_score` |

Sample fixtures in `examples/adapter_fixtures/` show the exact native output shape each adapter expects. See `docs/integration_guide.md` for the input contract, metric-naming conventions, candidate-ID mapping, and an end-to-end walkthrough of a typical lab study.


## Repo layout

```
.
├── README.md
├── LICENSE                 Apache-2.0
├── pyproject.toml
├── ac_harness/             the package (ingest, planner, executor, fitter,
│                           evaluator, decision, store, benchmarks, cli)
├── docs/                   ac_core_boundary, workflows, schemas, decision_state
├── examples/               sample AC-Core outputs + imported results
└── tests/                  pytest suite (runs without GPUs)
```

See `docs/` for boundary, workflow, schema, and decision-state documentation.

## License

Apache-2.0.
