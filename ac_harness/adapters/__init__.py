# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Adapters: client-side shims that convert a lab's existing tool output
(lm-evaluation-harness, vLLM benchmark_serving, a training loop's eval
callback, ...) into the JSON shape `ach import-results` consumes.

These are deliberately small and stateless so they can be vendored into
a lab's existing recipe without pulling AC-Harness into the runtime
environment. Each module is runnable as both a library function and a
script:

    python -m ac_harness.adapters.lm_eval --candidate-id X --input Y --out Z
    python -m ac_harness.adapters.vllm_serving --candidate-id X --input Y --out Z
    python -m ac_harness.adapters.training_callback --help

The output of every adapter has the same shape:

    {
      "benchmark_type": "<eval | serving_bench | training | kernel_microbench>",
      "hardware_id":   "<e.g. h100_sxm>",
      "runtime":       "<e.g. lm_eval_harness, vllm, torchtitan>",
      "workload_id":   "<lab-defined string>",
      "notes":         "<free text>",
      "results": [
        {
          "candidate_id": "<must match a CandidateSet entry>",
          "metrics":      {"<metric_name>": <float>, ...},
          "units":        {"<metric_name>": "<unit string>", ...},
          "seed":         <int | null>,
          "provenance":   "<free-text source description>"
        },
        ...
      ]
    }
"""
