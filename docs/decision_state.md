# Decision state

The DecisionState is what the harness "knows right now." It is rebuilt from store contents on every call to `build_decision_state(store)` and persisted on every `ach decision-report` run.

## What it answers

```text
What have we measured?
What do we believe now?
Which candidates are currently promising?
Which uncertainty matters most?
What should we run next?
What result would change the recommendation?
```

## What it does NOT say

> This is the optimal architecture.

The report instead says:

> Given current evidence, these candidates are currently supported.
> These candidates remain uncertain.
> These measurements would most change the decision.

## Anatomy of a DecisionState

`observed_summary`
: per-candidate measurement counts, per-metric counts, which of the key buckets (`throughput`, `kernel_bandwidth`, `quality_loss`, `quality_task`) have at least one measurement.

`uncertainty_summary`
: missing key buckets per candidate, list of "prediction-only" candidates (have a prediction but no observed measurement), and "risky-unmeasured" candidates (AC-Core tagged risky/research AND no quality measurement yet).

`current_frontier`
: candidates on the observed Pareto front (throughput axis × quality axis) when measurements exist. Falls back to a prediction-only frontier when no measurements exist. Throughput axis picks the highest-priority available metric: `serving_throughput_tps` → `throughput_tps` → `decode_throughput_tps` → `decode_kv_bandwidth_gbps`. Quality axis prefers higher-is-better metrics (needle, MQAR, downstream, copy), then inverts `val_loss`.

`disagreements`
: the five §13.2 rules — predicted serving gain not observed, low-risk-but-quality-regression, perplexity vs long-context, MoE throughput vs load imbalance, state/hybrid serving vs recall.

`recommended_next_experiments`
: ranked ExperimentPlan IDs (populated when `decision-report` runs after `plan-next`).

`human_decisions_required`
: open `HumanDecision` records for the §14 review points.

## DecisionStateReport.md sections

1. Research question / decision
2. Candidate set
3. Current evidence table (mean ± n per metric, top-6 columns by frequency)
4. Current supported frontier (with kind ∈ {observed, prediction_only, mixed})
5. Uncertain candidates (missing buckets per candidate)
6. Disagreements / surprises
7. Fitted calibration / residuals available
8. Recommended next experiments
9. Human decision points
10. Export files for AC-Core

## When the harness asks for a human

The §14 review checkpoints:

- Before expensive runs
- When evaluator metrics disagree
- When fitted residuals are unstable
- When a prediction-only candidate is about to be treated as supported
- When imported results lack provenance
- When a calibration file would overwrite a validated calibration
- When a result contradicts the current compiler prior

These are stored as `HumanDecision` records; v1 surfaces them in §9 of the report and through the store API. A future revision will add interactive prompts (CLI confirmation, web UI, etc.).
