# Code architecture

## Core data flow

```text
source snapshots + manifests
        ↓
frozen anchor manifest
        ↓
vLLM checkpoint workers
        ↓
anchor observations r[m, a]
        ↓
fit state encoder on training nodes only
        ↓
z[m] = E(r[m])
        ↓
target observations y[m, x]
        ↓
behavior and drift predictors
        ↓
held-out node/domain evaluation
```

`m` identifies a pinned model node and `a` an anchor item. `x` identifies a
target prompt/conversation. Nothing downstream may silently change either a
model revision or an anchor/target item.

## First deliverable: reproducible behavioral state extraction

The first executable milestone is deliberately modest:

```bash
persona-audit run-anchors --nodes manifests/nodes.tsv \
  --anchors data/anchors/anthropic_persona_v1.jsonl \
  --output runs/anchors/v1

persona-audit fit-state --anchor-run runs/anchors/v1 \
  --train-split splits/core_development.json \
  --method pca --dimensions 2,4,8,16 \
  --output runs/state/v1
```

The output is an immutable state artifact containing:

- the exact training-node IDs;
- hashes of the node and anchor manifests;
- feature scaling fitted only on training nodes;
- PCA/factor-model parameters for each requested dimension;
- `z_m` for transformable nodes; and
- software/configuration versions.

The initial encoder is PCA or factor analysis, not a learned neural network.
This is the scientifically conservative baseline and makes each state axis
inspectable. A supervised/learned encoder is a later ablation.

## Package layout

```text
src/persona_audit/
  manifests/       parse and validate nodes, edges, prompts, and source hashes
  sources/         snapshot public source data and record commits/licenses
  anchors/         canonical raw-prompt rendering and Yes/No log-prob scoring
  inference/       resumable one-checkpoint-per-GPU vLLM workers
  storage/         schemas, atomic writes, run metadata, and result readers
  state/           response matrix construction, scaling, PCA/factors, z_m export
  targets/         target protocol runners and rubric-ready observation schemas
  predictors/      chronology, capability, PCA, bilinear, and delta baselines
  evaluation/      node/domain splits, calibration, bootstrap intervals, reports
  cli.py           command-line entry point

configs/
  anchors/
  inference/
  state/
  predictors/
  splits/

tests/
```

## Data contracts

### Anchor manifest

One JSONL record per frozen item:

```json
{
  "anchor_id": "anthropic.persona.sycophancy.000123",
  "source": "anthropic-evals",
  "source_revision": "<commit>",
  "prompt_raw": "...",
  "candidates": [" Yes", " No"],
  "behavior_consistent_candidate": " Yes",
  "label_confidence": 0.97,
  "family": "sycophancy"
}
```

### Anchor observation

One row per `(model_node, anchor_id)`:

```text
node_id, anchor_id, model_sha, anchor_manifest_sha,
logp_yes, logp_no, logit_margin, behavior_probability,
engine_config_sha, timestamp
```

The primary feature is the two-candidate normalized probability of the
behavior-consistent answer. Keep both raw candidate log-probabilities so the
scoring choice can be audited later.

### State artifact

```text
state_run/
  metadata.json
  scaler.joblib
  encoder_d8.joblib
  states_d8.parquet
  fit_nodes.json
```

`states_d8.parquet` has one row per node and columns `z_00` through `z_07`.

### Target observation

```text
node_id, target_id, target_family, history_raw, prompt_raw,
protocol_id, response_raw, outcome_type, outcome_value,
judge_id, decode_config_sha
```

## vLLM worker model

The coordinator creates a durable work queue from the node manifest. Each of
eight independent workers claims one node, downloads that node's exact Hub SHA
to local scratch, runs a large batched anchor pass, writes output atomically,
and releases the model before claiming another node. It must be safe to rerun:
completed `(node_id, anchor_id)` observations are skipped only after manifest
and inference-config hashes match.

The coordinator never treats a mutable Hugging Face branch as an input. It
loads the SHA in `nodes.tsv` or `wild_candidates.tsv`.

## Predictor sequence

Build in this order:

1. Chronology/tokens and parent-score baselines.
2. Ridge models from `z_m` to aggregate target metrics.
3. Frozen prompt embeddings plus a low-rank bilinear predictor:
   `M(history, prompt, z_m) -> outcome distribution`.
4. Delta version:
   `M_delta(z_parent, z_child - z_parent, history, prompt) -> delta outcome`.
5. Target-side calibration/refit baseline and uncertainty calibration.

All scaler, encoder, and predictor fitting happens inside the relevant outer
cross-validation fold. Test-node target labels are never used to choose state
dimensions, anchors, or regularisation.

## Immediate code order

1. Create Python package, schemas, and manifest validator.
2. Create source-snapshot command for the Anthropic anchor corpus.
3. Create the frozen 300-item anchor sampler and its manifest hash.
4. Implement a single-node local vLLM anchor runner.
5. Add the eight-worker resumable coordinator.
6. Implement `fit-state` and export `z_m`.
7. Add a smoke test using two local/mock nodes and ten anchors.
