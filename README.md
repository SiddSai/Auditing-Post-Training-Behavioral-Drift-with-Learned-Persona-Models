# OLMo 3 behavioral-drift study panel

This directory freezes the initial model panel for the predictive behavioral
state project.  It deliberately separates a clean, documented training
trajectory from a later, heterogeneous intervention panel.

## Running the pipeline

### VM quickstart

After cloning the repository and activating a fresh Python 3.10–3.12 conda
environment, run these commands from the repository root:

```bash
bash scripts/01_install.sh
bash scripts/02_prepare_anchors.sh
bash scripts/03_preflight.sh
bash scripts/04_pilot.sh
```

`04_pilot.sh` is deliberately a three-model smoke test, not the expensive full
panel. Inspect its `runs/pilot_v1/*.metadata.json` artifacts before launching
the 58-node sweep. After the full sweep, run `bash scripts/05_fit_full_state.sh`.
It creates the primary 46-official-node PCA state, an all-panel descriptive
map, and a base-trajectory-only reference ablation. The target-domain/rubric
predictor is the next research module.

Install the package with inference dependencies on the GPU machine:

```bash
pip install -e '.[inference]'

# 1. Fetch Anthropic's official persona corpus at our pinned Git SHA.
persona-audit snapshot-anthropic-persona \
  --output-dir data/snapshots/anthropic-evals

# 2. Convert its actual JSONL schema to a canonical raw-prompt pool.
persona-audit import-anthropic-persona \
  --source-dir data/snapshots/anthropic-evals \
  --source-revision 84fcc677e52e1902d696c32cd1a6b663e70d3993 \
  --output data/anchors/anthropic_persona_pool.jsonl

# 3. Freeze a deterministic, balanced 300-anchor battery.
persona-audit sample-anchors --input data/anchors/anthropic_persona_pool.jsonl \
  --output data/anchors/anthropic_persona_v1.jsonl --size 300 --seed 20261005

# 4. Validate exact node and anchor inputs before launching GPUs.
persona-audit validate --nodes manifests/nodes.tsv \
  --anchors data/anchors/anthropic_persona_v1.jsonl
```

The worker never uses a model-specific chat template. It loads one pinned
checkpoint at a time, scores the raw anchor prompts in large batches, writes
an atomic per-node result, and then advances to the next available node.

```bash
# Start one process per GPU with a shared output and cache directory.
CUDA_VISIBLE_DEVICES=0 persona-audit anchor-worker \
  --nodes manifests/nodes.tsv --anchors data/anchors/anthropic_persona_v1.jsonl \
  --output-dir runs/anchors/v1 --cache-dir /local-scratch/hf-cache

# Add the 12 admitted observational descendants for the 58-node panel.
CUDA_VISIBLE_DEVICES=0 persona-audit anchor-worker \
  --nodes manifests/nodes.tsv --wild-nodes manifests/wild_candidates.tsv \
  --anchors data/anchors/anthropic_persona_v1.jsonl \
  --output-dir runs/anchors/v1 --cache-dir /local-scratch/hf-cache

# Merge completed worker artifacts and derive a fold-safe z_m state.
persona-audit collect-observations --run-dir runs/anchors/v1 \
  --output runs/anchors/v1/observations.jsonl
persona-audit fit-state --observations runs/anchors/v1/observations.jsonl \
  --anchors data/anchors/anthropic_persona_v1.jsonl \
  --fit-nodes splits/development_nodes.json \
  --transform-nodes splits/all_nodes.json \
  --dimensions 8 --output-dir runs/state/v1
```

See [the architecture](docs/ARCHITECTURE.md) for the data contracts and
leakage controls.

### Next: frozen single-turn target panel

After the anchor sweep and `z_m` artifacts are present, build and run the
separate target panel:

```bash
bash scripts/06_prepare_targets.sh
bash scripts/07_run_targets_4xh100.sh
# Once all 58 nodes have metadata:
bash scripts/08_collect_and_score_targets.sh
```

This fetches the five source-pinned target datasets (SycophancyEval, XSTest,
Do-Not-Answer, BOLD, and TruthfulQA), freezes 300 development plus 150 held-out
evaluation prompts per dataset, then generates raw greedy completions. See
[the target-panel protocol](docs/TARGET_PANEL.md) for what each source's
published scoring procedure supports and why judge-required metrics are kept
separate rather than replaced with ad hoc heuristics.

The implementation deliberately uses vLLM's offline `LLM.generate` API rather
than chat completion APIs: that API does not apply a chat template
automatically. Candidate answers are scored as requested next-token log
probabilities, not sampled generations. The runner rejects multi-token answer
candidates instead of quietly substituting an approximation. This makes the
initial state exactly reproducible for every OLMo-derived checkpoint with the
shared tokenizer.

To launch an eight-GPU pass, start the same `anchor-worker` command in eight
processes with a distinct `CUDA_VISIBLE_DEVICES` value and the same output and
cache directories. The filesystem claim protocol distributes nodes. A completed
node is reused only if its checkpoint SHA, anchor-manifest hash, engine
configuration, and observation hash all match.

### Faithful source data; deliberate cross-model prompt protocol

Anthropic's official persona repository supplies behavior-specific JSONL files
whose `question`, `answer_matching_behavior`, `answer_not_matching_behavior`,
and `label_confidence` fields define each observation. Its original evaluation
uses an Anthropic-model-specific `<EOT> … Human … Assistant` wrapper. Our
importer retains the source **question** and ` Yes`/` No` labels verbatim but
deliberately omits that wrapper—along with every system prompt and chat
template—so OLMo checkpoints are compared using one fixed raw-prompt protocol.
The importer writes this intentional deviation, every source-file hash, and
the full upstream commit SHA to its provenance artifact.

## Primary panel

`manifests/nodes.tsv` contains 46 immutable model states:

- 40 states from the official OLMo 3 7B base-training trajectory: 22 stage-1,
  9 stage-2, 8 stage-3, and the released `main` base model.
- 6 official post-training releases: SFT, DPO, and final RLVR endpoints for
  both Instruct and Think.

Pass `--wild-nodes manifests/wild_candidates.tsv` to add the 12 admitted wild
descendants. They get a stable `wild--<publisher>--<model>` run identifier;
the original Hub ID and immutable revision remain in each worker artifact.

Every row uses a Hugging Face commit SHA resolved on 2026-10-05, rather than
a mutable branch name. `revision` is retained solely as a human-readable
checkpoint label. The temporal grid is intentionally denser early in
pretraining and around training-stage transitions; it is not intended to make
the checkpoints look like independent training runs.

## Lineage

`manifests/edges.tsv` is a declared lineage graph. The exact parent SHA for
the public post-training endpoints has not yet been verified against the
training logs, so those edges are labelled `declared_release_lineage` rather
than `weight_verified`. Do not use an edge for a causal parent-to-child claim
until it is weight-verified.

## Wild-intervention candidates

`manifests/wild_candidates.tsv` is the secondary **wild observational panel**.
The current candidates passed repository-level parent, SHA, full-weight, and
chat-template checks; see `manifests/wild_provenance_audit.md`. They are
admitted for external-validity tests, not causal intervention analyses.

A candidate needs all of the following before it can be promoted to a causal
intervention edge:

1. exact base-model/parent declaration and immutable SHA;
2. loadable full weights or a reproducible adapter;
3. documented data, objective, and training recipe; and
4. a known chat template and license compatible with our evaluation.

Quantizations, format conversions, and undocumented merges are excluded.

## Sources

- Official training scripts, stages, and released post-training lineage:
  <https://github.com/allenai/Olmo-core/tree/main/src/scripts/official/OLMo3>
- Full official base-checkpoint list:
  <https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv>
- Base model revisions: <https://huggingface.co/allenai/Olmo-3-1025-7B/branches>
- OLMo 3 model collection: <https://huggingface.co/collections/allenai/olmo-3>
- Closest related checkpoint sweep (16-checkpoint subset):
  <https://github.com/epfl-dlab/pretraining_persona>
- vLLM offline inference and candidate-token logprob APIs:
  <https://docs.vllm.ai/en/stable/getting_started/quickstart/>,
  <https://docs.vllm.ai/en/stable/api/vllm/sampling_params/>
- Hugging Face immutable snapshot download API:
  <https://huggingface.co/docs/huggingface_hub/en/guides/download>
- Anthropic's official persona data and field definitions:
  <https://github.com/anthropics/evals/tree/main/persona>
