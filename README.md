# OLMo 3 behavioral-drift study panel

This directory freezes the initial model panel for the predictive behavioral
state project.  It deliberately separates a clean, documented training
trajectory from a later, heterogeneous intervention panel.

## Running the pipeline

Install the package with inference dependencies on the GPU machine:

```bash
pip install -e '.[inference]'
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
