# OLMo 3 behavioral-drift study panel

This directory freezes the initial model panel for the predictive behavioral
state project.  It deliberately separates a clean, documented training
trajectory from a later, heterogeneous intervention panel.

## Primary panel

`manifests/nodes.tsv` contains 46 immutable model states:

- 40 states from the official OLMo 3 7B base-training trajectory: 22 stage-1,
  9 stage-2, 8 stage-3, and the released `main` base model.
- 6 official post-training releases: SFT, DPO, and final RLVR endpoints for
  both Instruct and Think.

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
