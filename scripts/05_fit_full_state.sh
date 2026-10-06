#!/usr/bin/env bash
set -euo pipefail

run_dir='runs/anchors/v1'
anchors='data/anchors/anthropic_persona_v1.jsonl'
splits_dir='runs/state/v1/splits'

persona-audit collect-observations --run-dir "$run_dir" \
  --output "$run_dir/observations.jsonl"
persona-audit write-panel-splits --nodes manifests/nodes.tsv \
  --wild-nodes manifests/wild_candidates.tsv --output-dir "$splits_dir"

# Primary analysis map: the 46 documented official OLMo states define the PCA
# geometry; wild descendants are projected in without defining its axes.
persona-audit fit-state --observations "$run_dir/observations.jsonl" \
  --anchors "$anchors" --fit-nodes "$splits_dir/official_panel_nodes.json" \
  --transform-nodes "$splits_dir/all_panel_nodes.json" \
  --method pca --dimensions 8 --feature behavior_logit_margin \
  --output-dir runs/state/v1/official_panel_pca_d8

# Primary descriptive map: all 58 observed models define the PCA geometry.
persona-audit fit-state --observations "$run_dir/observations.jsonl" \
  --anchors "$anchors" --fit-nodes "$splits_dir/all_panel_nodes.json" \
  --transform-nodes "$splits_dir/all_panel_nodes.json" \
  --method pca --dimensions 8 --feature behavior_logit_margin \
  --output-dir runs/state/v1/all_panel_pca_d8

# Pretraining-only reference ablation: axes fit solely on the 40 base checkpoints.
persona-audit fit-state --observations "$run_dir/observations.jsonl" \
  --anchors "$anchors" --fit-nodes "$splits_dir/base_trajectory_nodes.json" \
  --transform-nodes "$splits_dir/all_panel_nodes.json" \
  --method pca --dimensions 8 --feature behavior_logit_margin \
  --output-dir runs/state/v1/base_reference_pca_d8
