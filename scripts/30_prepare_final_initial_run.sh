#!/usr/bin/env bash
set -euo pipefail

# Reconstruct every frozen input from source revisions before the one final
# initial-result collection. This script performs no model inference.
bash scripts/02_prepare_anchors.sh
bash scripts/06_prepare_targets.sh
bash scripts/20_prepare_native_v2_panel.sh

anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
persona-audit prepare-direct-answer-anchors \
  --input data/anchors/anthropic_persona_v1.jsonl --output "$anchors"
persona-audit filter-target-families \
  --input data/targets/single_turn_v1.jsonl \
  --output data/targets/final_initial_native_three_v1.jsonl \
  --families ifeval xstest do_not_answer
persona-audit audit-target-anchor-disjointness \
  --targets data/targets/final_initial_native_three_v1.jsonl --anchors "$anchors" \
  --output data/targets/single_turn_v1.direct_answer_anchor_disjointness.json
persona-audit audit-direct-answer-tokenizers \
  --nodes data/panels/official_plus_posttrain_trajectory_v2.tsv \
  --wild-nodes manifests/wild_candidates.tsv --anchors "$anchors" \
  --interfaces data/interfaces/native_interface_v2.jsonl \
  --output data/audits/direct_answer_tokenizer_preflight_v1.json

echo 'Frozen final-initial inputs are prepared and every assistant tokenizer passed preflight.'
