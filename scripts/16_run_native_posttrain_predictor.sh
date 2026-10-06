#!/usr/bin/env bash
set -euo pipefail

# Interface-valid post-training analysis. These completions already exist in
# runs/targets_native_interface/v1: each chat/think checkpoint was rendered
# through its pinned tokenizer chat template with one user turn and no
# researcher-injected system prompt. We intentionally do not mix raw base
# completions into this panel.
TARGETS="data/targets/single_turn_v1.jsonl"
NATIVE="runs/targets_native_interface/v1/native_scorers"
ROOT="runs/predictor/v1/native_posttrain"

for file in "$NATIVE/ifeval_scores.jsonl" "$NATIVE/xstest_scores.jsonl" "$NATIVE/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file; run scripts/12 and scripts/13 first." >&2; exit 1; }
done

persona-audit prepare-predictor-outcomes \
  --targets "$TARGETS" \
  --ifeval "$NATIVE/ifeval_scores.jsonl" \
  --xstest "$NATIVE/xstest_scores.jsonl" \
  --do-not-answer "$NATIVE/do_not_answer_longformer_scores.jsonl" \
  --output "$ROOT/outcomes.jsonl"

# Six official Instruct/Think endpoints fit the unsupervised state; twelve
# independent OLMo-Instruct descendants form the held-out external panel.
# Dimensions >4 would be poorly identified with six state-fitting models.
for method in pca factor; do
  for dimensions in 2 4; do
    out="$ROOT/${method}_d${dimensions}"
    persona-audit run-predictor \
      --anchor-observations runs/anchors/v1/observations.jsonl \
      --anchors data/anchors/anthropic_persona_v1.jsonl \
      --targets "$TARGETS" \
      --outcomes "$ROOT/outcomes.jsonl" \
      --nodes manifests/nodes.tsv \
      --wild-nodes manifests/wild_candidates.tsv \
      --analysis-panel native_posttrain \
      --state-method "$method" \
      --state-dimensions "$dimensions" \
      --output-dir "$out"
    persona-audit audit-predictor \
      --predictions "$out/predictions.jsonl" \
      --state-geometry "$out/state_geometry.jsonl" \
      --output-dir "$out/audit"
  done
done

echo "Wrote interface-valid post-training results under $ROOT"
