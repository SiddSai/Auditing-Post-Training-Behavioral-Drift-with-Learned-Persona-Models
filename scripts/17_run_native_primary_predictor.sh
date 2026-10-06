#!/usr/bin/env bash
set -euo pipefail

# Primary, interface-valid concurrent behavioral-prediction experiment.
#
# All 18 native-interface models contribute labeled development-prompt
# outcomes: six official OLMo Instruct/Think endpoints and twelve admitted
# OLMo-derived descendants.  Evaluation is exclusively on the frozen 150
# held-out prompts per family for those same models.  z_m is fit from anchors
# only; no target outcome enters the state representation.
TARGETS="data/targets/single_turn_v1.jsonl"
NATIVE="runs/targets_native_interface/v1/native_scorers"
ROOT="runs/predictor/v1/native_primary_prompt_holdout"

for file in "$NATIVE/ifeval_scores.jsonl" "$NATIVE/xstest_scores.jsonl" "$NATIVE/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file; run scripts/12 and scripts/13 first." >&2; exit 1; }
done

persona-audit prepare-predictor-outcomes \
  --targets "$TARGETS" \
  --ifeval "$NATIVE/ifeval_scores.jsonl" \
  --xstest "$NATIVE/xstest_scores.jsonl" \
  --do-not-answer "$NATIVE/do_not_answer_longformer_scores.jsonl" \
  --output "$ROOT/outcomes.jsonl"

# PCA is primary. Factor analysis is a predeclared representation robustness
# check, not a test-set-selected method.  d=2/4 keeps the state compact for
# this 18-model panel; d=8 is included as a higher-capacity sensitivity run.
for method in pca factor; do
  for dimensions in 2 4 8; do
    out="$ROOT/${method}_d${dimensions}"
    persona-audit run-predictor \
      --anchor-observations runs/anchors/v1/observations.jsonl \
      --anchors data/anchors/anthropic_persona_v1.jsonl \
      --targets "$TARGETS" \
      --outcomes "$ROOT/outcomes.jsonl" \
      --nodes manifests/nodes.tsv \
      --wild-nodes manifests/wild_candidates.tsv \
      --analysis-panel native_all_prompt_holdout \
      --state-method "$method" \
      --state-dimensions "$dimensions" \
      --output-dir "$out"
    persona-audit audit-predictor \
      --predictions "$out/predictions.jsonl" \
      --state-geometry "$out/state_geometry.jsonl" \
      --output-dir "$out/audit"
  done
done

echo "Wrote primary native prompt-held-out results under $ROOT"
