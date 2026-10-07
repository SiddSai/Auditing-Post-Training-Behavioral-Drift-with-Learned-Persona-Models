#!/usr/bin/env bash
set -euo pipefail

# Exploratory external-transfer stress test. The 12 wild descendants have
# already been used in the all-model prompt-held-out analysis, so this is NOT
# a fresh confirmatory holdout. It asks the complementary question: can a
# state-to-behavior relation fitted only on six official post-training models
# transfer to those descendants? PCA-4 is fixed here for compactness; no
# result from this script should select a representation.
TARGETS="data/targets/single_turn_v1.jsonl"
NATIVE="runs/targets_native_interface/v1/native_scorers"
ROOT="runs/predictor/v1/exploratory_official_to_wild_pca_d4"

for file in "$NATIVE/ifeval_scores.jsonl" "$NATIVE/xstest_scores.jsonl" "$NATIVE/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file; run scripts/12 and scripts/13 first." >&2; exit 1; }
done

persona-audit prepare-predictor-outcomes \
  --targets "$TARGETS" \
  --ifeval "$NATIVE/ifeval_scores.jsonl" \
  --xstest "$NATIVE/xstest_scores.jsonl" \
  --do-not-answer "$NATIVE/do_not_answer_longformer_scores.jsonl" \
  --output "$ROOT/outcomes.jsonl"

persona-audit run-predictor \
  --anchor-observations runs/anchors/v1/observations.jsonl \
  --anchors data/anchors/anthropic_persona_v1.jsonl \
  --targets "$TARGETS" \
  --outcomes "$ROOT/outcomes.jsonl" \
  --nodes manifests/nodes.tsv \
  --wild-nodes manifests/wild_candidates.tsv \
  --analysis-panel native_posttrain \
  --state-method pca \
  --state-dimensions 4 \
  --output-dir "$ROOT"

persona-audit audit-predictor \
  --predictions "$ROOT/predictions.jsonl" \
  --state-geometry "$ROOT/state_geometry.jsonl" \
  --output-dir "$ROOT/audit" \
  --bootstrap-replicates 2000

echo "Wrote exploratory—not confirmatory—official-to-wild transfer results under $ROOT"
