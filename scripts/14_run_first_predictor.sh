#!/usr/bin/env bash
set -euo pipefail

# First result: raw-prompt panel; three entirely source-local outcome families,
# plus the already-recorded SycophancyEval modern-judge adaptation when present.
# The frozen target manifest is intentionally required: it supplies the actual
# raw prompt text and development/evaluation assignment.
TARGETS="data/targets/single_turn_v1.jsonl"
OUTCOMES="runs/predictor/v1/outcomes.jsonl"
OUTDIR="runs/predictor/v1/raw_primary"

if [[ ! -f "$TARGETS" ]]; then
  echo "Missing $TARGETS. Copy the frozen target manifest from the generation VM before running the predictor." >&2
  exit 1
fi

persona-audit prepare-predictor-outcomes \
  --targets "$TARGETS" \
  --ifeval runs/targets/v1/native_scorers/ifeval_scores.jsonl \
  --xstest runs/targets/v1/native_scorers/xstest_scores.jsonl \
  --do-not-answer runs/targets/v1/native_scorers/do_not_answer_longformer_scores.jsonl \
  --sycophancy runs/targets/v1/native_scorers/sycophancy_scores.jsonl \
  --output "$OUTCOMES"

persona-audit run-predictor \
  --anchor-observations runs/anchors/v1/observations.jsonl \
  --anchors data/anchors/anthropic_persona_v1.jsonl \
  --targets "$TARGETS" \
  --outcomes "$OUTCOMES" \
  --nodes manifests/nodes.tsv \
  --wild-nodes manifests/wild_candidates.tsv \
  --output-dir "$OUTDIR"

echo "Wrote $OUTDIR/{metrics.csv,predictions.jsonl,metadata.json}"
