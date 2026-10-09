#!/usr/bin/env bash
set -euo pipefail

# Purely descriptive companion to the held-out predictor. This intentionally
# fits one common PCA across every measured assistant model, so its axes are
# suitable for plots. It is never a substitute for the fold-refit states used
# in leave-one-model-out or leave-one-lineage-out evaluation.
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v5.tsv'
wild='data/panels/wild_standardized_valid_v5.tsv'
edges='manifests/posttrain_trajectory_edges.tsv'
observations='runs/final_standardized_v5/anchors/observations.jsonl'
outcomes='runs/final_standardized_v5/unseen_model_v1/outcomes.jsonl'
output='runs/final_standardized_v5/descriptive_state_score_atlas'

for file in "$anchors" "$official" "$wild" "$edges" "$observations" "$outcomes"; do
  [[ -f "$file" ]] || { echo "Missing $file. Restore frozen v5 inputs from the verified archive first." >&2; exit 1; }
done

persona-audit build-state-score-atlas \
  --observations "$observations" \
  --anchors "$anchors" \
  --nodes "$official" \
  --wild-nodes "$wild" \
  --outcomes "$outcomes" \
  --edges "$edges" \
  --dimensions 4 \
  --clusters 4 \
  --output-dir "$output"

echo "Wrote descriptive atlas to $output"
