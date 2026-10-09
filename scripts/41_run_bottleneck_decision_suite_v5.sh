#!/usr/bin/env bash
set -euo pipefail

# CPU-only decision suite. It does not generate completions or modify the
# primary held-out result. See docs/BOTTLENECK_DECISION_SUITE.md for the
# preregistered interpretation of each diagnostic.
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v5.tsv'
wild='data/panels/wild_standardized_valid_v5.tsv'
observations='runs/final_standardized_v5/anchors/observations.jsonl'
outcomes='runs/final_standardized_v5/unseen_model_v1/outcomes.jsonl'
output='runs/final_standardized_v5/bottleneck_decision_suite'

for file in "$anchors" "$official" "$wild" "$observations" "$outcomes"; do
  [[ -f "$file" ]] || { echo "Missing $file. Restore the verified v5 archive first." >&2; exit 1; }
done

persona-audit run-bottleneck-decision-suite \
  --observations "$observations" \
  --anchors "$anchors" \
  --nodes "$official" \
  --wild-nodes "$wild" \
  --outcomes "$outcomes" \
  --dimensions 4 \
  --repeats 40 \
  --output-dir "$output"

echo "Decision suite complete: $output"
