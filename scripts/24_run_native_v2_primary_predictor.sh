#!/usr/bin/env bash
set -euo pipefail

# Primary concurrent behavioral-prediction result.  All 71 native-interface
# models are known during fitting; their 300 development prompts fit outcome
# predictors and their disjoint 150 evaluation prompts are the test set.
# z_m is fit only from the 300 rendered persona anchors.
targets='data/targets/single_turn_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v2.tsv'
wild='manifests/wild_candidates.tsv'
anchors='data/anchors/anthropic_persona_v1.jsonl'
anchor_obs='runs/anchors_native_interface/v2/observations.jsonl'
native='runs/targets_native_interface/v2/native_scorers'
root='runs/predictor/v2/native_primary_prompt_holdout'

for file in "$anchor_obs" "$native/ifeval_scores.jsonl" "$native/xstest_scores.jsonl" "$native/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file; complete scripts/20--23 first." >&2; exit 1; }
done

persona-audit prepare-predictor-outcomes \
  --targets "$targets" --ifeval "$native/ifeval_scores.jsonl" \
  --xstest "$native/xstest_scores.jsonl" --do-not-answer "$native/do_not_answer_longformer_scores.jsonl" \
  --output "$root/outcomes.jsonl"

# PCA is primary. Factor analysis and dimension choices are representation
# robustness checks, not choices selected on held-out target prompts.
for method in pca factor; do
  for dimensions in 2 4 8; do
    out="$root/${method}_d${dimensions}"
    persona-audit run-predictor \
      --anchor-observations "$anchor_obs" --anchors "$anchors" \
      --targets "$targets" --outcomes "$root/outcomes.jsonl" \
      --nodes "$official" --wild-nodes "$wild" \
      --analysis-panel native_all_prompt_holdout \
      --state-method "$method" --state-dimensions "$dimensions" \
      --output-dir "$out"
    persona-audit audit-predictor \
      --predictions "$out/predictions.jsonl" --state-geometry "$out/state_geometry.jsonl" \
      --output-dir "$out/audit" --bootstrap-replicates 2000
  done
done

persona-audit report-predictor --run-root "$root" --output "$root/primary_representation_report.csv"
echo "Wrote 71-model native v2 prompt-held-out predictor results under $root"
