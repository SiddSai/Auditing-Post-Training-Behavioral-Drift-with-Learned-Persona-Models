#!/usr/bin/env bash
set -euo pipefail

# External generalization stress test. States and behavior predictors are fit
# on 50 official post-training states, then evaluated on 21 independently
# released OLMo descendants. This is the model-holdout claim; it has no
# same-model development-rate calibration feature.
targets='data/targets/single_turn_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v2.tsv'
wild='manifests/wild_candidates.tsv'
anchors='data/anchors/anthropic_persona_v1.jsonl'
anchor_obs='runs/anchors_native_interface/v2/observations.jsonl'
native='runs/targets_native_interface/v2/native_scorers'
root='runs/predictor/v2/native_official_to_external'

for file in "$anchor_obs" "$native/ifeval_scores.jsonl" "$native/xstest_scores.jsonl" "$native/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file; complete scripts/20--23 first." >&2; exit 1; }
done

persona-audit prepare-predictor-outcomes \
  --targets "$targets" --ifeval "$native/ifeval_scores.jsonl" \
  --xstest "$native/xstest_scores.jsonl" --do-not-answer "$native/do_not_answer_longformer_scores.jsonl" \
  --output "$root/outcomes.jsonl"

for method in pca factor; do
  for dimensions in 2 4 8; do
    out="$root/${method}_d${dimensions}"
    persona-audit run-predictor \
      --anchor-observations "$anchor_obs" --anchors "$anchors" \
      --targets "$targets" --outcomes "$root/outcomes.jsonl" \
      --nodes "$official" --wild-nodes "$wild" \
      --analysis-panel native_posttrain \
      --state-method "$method" --state-dimensions "$dimensions" \
      --output-dir "$out"
    persona-audit audit-predictor \
      --predictions "$out/predictions.jsonl" --state-geometry "$out/state_geometry.jsonl" \
      --output-dir "$out/audit" --bootstrap-replicates 2000
  done
done

persona-audit report-predictor --run-root "$root" --output "$root/external_representation_report.csv"
echo "Wrote official-to-external native v2 predictor results under $root"
