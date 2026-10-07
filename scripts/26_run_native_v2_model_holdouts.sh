#!/usr/bin/env bash
set -euo pipefail

# Proper unseen-model analyses over the 71-model native assistant panel.
# 1) LOMO: hold out each model independently (interpolation among nearby
#    checkpoints is allowed).
# 2) LOGO: hold out an entire official trajectory or external publisher group
#    (the stricter lineage/recipe generalization result).
targets='data/targets/single_turn_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v2.tsv'
wild='manifests/wild_candidates.tsv'
anchors='data/anchors/anthropic_persona_v1.jsonl'
anchor_obs='runs/anchors_native_interface/v2/observations.jsonl'
native='runs/targets_native_interface/v2/native_scorers'
root='runs/predictor/v2/native_model_holdouts'

for file in "$anchor_obs" "$native/ifeval_scores.jsonl" "$native/xstest_scores.jsonl" "$native/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file; complete scripts/20--23 first." >&2; exit 1; }
done

persona-audit prepare-predictor-outcomes \
  --targets "$targets" --ifeval "$native/ifeval_scores.jsonl" \
  --xstest "$native/xstest_scores.jsonl" --do-not-answer "$native/do_not_answer_longformer_scores.jsonl" \
  --output "$root/outcomes.jsonl"

# PCA d4 is the pre-specified compact primary state. Factor d4 is a direct
# representation robustness check. Higher dimensions are deferred until these
# truly held-out model analyses are inspected.
for panel in native_leave_one_model_out native_leave_one_lineage_out; do
  for method in pca factor; do
    out="$root/${panel}_${method}_d4"
    persona-audit run-predictor \
      --anchor-observations "$anchor_obs" --anchors "$anchors" \
      --targets "$targets" --outcomes "$root/outcomes.jsonl" \
      --nodes "$official" --wild-nodes "$wild" \
      --analysis-panel "$panel" --state-method "$method" --state-dimensions 4 \
      --output-dir "$out"
    persona-audit audit-predictor \
      --predictions "$out/predictions.jsonl" --state-geometry "$out/state_geometry.jsonl" \
      --output-dir "$out/audit" --bootstrap-replicates 2000
  done
done

persona-audit report-predictor --run-root "$root" --output "$root/model_holdout_representation_report.csv"
echo "Wrote native v2 leave-one-model-out and leave-one-lineage-out results under $root"
