#!/usr/bin/env bash
set -euo pipefail

targets='data/targets/final_initial_native_three_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v4.tsv'
wild='manifests/wild_candidates.tsv'
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
anchor_obs='runs/final_standardized_v4/anchors/observations.jsonl'
native='runs/final_standardized_v4/targets/native_scorers'
root='runs/final_standardized_v4/predictor'

for file in "$anchor_obs" "$native/ifeval_scores.jsonl" "$native/xstest_scores.jsonl" "$native/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 1; }
done
persona-audit prepare-predictor-outcomes --targets "$targets" --ifeval "$native/ifeval_scores.jsonl" --xstest "$native/xstest_scores.jsonl" --do-not-answer "$native/do_not_answer_longformer_scores.jsonl" --output "$root/outcomes.jsonl"

run_one () {
  local panel="$1" method="$2" label="$3"
  local out="$root/$label"
  persona-audit run-predictor --anchor-observations "$anchor_obs" --anchors "$anchors" --targets "$targets" --outcomes "$root/outcomes.jsonl" --nodes "$official" --wild-nodes "$wild" --analysis-panel "$panel" --state-method "$method" --state-dimensions 4 --output-dir "$out"
  persona-audit audit-predictor --predictions "$out/predictions.jsonl" --state-geometry "$out/state_geometry.jsonl" --output-dir "$out/audit" --bootstrap-replicates 2000
}

run_one native_all_prompt_holdout pca primary_prompt_holdout_pca_d4
run_one native_all_prompt_holdout factor robustness_prompt_holdout_factor_d4
run_one native_leave_one_model_out pca leave_one_model_out_pca_d4
run_one native_leave_one_model_out factor robustness_leave_one_model_out_factor_d4
run_one native_leave_one_lineage_out pca leave_one_lineage_out_pca_d4
persona-audit report-predictor --run-root "$root" --output "$root/representation_report.csv"
echo "Predictor suite complete under $root"
