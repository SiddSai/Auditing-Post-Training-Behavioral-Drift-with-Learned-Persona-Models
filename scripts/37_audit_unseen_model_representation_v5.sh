#!/usr/bin/env bash
set -euo pipefail

# Phase 1--2: target-label-free anchor audit, then an exploratory LOMO state
# sweep.  LOMO is development for representation selection; do not quote it as
# the final generalization headline. The later LOLO run is confirmation.
targets='data/targets/final_initial_native_three_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v5.tsv'
wild='data/panels/wild_standardized_valid_v5.tsv'
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
anchor_obs='runs/final_standardized_v5/anchors/observations.jsonl'
native='runs/final_standardized_v5/targets/native_scorers'
root='runs/final_standardized_v5/unseen_model_v1'

for file in "$anchor_obs" "$native/ifeval_scores.jsonl" "$native/xstest_scores.jsonl" "$native/do_not_answer_longformer_scores.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 1; }
done
mkdir -p "$root"
persona-audit prepare-predictor-outcomes --targets "$targets" --ifeval "$native/ifeval_scores.jsonl" --xstest "$native/xstest_scores.jsonl" --do-not-answer "$native/do_not_answer_longformer_scores.jsonl" --output "$root/outcomes.jsonl"
persona-audit audit-anchor-panel --observations "$anchor_obs" --anchors "$anchors" --nodes "$official" --wild-nodes "$wild" --assistant-only --dimensions 4 --split-half-repeats 200 --output-dir "$root/anchor_audit"

run_one () {
  local method="$1"; local dimension="$2"; local out="$root/lomo_${method}_d${dimension}"
  persona-audit run-predictor --anchor-observations "$anchor_obs" --anchors "$anchors" --targets "$targets" --outcomes "$root/outcomes.jsonl" --nodes "$official" --wild-nodes "$wild" --analysis-panel native_leave_one_model_out --state-method "$method" --state-dimensions "$dimension" --output-dir "$out"
  persona-audit audit-predictor --predictions "$out/predictions.jsonl" --state-geometry "$out/state_geometry.jsonl" --output-dir "$out/audit" --bootstrap-replicates 1000
}

# Predeclared state sweep. PCA d4 remains the original baseline; Factor and
# fractional-response IRT test distinct measurement assumptions.
run_one pca 2
run_one pca 4
run_one pca 8
run_one factor 4
run_one soft_irt 2
run_one soft_irt 4
run_one soft_irt 8
persona-audit report-predictor --run-root "$root" --output "$root/representation_report.csv"
echo "Completed anchor audit and exploratory LOMO state sweep: $root"
