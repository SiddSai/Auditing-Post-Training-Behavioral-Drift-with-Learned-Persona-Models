#!/usr/bin/env bash
set -euo pipefail

# Phase 3--4: frozen semantic prompt embeddings plus a low-rank bilinear
# state--prompt decoder. It is an explicitly stronger decoder, not a revision
# of the anchor state. Install once before this command:
#   pip install -e '.[analysis]'
targets='data/targets/final_initial_native_three_v1.jsonl'
official='data/panels/official_plus_posttrain_trajectory_v5.tsv'
wild='data/panels/wild_standardized_valid_v5.tsv'
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
anchor_obs='runs/final_standardized_v5/anchors/observations.jsonl'
root='runs/final_standardized_v5/unseen_model_v1'
outcomes="$root/outcomes.jsonl"
[[ -f "$outcomes" ]] || { echo "Run scripts/37_audit_unseen_model_representation_v5.sh first." >&2; exit 1; }

run_one () {
  local panel="$1"; local method="$2"; local label="$3"; local out="$root/$label"
  persona-audit run-predictor --anchor-observations "$anchor_obs" --anchors "$anchors" --targets "$targets" --outcomes "$outcomes" --nodes "$official" --wild-nodes "$wild" --analysis-panel "$panel" --state-method "$method" --state-dimensions 4 --prompt-representation sentence_transformer --prompt-embedding-model sentence-transformers/all-MiniLM-L6-v2 --prompt-dimensions 32 --bilinear-rank 2 --include-bilinear --output-dir "$out"
  persona-audit audit-predictor --predictions "$out/predictions.jsonl" --state-geometry "$out/state_geometry.jsonl" --output-dir "$out/audit" --bootstrap-replicates 2000
}

# The representation sweep selected these two candidates on exploratory LOMO:
# PCA d8 (strongest broad row-level result) and factor d4 (parsimonious state
# with the clearest incremental Do-Not-Answer signal beyond metadata).  LOLO
# is their correlated-lineage confirmation, not a model-selection criterion.
run_one native_leave_one_model_out pca contextual_lomo_pca_d8
run_one native_leave_one_model_out factor contextual_lomo_factor_d4
run_one native_leave_one_lineage_out pca contextual_lolo_pca_d8
run_one native_leave_one_lineage_out factor contextual_lolo_factor_d4
persona-audit report-predictor --run-root "$root" --output "$root/representation_report.csv"
echo "Completed contextual unseen-model decoder tests: $root"
