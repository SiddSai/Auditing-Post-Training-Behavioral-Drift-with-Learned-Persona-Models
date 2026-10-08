#!/usr/bin/env bash
set -euo pipefail

# Phase 5: exploit the already-generated OLMo post-training trajectories.
# This is observed adjacent-checkpoint drift, not a claimed intervention or
# prospective forecast.
official='data/panels/official_plus_posttrain_trajectory_v5.tsv'
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
anchor_obs='runs/final_standardized_v5/anchors/observations.jsonl'
edges='manifests/posttrain_trajectory_edges.tsv'
outcomes='runs/final_standardized_v5/unseen_model_v1/outcomes.jsonl'
root='runs/final_standardized_v5/unseen_model_v1/trajectory'
[[ -f "$outcomes" ]] || { echo "Run scripts/37_audit_unseen_model_representation_v5.sh first." >&2; exit 1; }
persona-audit analyze-trajectory-drift --observations "$anchor_obs" --anchors "$anchors" --nodes "$official" --edges "$edges" --outcomes "$outcomes" --state-method pca --state-dimensions 4 --output-dir "$root/pca_d4"
persona-audit analyze-trajectory-drift --observations "$anchor_obs" --anchors "$anchors" --nodes "$official" --edges "$edges" --outcomes "$outcomes" --state-method soft_irt --state-dimensions 4 --output-dir "$root/soft_irt_d4"
echo "Completed observed checkpoint-drift analyses: $root"
