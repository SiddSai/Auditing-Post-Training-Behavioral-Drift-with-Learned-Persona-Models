#!/usr/bin/env bash
set -euo pipefail

# Sole full anchor collection after the native Think answer-boundary pilot.
# A native Think model supplies its own reasoning trace; every other assistant
# model is still measured directly at its native assistant boundary.
run='runs/final_initial_v2/anchors'
gpu_ids="${GPU_IDS:-0 1}"
mkdir -p "$run/logs"
for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit anchor-worker \
    --nodes data/panels/official_plus_posttrain_trajectory_v2.tsv \
    --wild-nodes manifests/wild_candidates.tsv \
    --anchors data/anchors/anthropic_persona_direct_answer_v1.jsonl \
    --interfaces data/interfaces/native_interface_v2.jsonl \
    --interface-renderings native_chat_template --anchor-protocol native_think_then_answer \
    --output-dir "$run" --cache-dir "$PWD/.hf-cache" --batch-size 128 --think-max-tokens 512 \
    > "$run/logs/worker-${gpu}.log" 2>&1 &
done
wait
[[ "$(find "$run/metadata" -name '*.json' | wc -l | tr -d ' ')" == 71 ]] || { echo 'Full anchor run is incomplete; inspect logs before retrying.' >&2; exit 1; }
persona-audit collect-observations --run-dir "$run" --output "$run/observations.jsonl"
persona-audit validate-anchor-measurement --observations "$run/observations.jsonl" \
  --output "$run/measurement_gate.csv" --min-node-median-mass 1e-4
[[ "$(wc -l < "$run/observations.jsonl")" == 21300 ]] || { echo 'Expected 21,300 anchor rows.' >&2; exit 1; }
echo 'Full native-Think-aware anchor run passed its mass gate.'
