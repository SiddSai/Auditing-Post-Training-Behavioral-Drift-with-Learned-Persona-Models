#!/usr/bin/env bash
set -euo pipefail

run='runs/final_primary_v3/anchors'
official='data/panels/official_plus_posttrain_trajectory_v3.tsv'
wild='data/panels/wild_anchor_compatible_v1.tsv'
interfaces='data/interfaces/native_interface_v3.jsonl'
gpu_ids="${GPU_IDS:-0 1}"
mkdir -p "$run/logs"
for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit anchor-worker \
    --nodes "$official" --wild-nodes "$wild" \
    --anchors data/anchors/anthropic_persona_direct_answer_v1.jsonl \
    --interfaces "$interfaces" --interface-renderings native_chat_template \
    --anchor-protocol native_think_then_answer --output-dir "$run" --cache-dir "$PWD/.hf-cache" \
    --batch-size 128 --think-max-tokens 2048 > "$run/logs/worker-${gpu}.log" 2>&1 &
done
wait
[[ "$(find "$run/metadata" -name '*.json' | wc -l | tr -d ' ')" == 70 ]] || { echo 'Full anchor run is incomplete; inspect logs before retrying.' >&2; exit 1; }
persona-audit collect-observations --run-dir "$run" --output "$run/observations.jsonl"
persona-audit validate-anchor-measurement --observations "$run/observations.jsonl" --output "$run/measurement_gate.csv" --min-node-median-mass 1e-4
[[ "$(wc -l < "$run/observations.jsonl")" == 21000 ]] || { echo 'Expected 21,000 anchor rows.' >&2; exit 1; }
echo 'Full anchor-compatible primary run passed its measurement gate.'
