#!/usr/bin/env bash
set -euo pipefail

run='runs/final_primary_v3/targets'
official='data/panels/official_plus_posttrain_trajectory_v3.tsv'
wild='data/panels/wild_anchor_compatible_v1.tsv'
interfaces='data/interfaces/native_interface_v3.jsonl'
gpu_ids="${GPU_IDS:-0 1}"
mkdir -p "$run/logs"
for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit target-worker \
    --nodes "$official" --wild-nodes "$wild" --targets data/targets/final_initial_native_three_v1.jsonl \
    --interfaces "$interfaces" --interface-renderings native_chat_template \
    --output-dir "$run" --cache-dir "$PWD/.hf-cache" --batch-size 128 > "$run/logs/worker-${gpu}.log" 2>&1 &
done
echo "Workers started on GPUs: $gpu_ids"
echo "Monitor: find $run/metadata -name '*.json' | wc -l  # expected 70"
