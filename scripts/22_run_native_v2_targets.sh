#!/usr/bin/env bash
set -euo pipefail

# Requires scripts/20_prepare_native_v2_panel.sh and the frozen target panel.
# Greedy generation, one raw benchmark prompt in a single native user turn,
# with no researcher-authored system message.
run='runs/targets_native_interface/v2'
gpu_ids="${GPU_IDS:-0 1 2 3}"
mkdir -p "$run/logs"

for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit target-worker \
    --nodes data/panels/official_plus_posttrain_trajectory_v2.tsv \
    --wild-nodes manifests/wild_candidates.tsv \
    --targets data/targets/single_turn_v1.jsonl \
    --interfaces data/interfaces/native_interface_v2.jsonl \
    --interface-renderings native_chat_template \
    --output-dir "$run" --cache-dir "$PWD/.hf-cache" --batch-size 128 \
    > "$run/logs/worker-${gpu}.log" 2>&1 &
done

echo "Native-target workers started on GPUs: $gpu_ids"
echo "Expected completions: 71"
echo "Monitor: find $run/metadata -name '*.json' | wc -l"
