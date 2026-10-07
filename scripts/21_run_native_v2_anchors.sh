#!/usr/bin/env bash
set -euo pipefail

# Requires scripts/20_prepare_native_v2_panel.sh. One vLLM worker per visible
# GPU claims a single checkpoint at a time. All assistant checkpoints are
# scored at their tokenizer's native assistant generation position.
run='runs/anchors_native_interface/v2'
gpu_ids="${GPU_IDS:-0 1 2 3}"
mkdir -p "$run/logs"

for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit anchor-worker \
    --nodes data/panels/official_plus_posttrain_trajectory_v2.tsv \
    --wild-nodes manifests/wild_candidates.tsv \
    --anchors data/anchors/anthropic_persona_v1.jsonl \
    --interfaces data/interfaces/native_interface_v2.jsonl \
    --interface-renderings native_chat_template \
    --output-dir "$run" --cache-dir "$PWD/.hf-cache" --batch-size 512 \
    > "$run/logs/worker-${gpu}.log" 2>&1 &
done

echo "Native-anchor workers started on GPUs: $gpu_ids"
echo "Expected completions: 71"
echo "Monitor: find $run/metadata -name '*.json' | wc -l"
