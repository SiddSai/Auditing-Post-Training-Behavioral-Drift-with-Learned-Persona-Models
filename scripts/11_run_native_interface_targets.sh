#!/usr/bin/env bash
set -euo pipefail

# A separate output namespace prevents accidental mixing with the invariant raw
# prompt panel. The manifest is a committed audit of each pinned tokenizer's
# native template; no researcher-authored system message is supplied.
mkdir -p runs/targets_native_interface/v1/logs
gpu_ids="${GPU_IDS:-0 1 2 3}"
for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit target-worker \
    --nodes manifests/nodes.tsv --wild-nodes manifests/wild_candidates.tsv \
    --targets data/targets/single_turn_v1.jsonl \
    --interfaces data/interfaces/native_interface_v1.jsonl \
    --interface-renderings native_chat_template \
    --output-dir runs/targets_native_interface/v1 \
    --cache-dir "$PWD/.hf-cache" --batch-size 128 \
    > "runs/targets_native_interface/v1/logs/worker-${gpu}.log" 2>&1 &
done
echo "Native-interface workers started on GPUs: $gpu_ids (18 chat-capable models)"
echo 'Monitor: find runs/targets_native_interface/v1/metadata -name "*.json" | wc -l'
