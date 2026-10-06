#!/usr/bin/env bash
set -euo pipefail

# Run after scripts/06_prepare_targets.sh. One independent vLLM process per
# selected GPU claims a checkpoint at a time; checkpoints are cached and never
# all kept in GPU memory together. Override GPU_IDS for any VM shape, e.g.
# `GPU_IDS="0 1" bash scripts/07_run_targets_4xh100.sh`.
mkdir -p runs/targets/v1/logs
gpu_ids="${GPU_IDS:-0 1 2 3}"
for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit target-worker \
    --nodes manifests/nodes.tsv --wild-nodes manifests/wild_candidates.tsv \
    --targets data/targets/single_turn_v1.jsonl --output-dir runs/targets/v1 \
    --cache-dir "$PWD/.hf-cache" --batch-size 128 \
    > "runs/targets/v1/logs/worker-${gpu}.log" 2>&1 &
done
echo "Workers started on GPUs: $gpu_ids"
echo 'Monitor: find runs/targets/v1/metadata -name "*.json" | wc -l'
