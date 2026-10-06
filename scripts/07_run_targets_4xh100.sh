#!/usr/bin/env bash
set -euo pipefail

# Run after scripts/06_prepare_targets.sh. One independent vLLM process per
# GPU claims a checkpoint at a time; checkpoints are cached and never all kept
# in GPU memory together. `runs/targets/v1` is immutable once collection starts.
mkdir -p runs/targets/v1/logs
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit target-worker \
    --nodes manifests/nodes.tsv --wild-nodes manifests/wild_candidates.tsv \
    --targets data/targets/single_turn_v1.jsonl --output-dir runs/targets/v1 \
    --cache-dir "$PWD/.hf-cache" --batch-size 128 \
    > "runs/targets/v1/logs/worker-${gpu}.log" 2>&1 &
done
echo 'Workers started. Monitor: find runs/targets/v1/metadata -name "*.json" | wc -l'
