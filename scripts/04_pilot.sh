#!/usr/bin/env bash
set -euo pipefail

# Run this once per chosen GPU. It produces three interpretable checkpoints:
# early base, final base, and final RLVR Instruct.
cache_dir="${HF_HOME:-$PWD/.hf-cache}"
run_dir='runs/pilot_v1'
anchors='data/anchors/anthropic_persona_v1.jsonl'

for node_id in base-s1-003000 base-main instruct-rlvr; do
  persona-audit run-node --nodes manifests/nodes.tsv --node-id "$node_id" \
    --anchors "$anchors" --output-dir "$run_dir" --cache-dir "$cache_dir"
done
persona-audit collect-observations --run-dir "$run_dir" \
  --output "$run_dir/observations.jsonl"
