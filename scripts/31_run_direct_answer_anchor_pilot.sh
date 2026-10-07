#!/usr/bin/env bash
set -euo pipefail

# Necessary protocol gate, not a result-selection experiment. The six models
# span official Instruct/Think endpoints plus independent Instruct/Think
# descendants; all use a deterministic 48-item sample of the frozen battery.
run='runs/final_initial_v1/anchor_pilot'
anchors='data/anchors/anthropic_persona_direct_answer_pilot_v1.jsonl'
gpu_ids="${GPU_IDS:-0 1}"

persona-audit sample-anchors \
  --input data/anchors/anthropic_persona_direct_answer_v1.jsonl \
  --output "$anchors" --size 48 --seed 20261007
mkdir -p "$run/logs"
for gpu in $gpu_ids; do
  CUDA_VISIBLE_DEVICES="$gpu" persona-audit anchor-worker \
    --nodes data/panels/official_plus_posttrain_trajectory_v2.tsv \
    --wild-nodes manifests/wild_candidates.tsv --anchors "$anchors" \
    --interfaces data/interfaces/native_interface_v2.jsonl \
    --interface-renderings native_chat_template --anchor-protocol direct_answer_no_think \
    --node-ids-file manifests/direct_answer_anchor_pilot_nodes_v1.txt \
    --output-dir "$run" --cache-dir "$PWD/.hf-cache" --batch-size 512 \
    > "$run/logs/worker-${gpu}.log" 2>&1 &
done
wait
persona-audit collect-observations --run-dir "$run" --output "$run/observations.jsonl"
persona-audit validate-anchor-measurement --observations "$run/observations.jsonl" \
  --output "$run/measurement_gate.csv" --min-node-median-mass 1e-4
wc -l "$run/observations.jsonl"
echo 'Direct-answer pilot passed. Inspect measurement_gate.csv, then run script 32.'
