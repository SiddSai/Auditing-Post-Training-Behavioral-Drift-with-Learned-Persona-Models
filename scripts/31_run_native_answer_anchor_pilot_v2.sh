#!/usr/bin/env bash
set -euo pipefail

# Protocol gate for the repaired anchor measurement. Instruct models are
# measured at their ordinary assistant boundary; native Think models first
# greedily close their own <think> trace, then are measured at the published
# post-</think> answer boundary. This is a validation pilot, not a result
# selection experiment. Its output is intentionally versioned beside the
# failed no-think attempt.
# v3 preserves the failed 512-token trace pilot for provenance. The only
# changed input is the documented 2,048-token native-think trace allowance.
run='runs/final_initial_v2/anchor_pilot_native_think_v3'
anchors='data/anchors/anthropic_persona_direct_answer_pilot_v2.jsonl'
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
    --interface-renderings native_chat_template --anchor-protocol native_think_then_answer \
    --node-ids-file manifests/direct_answer_anchor_pilot_nodes_v1.txt \
    --output-dir "$run" --cache-dir "$PWD/.hf-cache" --batch-size 128 --think-max-tokens 2048 \
    > "$run/logs/worker-${gpu}.log" 2>&1 &
done
wait
[[ "$(find "$run/metadata" -name '*.json' | wc -l | tr -d ' ')" == 6 ]] || { echo 'Anchor pilot is incomplete; inspect logs before retrying.' >&2; exit 1; }
persona-audit collect-observations --run-dir "$run" --output "$run/observations.jsonl"
persona-audit validate-anchor-measurement --observations "$run/observations.jsonl" \
  --output "$run/measurement_gate.csv" --min-node-median-mass 1e-4
wc -l "$run/observations.jsonl"
echo 'Native-Think anchor pilot passed. Inspect measurement_gate.csv, then run script 32b.'
