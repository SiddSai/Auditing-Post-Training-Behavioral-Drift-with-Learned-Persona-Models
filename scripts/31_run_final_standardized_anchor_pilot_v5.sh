#!/usr/bin/env bash
set -euo pipefail
run='runs/final_standardized_v5/anchor_pilot'; official='data/panels/official_plus_posttrain_trajectory_v5.tsv'; wild='data/panels/wild_standardized_valid_v5.tsv'; interfaces='data/interfaces/native_interface_v5.jsonl'; anchors='data/anchors/anthropic_persona_direct_answer_pilot_v5.jsonl'; gpu_ids="${GPU_IDS:-0 1}"
persona-audit sample-anchors --input data/anchors/anthropic_persona_direct_answer_v1.jsonl --output "$anchors" --size 48 --seed 20261008
mkdir -p "$run/logs"
for gpu in $gpu_ids; do CUDA_VISIBLE_DEVICES="$gpu" persona-audit anchor-worker --nodes "$official" --wild-nodes "$wild" --anchors "$anchors" --interfaces "$interfaces" --interface-renderings native_chat_template --anchor-protocol think_not_thinking_empty_prefill --output-dir "$run" --cache-dir "$PWD/.hf-cache" --batch-size 128 > "$run/logs/worker-${gpu}.log" 2>&1 & done
wait
[[ "$(find "$run/metadata" -name '*.json' | wc -l | tr -d ' ')" == 70 ]] || { echo 'Anchor pilot is incomplete; inspect logs before retrying.' >&2; exit 1; }
persona-audit collect-observations --run-dir "$run" --output "$run/observations.jsonl"
persona-audit validate-anchor-measurement --observations "$run/observations.jsonl" --output "$run/measurement_gate.csv" --min-node-median-mass 1e-4
[[ "$(wc -l < "$run/observations.jsonl")" == 3360 ]] || { echo 'Expected 3,360 pilot anchor rows.' >&2; exit 1; }
echo 'All 70 primary assistant models passed the standardized direct-answer gate.'
