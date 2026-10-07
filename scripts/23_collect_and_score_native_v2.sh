#!/usr/bin/env bash
set -euo pipefail

# Collect the two v2 panels only after every expected node is complete. These
# are source-native deterministic scorers; this script deliberately does not
# invoke a paid LLM judge.
anchor_run='runs/anchors_native_interface/v2'
target_run='runs/targets_native_interface/v2'
targets='data/targets/single_turn_v1.jsonl'
out="$target_run/native_scorers"

persona-audit collect-observations --run-dir "$anchor_run" --output "$anchor_run/observations.jsonl"
persona-audit collect-target-observations --run-dir "$target_run" --output "$target_run/observations.jsonl"

python - <<'PY'
import nltk
for package in ('punkt', 'punkt_tab'):
    nltk.download(package, quiet=True)
PY

mkdir -p "$out"
persona-audit write-native-scorer-inputs --observations "$target_run/observations.jsonl" --targets "$targets" --output-dir "$out"
persona-audit score-xstest-native --observations "$target_run/observations.jsonl" --targets "$targets" --output "$out/xstest_scores.jsonl"
persona-audit score-ifeval-native --observations "$target_run/observations.jsonl" --targets "$targets" --source-root data/snapshots/target_sources_v1 --output "$out/ifeval_scores.jsonl"
persona-audit score-do-not-answer-longformer --observations "$target_run/observations.jsonl" --output "$out/do_not_answer_longformer_scores.jsonl" --device 0 --batch-size 64

wc -l "$anchor_run/observations.jsonl" "$target_run/observations.jsonl" "$out/ifeval_scores.jsonl" "$out/xstest_scores.jsonl" "$out/do_not_answer_longformer_scores.jsonl"
echo 'v2 collection and source-native scoring complete.'
