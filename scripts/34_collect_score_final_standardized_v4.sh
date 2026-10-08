#!/usr/bin/env bash
set -euo pipefail

target_run='runs/final_standardized_v4/targets'
targets='data/targets/final_initial_native_three_v1.jsonl'
out="$target_run/native_scorers"

[[ "$(find "$target_run/metadata" -name '*.json' | wc -l | tr -d ' ')" == 71 ]] || { echo 'Target run is incomplete.' >&2; exit 1; }
persona-audit collect-target-observations --run-dir "$target_run" --output "$target_run/observations.jsonl"
mkdir -p "$out"
python - <<'PY'
import nltk
for package in ('punkt', 'punkt_tab'):
    nltk.download(package, quiet=True)
PY
persona-audit write-native-scorer-inputs --observations "$target_run/observations.jsonl" --targets "$targets" --output-dir "$out"
persona-audit score-xstest-native --observations "$target_run/observations.jsonl" --targets "$targets" --output "$out/xstest_scores.jsonl"
persona-audit score-ifeval-native --observations "$target_run/observations.jsonl" --targets "$targets" --source-root data/snapshots/target_sources_v1 --output "$out/ifeval_scores.jsonl"
persona-audit score-do-not-answer-longformer --observations "$target_run/observations.jsonl" --output "$out/do_not_answer_longformer_scores.jsonl" --device 0 --batch-size 64

[[ "$(wc -l < "$target_run/observations.jsonl")" == 95850 ]] || { echo 'Expected 95,850 target rows.' >&2; exit 1; }
for scored in ifeval_scores.jsonl xstest_scores.jsonl do_not_answer_longformer_scores.jsonl; do
  [[ "$(wc -l < "$out/$scored")" == 31950 ]] || { echo "Expected 31,950 rows in $scored" >&2; exit 1; }
done
echo 'Standardized target collection and all three source-native scores passed row-count checks.'
