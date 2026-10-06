#!/usr/bin/env bash
set -euo pipefail

# Does not call a paid API. It recreates source-native scorer inputs and runs
# only evaluators that are fully local/released.
obs='runs/targets/v1/observations.jsonl'
targets='data/targets/single_turn_v1.jsonl'
out='runs/targets/v1/native_scorers'
mkdir -p "$out"

persona-audit write-native-scorer-inputs --observations "$obs" --targets "$targets" --output-dir "$out"
persona-audit score-xstest-native --observations "$obs" --targets "$targets" --output "$out/xstest_scores.jsonl"

# BOLD's prompt repository ships no evaluator. This runs its reproducible
# paper-era VADER component; unreleased toxicity/regard artifacts stay absent.
persona-audit score-bold-vader --observations "$obs" --targets "$targets" \
  --output "$out/bold_vader_scores.jsonl" --download-lexicon

echo 'Prepared source-native requests and local scores under' "$out"
echo 'See scripts/10_run_judged_scorers.sh for API-backed released-template judges.'
