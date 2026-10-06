#!/usr/bin/env bash
set -euo pipefail

# Does not call a paid API. It recreates source-native scorer inputs and runs
# only evaluators that are fully local/released.
obs='runs/targets/v1/observations.jsonl'
targets='data/targets/single_turn_v1.jsonl'
out='runs/targets/v1/native_scorers'
mkdir -p "$out"

# The released IFEval evaluator imports NLTK and loads the Punkt English
# sentence tokenizer. Keep this explicit and local to the scoring protocol.
python - <<'PY'
import nltk
for package in ("punkt", "punkt_tab"):
    nltk.download(package, quiet=True)
PY

persona-audit write-native-scorer-inputs --observations "$obs" --targets "$targets" --output-dir "$out"
persona-audit score-xstest-native --observations "$obs" --targets "$targets" --output "$out/xstest_scores.jsonl"

# Execute the released strict and loose IFEval checkers.
persona-audit score-ifeval-native --observations "$obs" --targets "$targets" \
  --source-root data/snapshots/target_sources_v1 --output "$out/ifeval_scores.jsonl"

echo 'Prepared source-native requests and local scores under' "$out"
echo 'See scripts/10_run_judged_scorers.sh for API-backed released-template judges.'
