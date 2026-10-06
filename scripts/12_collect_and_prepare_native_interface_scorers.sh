#!/usr/bin/env bash
set -euo pipefail

# Keep this deployment-interface companion panel separate from the invariant
# raw-string panel at runs/targets/v1.
run='runs/targets_native_interface/v1'
obs="$run/observations.jsonl"
targets='data/targets/single_turn_v1.jsonl'
out="$run/native_scorers"

persona-audit collect-target-observations --run-dir "$run" --output "$obs"
wc -l "$obs"

# Required by the released Google IFEval checker.
python - <<'PY'
import nltk
for package in ("punkt", "punkt_tab"):
    nltk.download(package, quiet=True)
PY

mkdir -p "$out"
persona-audit write-native-scorer-inputs --observations "$obs" --targets "$targets" --output-dir "$out"
persona-audit score-xstest-native --observations "$obs" --targets "$targets" --output "$out/xstest_scores.jsonl"
persona-audit score-ifeval-native --observations "$obs" --targets "$targets" \
  --source-root data/snapshots/target_sources_v1 --output "$out/ifeval_scores.jsonl"

echo "Prepared native-interface local scores and judge requests under $out"
echo "Do not mix these files with runs/targets/v1/native_scorers."
