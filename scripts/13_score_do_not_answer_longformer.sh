#!/usr/bin/env bash
set -euo pipefail

# Source-native Do-Not-Answer evaluation. This executes the released
# response-only Longformer pipelines, not an API judge or reconstructed rubric.
persona-audit score-do-not-answer-longformer \
  --observations runs/targets/v1/observations.jsonl \
  --output runs/targets/v1/native_scorers/do_not_answer_longformer_scores.jsonl \
  --device 0 --batch-size 64

persona-audit score-do-not-answer-longformer \
  --observations runs/targets_native_interface/v1/observations.jsonl \
  --output runs/targets_native_interface/v1/native_scorers/do_not_answer_longformer_scores.jsonl \
  --device 0 --batch-size 64

echo 'Finished source-native Do-Not-Answer Longformer scores for raw and native-interface panels.'
