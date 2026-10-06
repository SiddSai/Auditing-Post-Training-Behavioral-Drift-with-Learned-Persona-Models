#!/usr/bin/env bash
set -euo pipefail

persona-audit collect-target-observations --run-dir runs/targets/v1 \
  --output runs/targets/v1/observations.jsonl
wc -l runs/targets/v1/observations.jsonl
echo 'Next: bash scripts/09_prepare_native_scorers.sh'
