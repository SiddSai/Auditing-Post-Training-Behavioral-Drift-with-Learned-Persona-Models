#!/usr/bin/env bash
set -euo pipefail

persona-audit collect-target-observations --run-dir runs/targets/v1 \
  --output runs/targets/v1/observations.jsonl
persona-audit score-target-observations --observations runs/targets/v1/observations.jsonl \
  --output runs/targets/v1/outcomes.jsonl
wc -l runs/targets/v1/observations.jsonl runs/targets/v1/outcomes.jsonl
