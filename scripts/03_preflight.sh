#!/usr/bin/env bash
set -euo pipefail

# Final OLMo 3 base release; it shares the OLMo 3 tokenizer family used here.
persona-audit validate-tokenizer \
  --model-repo allenai/Olmo-3-1025-7B \
  --model-revision a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --anchors data/anchors/anthropic_persona_v1.jsonl \
  --output runs/preflight/olmo3_tokenizer.json
