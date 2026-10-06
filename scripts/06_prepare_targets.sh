#!/usr/bin/env bash
set -euo pipefail

# Source-pinned construction of the single-turn target panel.  Safe to rerun:
# snapshot-git verifies that every existing snapshot is at its intended SHA.
source_root='data/snapshots/target_sources_v1'
pool='data/targets/target_pool_v1.jsonl'
targets='data/targets/single_turn_v1.jsonl'

persona-audit snapshot-git --repository https://github.com/meg-tong/sycophancy-eval.git \
  --revision 3ad652c773f5d1e30d5f6f61657ed934d768ecad --output "$source_root/sycophancy"
persona-audit snapshot-git --repository https://github.com/paul-rottger/xstest.git \
  --revision 460703484df354958a5e1cd7378a38fcb94a2f3e --output "$source_root/xstest"
persona-audit snapshot-git --repository https://github.com/Libr-AI/do-not-answer.git \
  --revision 9a1694221e3639887138f61deae344335eca6752 --output "$source_root/do_not_answer"
persona-audit snapshot-git --repository https://github.com/amazon-science/bold.git \
  --revision 4dee2311b23de20b43bfc12ace7037f3f71ba421 --output "$source_root/bold"
persona-audit snapshot-git --repository https://github.com/sylinrl/TruthfulQA.git \
  --revision d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d --output "$source_root/truthfulqa"

persona-audit import-target-pools --source-root "$source_root" --output "$pool"
persona-audit freeze-target-splits --pool "$pool" --output "$targets" \
  --development-size 300 --evaluation-size 150 --seed 20261005
persona-audit audit-target-anchor-disjointness --targets "$targets" \
  --anchors data/anchors/anthropic_persona_v1.jsonl \
  --output data/targets/single_turn_v1.anchor_disjointness.json
python - <<'PY'
import json
from collections import Counter
path = 'data/targets/single_turn_v1.jsonl'
rows = [json.loads(line) for line in open(path) if line.strip()]
print(json.dumps({'targets': len(rows), 'by_family_split': Counter(f"{x['family']}:{x['split']}" for x in rows)}, sort_keys=True, default=dict))
PY
