#!/usr/bin/env bash
set -euo pipefail

# Frozen primary panel: all 50 official post-training checkpoints and all 21
# provenance-checked wild descendants. Think templates use the documented
# empty-<think> direct-answer intervention at inference time; no descendants
# are excluded on the basis of a prior generated-CoT pilot.
bash scripts/02_prepare_anchors.sh
bash scripts/06_prepare_targets.sh

official='data/panels/official_plus_posttrain_trajectory_v4.tsv'
wild='manifests/wild_candidates.tsv'
interfaces='data/interfaces/native_interface_v4.jsonl'
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
targets='data/targets/final_initial_native_three_v1.jsonl'

persona-audit compose-node-manifests \
  --inputs manifests/nodes.tsv manifests/posttrain_trajectory_nodes.tsv \
  --output "$official"
persona-audit audit-native-interfaces --nodes "$official" --wild-nodes "$wild" --output "$interfaces"
persona-audit prepare-direct-answer-anchors --input data/anchors/anthropic_persona_v1.jsonl --output "$anchors"
persona-audit filter-target-families --input data/targets/single_turn_v1.jsonl --output "$targets" --families ifeval xstest do_not_answer
persona-audit audit-target-anchor-disjointness --targets "$targets" --anchors "$anchors" --output data/targets/final_initial_native_three_v1.direct_answer_anchor_disjointness.json
persona-audit audit-direct-answer-tokenizers --nodes "$official" --wild-nodes "$wild" --anchors "$anchors" --interfaces "$interfaces" --output data/audits/direct_answer_tokenizer_preflight_v4.json

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

official = Path('data/panels/official_plus_posttrain_trajectory_v4.tsv')
wild = Path('manifests/wild_candidates.tsv')
interfaces = [json.loads(line) for line in Path('data/interfaces/native_interface_v4.jsonl').read_text().splitlines()]
counts = Counter(row['rendering'] for row in interfaces)
assert len(official.read_text().splitlines()) - 1 == 90
assert len(wild.read_text().splitlines()) - 1 == 21
assert counts == {'raw_completion': 40, 'native_chat_template': 71}, counts
print({'official_nodes': 90, 'wild_nodes': 21, 'assistant_models': 71, 'renderings': dict(counts)})
PY

echo 'Frozen standardized primary v4 inputs are prepared: all 71 assistant models use one shared direct-answer instrument.'
