#!/usr/bin/env bash
set -euo pipefail

# Build the frozen, anchor-compatible primary panel. The original 21-member
# descendant candidate registry is retained untouched; one Think descendant
# is excluded here because its pre-registered pilot assigned a median
# probability mass of 1.95e-6 to the forced Yes/No answer set after completing
# native reasoning, far below the 1e-4 validity gate.
bash scripts/02_prepare_anchors.sh
bash scripts/06_prepare_targets.sh

official='data/panels/official_plus_posttrain_trajectory_v3.tsv'
wild='data/panels/wild_anchor_compatible_v1.tsv'
interfaces='data/interfaces/native_interface_v3.jsonl'
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'

persona-audit compose-node-manifests \
  --inputs manifests/nodes.tsv manifests/posttrain_trajectory_nodes.tsv \
  --output "$official"
persona-audit filter-wild-nodes \
  --input manifests/wild_candidates.tsv --output "$wild" \
  --exclude-candidate-ids AS-SiliconMind/SiliconMind-V1-Olmo-3-7B-Think \
  --rationale 'Failed frozen 48-anchor native-Think pilot: median bare Yes/No probability mass 1.95e-6 after a completed native reasoning trace; below the pre-registered 1e-4 measurement gate. The source candidate remains preserved in manifests/wild_candidates.tsv.'
persona-audit audit-native-interfaces --nodes "$official" --wild-nodes "$wild" --output "$interfaces"
persona-audit prepare-direct-answer-anchors --input data/anchors/anthropic_persona_v1.jsonl --output "$anchors"
persona-audit filter-target-families --input data/targets/single_turn_v1.jsonl --output data/targets/final_initial_native_three_v1.jsonl --families ifeval xstest do_not_answer
persona-audit audit-target-anchor-disjointness --targets data/targets/final_initial_native_three_v1.jsonl --anchors "$anchors" --output data/targets/single_turn_v1.direct_answer_anchor_disjointness.json
persona-audit audit-direct-answer-tokenizers --nodes "$official" --wild-nodes "$wild" --anchors "$anchors" --interfaces "$interfaces" --output data/audits/direct_answer_tokenizer_preflight_v3.json

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

official = Path('data/panels/official_plus_posttrain_trajectory_v3.tsv')
wild = Path('data/panels/wild_anchor_compatible_v1.tsv')
interfaces = [json.loads(line) for line in Path('data/interfaces/native_interface_v3.jsonl').read_text().splitlines()]
counts = Counter(row['rendering'] for row in interfaces)
assert len(official.read_text().splitlines()) - 1 == 90
assert len(wild.read_text().splitlines()) - 1 == 20
assert counts == {'raw_completion': 40, 'native_chat_template': 70}, counts
print({'official_nodes': 90, 'anchor_compatible_wild_nodes': 20, 'renderings': dict(counts)})
PY

echo 'Frozen final primary v3 inputs are prepared: 70 assistant models with a valid shared anchor instrument.'
