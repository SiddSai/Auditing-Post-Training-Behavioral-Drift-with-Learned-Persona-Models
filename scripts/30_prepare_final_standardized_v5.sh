#!/usr/bin/env bash
set -euo pipefail

# The v5 primary panel retains all provenance-checked candidates except the
# single descendant that failed the published empty-Think pilot validity gate.
bash scripts/02_prepare_anchors.sh
bash scripts/06_prepare_targets.sh
official='data/panels/official_plus_posttrain_trajectory_v5.tsv'
wild='data/panels/wild_standardized_valid_v5.tsv'
interfaces='data/interfaces/native_interface_v5.jsonl'
anchors='data/anchors/anthropic_persona_direct_answer_v1.jsonl'
targets='data/targets/final_initial_native_three_v1.jsonl'

persona-audit compose-node-manifests --inputs manifests/nodes.tsv manifests/posttrain_trajectory_nodes.tsv --output "$official"
persona-audit filter-wild-nodes --input manifests/wild_candidates.tsv --output "$wild" \
  --exclude-candidate-ids AS-SiliconMind/SiliconMind-V1-Olmo-3-7B-Think \
  --rationale 'Failed the all-model frozen 48-anchor Think-not-thinking pilot: median total probability mass on the forced {Yes, No} answer set was 3.53e-5, below the 1e-4 validity gate. No model-specific boundary repair is applied.'
persona-audit audit-native-interfaces --nodes "$official" --wild-nodes "$wild" --output "$interfaces"
persona-audit prepare-direct-answer-anchors --input data/anchors/anthropic_persona_v1.jsonl --output "$anchors"
persona-audit filter-target-families --input data/targets/single_turn_v1.jsonl --output "$targets" --families ifeval xstest do_not_answer
persona-audit audit-target-anchor-disjointness --targets "$targets" --anchors "$anchors" --output data/targets/single_turn_v1.direct_answer_anchor_disjointness.json
persona-audit audit-direct-answer-tokenizers --nodes "$official" --wild-nodes "$wild" --anchors "$anchors" --interfaces "$interfaces" --output data/audits/direct_answer_tokenizer_preflight_v5.json
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
official = Path('data/panels/official_plus_posttrain_trajectory_v5.tsv')
wild = Path('data/panels/wild_standardized_valid_v5.tsv')
interfaces = [json.loads(x) for x in Path('data/interfaces/native_interface_v5.jsonl').read_text().splitlines()]
assert len(official.read_text().splitlines()) - 1 == 90
assert len(wild.read_text().splitlines()) - 1 == 20
assert Counter(x['rendering'] for x in interfaces) == {'raw_completion': 40, 'native_chat_template': 70}
print('Frozen v5 primary: 70 valid assistant models; the one failed candidate is provenance-preserved in its filtered manifest.')
PY
