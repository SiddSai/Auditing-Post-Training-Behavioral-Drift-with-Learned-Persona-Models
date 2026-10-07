#!/usr/bin/env bash
set -euo pipefail

# This is the immutable, deployment-interface primary panel. It joins the
# official release endpoints to the separately versioned official trajectories,
# then audits every tokenizer/template revision before any inference begins.
official='data/panels/official_plus_posttrain_trajectory_v2.tsv'
interfaces='data/interfaces/native_interface_v2.jsonl'

persona-audit compose-node-manifests \
  --inputs manifests/nodes.tsv manifests/posttrain_trajectory_nodes.tsv \
  --output "$official"

persona-audit audit-native-interfaces \
  --nodes "$official" \
  --wild-nodes manifests/wild_candidates.tsv \
  --output "$interfaces"

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

official = Path('data/panels/official_plus_posttrain_trajectory_v2.tsv')
interface_rows = [json.loads(line) for line in Path('data/interfaces/native_interface_v2.jsonl').read_text().splitlines()]
counts = Counter(row['rendering'] for row in interface_rows)
print({'official_nodes': len(official.read_text().splitlines()) - 1, 'all_nodes': len(interface_rows), 'renderings': dict(counts)})
assert counts['native_chat_template'] == 71, counts
assert counts['raw_completion'] == 40, counts
PY

echo "Prepared native v2 panel: $official and $interfaces"
