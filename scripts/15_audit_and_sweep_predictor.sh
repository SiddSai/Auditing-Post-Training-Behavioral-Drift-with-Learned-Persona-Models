#!/usr/bin/env bash
set -euo pipefail

# Rebuild primary results with metadata controls and geometry artifacts, then
# run a frozen unsupervised state-representation sensitivity sweep.  No target
# labels choose a method or dimensionality here.
bash scripts/14_run_first_predictor.sh

PRIMARY="runs/predictor/v1/raw_primary"
persona-audit audit-predictor \
  --predictions "$PRIMARY/predictions.jsonl" \
  --state-geometry "$PRIMARY/state_geometry.jsonl" \
  --output-dir "$PRIMARY/audit"

for method in pca factor; do
  for dimensions in 2 4 8 16; do
    out="runs/predictor/v1/representation_sweep/${method}_d${dimensions}"
    persona-audit run-predictor \
      --anchor-observations runs/anchors/v1/observations.jsonl \
      --anchors data/anchors/anthropic_persona_v1.jsonl \
      --targets data/targets/single_turn_v1.jsonl \
      --outcomes runs/predictor/v1/outcomes.jsonl \
      --nodes manifests/nodes.tsv \
      --wild-nodes manifests/wild_candidates.tsv \
      --output-dir "$out" \
      --state-method "$method" \
      --state-dimensions "$dimensions"
    persona-audit audit-predictor \
      --predictions "$out/predictions.jsonl" \
      --state-geometry "$out/state_geometry.jsonl" \
      --output-dir "$out/audit"
  done
done

echo "Primary audit: $PRIMARY/audit"
echo "Frozen representation sweep: runs/predictor/v1/representation_sweep"
