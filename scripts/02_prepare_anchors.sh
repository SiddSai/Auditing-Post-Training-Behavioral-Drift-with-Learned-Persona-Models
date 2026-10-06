#!/usr/bin/env bash
set -euo pipefail

revision='84fcc677e52e1902d696c32cd1a6b663e70d3993'
snapshot='data/snapshots/anthropic-evals'
pool='data/anchors/anthropic_persona_pool.jsonl'
battery='data/anchors/anthropic_persona_v1.jsonl'

if [[ ! -f "$snapshot/SOURCE_PROVENANCE.json" ]]; then
  persona-audit snapshot-anthropic-persona --output-dir "$snapshot" --revision "$revision"
fi
if [[ ! -f "$pool.provenance.json" ]]; then
  persona-audit import-anthropic-persona --source-dir "$snapshot" \
    --source-revision "$revision" --output "$pool"
fi
if [[ ! -f "$battery.provenance.json" ]]; then
  persona-audit sample-anchors --input "$pool" --output "$battery" \
    --size 300 --seed 20261005
fi
persona-audit validate --nodes manifests/nodes.tsv --anchors "$battery"
