#!/usr/bin/env bash
set -euo pipefail
run='runs/final_standardized_v5'
[[ -f "$run/predictor/representation_report.csv" ]] || { echo 'Predictor suite is incomplete; refusing archive.' >&2; exit 1; }
mkdir -p exports
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="exports/persona_audit_final_standardized_v5_${stamp}.tar.gz"
manifest="exports/persona_audit_final_standardized_v5_${stamp}.sha256"
git ls-files -z | tar --null -T - -czf "$archive" data/targets data/panels data/interfaces data/audits "$run"
shasum -a 256 "$archive" | tee "$manifest"
echo "Archive: $archive"
echo "Checksum: $manifest"
