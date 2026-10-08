#!/usr/bin/env bash
set -euo pipefail

# Mandatory handoff artifact: run this before terminating the VM. It captures
# completions, deterministic source-native scores, predictor outputs, frozen
# manifests/interfaces, and tracked code—but never multi-GB model caches.
run='runs/final_standardized_v4'
[[ -f "$run/predictor/representation_report.csv" ]] || { echo 'Predictor suite is incomplete; refusing archive.' >&2; exit 1; }
mkdir -p exports
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="exports/persona_audit_final_standardized_v4_${stamp}.tar.gz"
manifest="exports/persona_audit_final_standardized_v4_${stamp}.sha256"

git ls-files -z | tar --null -T - -czf "$archive" \
  data/targets data/panels data/interfaces data/audits \
  "$run"
shasum -a 256 "$archive" | tee "$manifest"
echo "Archive: $archive"
echo "Checksum: $manifest"
echo 'Pull BOTH files to your Mac before terminating the VM:'
echo "rsync -avP ubuntu@YOUR_VM_IP:~/Auditing_Persona_Drift/$archive ./"
echo "rsync -avP ubuntu@YOUR_VM_IP:~/Auditing_Persona_Drift/$manifest ./"
