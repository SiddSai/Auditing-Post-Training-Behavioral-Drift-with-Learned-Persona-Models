#!/usr/bin/env bash
set -euo pipefail

# Run this before terminating the VM. It produces one self-contained archive
# of raw completions, deterministic scores, states/predictions, frozen inputs,
# and the exact checked-out code. It intentionally excludes model caches.
run='runs/final_initial_v1'
[[ -f "$run/predictor/representation_report.csv" ]] || { echo 'Predictor suite is incomplete; refusing archive.' >&2; exit 1; }
mkdir -p exports
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="exports/persona_audit_final_initial_v1_${stamp}.tar.gz"
manifest="exports/persona_audit_final_initial_v1_${stamp}.sha256"

git ls-files -z | tar --null -T - -czf "$archive" \
  data/targets data/panels data/interfaces \
  "$run"
shasum -a 256 "$archive" | tee "$manifest"
echo "Archive: $archive"
echo "Checksum: $manifest"
echo 'From your Mac, pull BOTH files before terminating:'
echo "rsync -avP ubuntu@YOUR_VM_IP:~/Auditing_Persona_Drift/$archive ./"
echo "rsync -avP ubuntu@YOUR_VM_IP:~/Auditing_Persona_Drift/$manifest ./"
