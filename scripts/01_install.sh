#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating an empty conda environment.
python - <<'PY'
import sys
if not (sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12):
    raise SystemExit("Use Python 3.10–3.12 for the GPU environment.")
print("Python", sys.version)
PY
python -m pip install --upgrade pip
python -m pip install -e '.[inference]'
python -m persona_audit.cli --help
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
