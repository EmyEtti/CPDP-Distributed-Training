#!/usr/bin/env bash
# Print the exact resource requirements and commands; no run without --execute.
# Select --experiment ID. See configs/wan/table8_profile.json for GPUs/nodes and dataset.
# Outputs: results/new_runs/table8_profile/ID/sSEED/.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$REPO_DIR/scripts/launch.py" --config "$REPO_DIR/configs/wan/table8_profile.json" "$@"
