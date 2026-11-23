#!/usr/bin/env bash
# Print the exact resource requirements and commands; no run without --execute.
# Select --experiment ID. See configs/tinyimagenet/table14.json for GPUs/nodes and dataset.
# Outputs: results/new_runs/table14/ID/sSEED/.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$REPO_DIR/scripts/launch.py" --config "$REPO_DIR/configs/tinyimagenet/table14.json" "$@"
