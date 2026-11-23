#!/usr/bin/env bash
# ResNet-50/CIFAR-100, two sites, 2 x 4 GPUs, LR 0.4; dry-run by default.
# Outputs: results/new_runs/lr_sensitivity/ID/sSEED/.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$REPO_DIR/scripts/launch.py" --config "$REPO_DIR/configs/wan/lr_sensitivity.json" "$@"
