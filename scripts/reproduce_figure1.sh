#!/usr/bin/env bash
# Lightweight plotting only; source CSVs contain reported paper values.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$REPO_DIR/plotting/plot_figures.py" --figure 1 "$@"
