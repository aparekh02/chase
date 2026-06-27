#!/usr/bin/env bash
# Run chase-mission's closed-loop VLA navigation.
#   ./run_nav.sh         headless — prints the tick-by-tick coordinate trace
#   ./run_nav.sh --view  opens the live MuJoCo viewer (watch the dog) via mjpython
set -euo pipefail

REPO="/Users/akshparekh/Documents/cadenza-cli"
VENV="/Users/akshparekh/Documents/cadenza-projects/rescue-dog/.venv/bin"
SCRIPT="$REPO/chase-mission/vla_navigate.py"

if [[ "${1:-}" == "--view" ]]; then
  PYTHONPATH="$REPO" "$VENV/mjpython" "$SCRIPT" --view
else
  PYTHONPATH="$REPO" "$VENV/python" "$SCRIPT"
fi
