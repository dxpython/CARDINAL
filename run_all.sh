#!/usr/bin/env bash
# One-click reproduce: run every experiment, then figures and tables.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_all.py
