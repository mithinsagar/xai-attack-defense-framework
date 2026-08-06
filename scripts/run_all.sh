#!/usr/bin/env bash
# =====================================================================
# XAI Attack and Defense Framework - End-to-End Pipeline Runner
# Author : Mithin Sagar S  (https://github.com/mithinsagar)
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo " XAI Attack & Defense Framework"
echo " Full pipeline run"
echo " Author: Mithin Sagar S"
echo "=========================================="

python main.py --phase preprocess
python main.py --phase train_base
python main.py --phase explain
python main.py --phase attack
python main.py --phase fewshot
python main.py --phase defense

echo "Pipeline complete. See ./results/ for CSVs and ./figures/ for plots."
