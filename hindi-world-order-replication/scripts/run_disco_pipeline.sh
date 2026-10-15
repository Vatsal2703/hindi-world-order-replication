#!/usr/bin/env bash
# run_disco_pipeline.sh — v1.1
# Rebuilds the DISCO Excel master sheet on this machine.
#
# Pipeline (3 steps):
#   [1/3] scripts/31_disco_affect_as_information.py      -> data/results/disco_affect_scored.csv   (~1 min)
#   [2/3] scripts/32_difficulty_parameters_both_corpora.py -> data/results/difficulty_params_results.csv (~20 min)
#   [3/3] scripts/33_disco_master_sheet.py               -> data/results/DISCO_master_sheet.xlsx   (seconds)
#
#   bash scripts/run_disco_pipeline.sh          # quick: Excel only (script 33, a few seconds)
#   bash scripts/run_disco_pipeline.sh full     # full : scripts 31 -> 32 -> 33 (~20 min)
#
# Run from the repo root (hindi-world-order-replication/).
# Needs: pandas statsmodels wordfreq openpyxl
#   python -m pip install pandas statsmodels wordfreq openpyxl
#
# Version history
#   v1.0  first version
#   v1.1  1 Oct clean-up: step list and timings updated (script 31 v1.2 is
#         scoring-only, ~1 min; script 32 v1.1 writes only the results CSV)
set -euo pipefail
VERSION="v1.1"
MODE="${1:-quick}"
echo "=== run_disco_pipeline.sh ${VERSION} (mode: ${MODE}) ==="

if [ ! -d scripts ] || [ ! -d data ]; then
  echo "ERROR: run this from the repo root (the folder that contains scripts/ and data/)"; exit 1
fi
python -c "import pandas, statsmodels, wordfreq, openpyxl" 2>/dev/null || {
  echo "ERROR: missing packages. Run:  python -m pip install pandas statsmodels wordfreq openpyxl"; exit 1; }

# keep the Mac awake during long runs
RUN="python"
command -v caffeinate >/dev/null && RUN="caffeinate -i python"

if [ "$MODE" = "full" ]; then
  echo "[1/3] script 31: DISCO emotion + surprisal scoring (~1 min)"
  $RUN scripts/31_disco_affect_as_information.py
  echo "[2/3] script 32: surprisal + DL vs valence/arousal/dominance, HUTB & DISCO (~20 min)"
  $RUN scripts/32_difficulty_parameters_both_corpora.py
fi
echo "[3/3] script 33: DISCO master sheet (Excel)"
$RUN scripts/33_disco_master_sheet.py

echo "Done. Excel: data/results/DISCO_master_sheet.xlsx"
echo "=== run_disco_pipeline.sh ${VERSION} finished ==="
