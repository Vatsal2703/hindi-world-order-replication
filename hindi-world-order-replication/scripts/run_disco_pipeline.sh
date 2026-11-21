#!/usr/bin/env bash
# run_disco_pipeline.sh — v1.2
# Rebuilds the DISCO Excel master sheet on this machine.
#
# Pipeline (3 steps):
#   [1/3] scripts/23_score_disco.py      -> data/results/disco_affect_scored.csv   (~1 min)
#   [2/3] scripts/24_regress_difficulty_emotion.py -> data/results/difficulty_params_results.csv (~20 min)
#   [3/3] scripts/25_build_disco_master_sheet.py               -> data/results/DISCO_master_sheet.xlsx   (seconds)
#
#   bash scripts/run_disco_pipeline.sh          # quick: Excel only (script 25, a few seconds)
#   bash scripts/run_disco_pipeline.sh full     # full : scripts 23 -> 24 -> 25 (~20 min)
#
# Run from the repo root (hindi-world-order-replication/).
# Needs: pandas statsmodels wordfreq openpyxl
#   python -m pip install pandas statsmodels wordfreq openpyxl
#
# Version history
#   v1.0  first version
#   v1.2  script 23 is v1.3 (modifier fix: negation scope from the parse, modifiers
#         not scored as emotion words); HUTB side: run
#         `python scripts/20_build_hutb_raw_dataset.py --emotion-only` once first
#   v1.1  1 Oct clean-up: step list and timings updated (script 23 v1.2 is
#         scoring-only, ~1 min; script 24 v1.1 writes only the results CSV)
set -euo pipefail
VERSION="v1.2"
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
  echo "[1/3] script 23: DISCO emotion + surprisal scoring (~1 min)"
  $RUN scripts/23_score_disco.py
  echo "[2/3] script 24: surprisal + DL vs valence/arousal/dominance, HUTB & DISCO (~20 min)"
  $RUN scripts/24_regress_difficulty_emotion.py
fi
echo "[3/3] script 25: DISCO master sheet (Excel)"
$RUN scripts/25_build_disco_master_sheet.py

echo "Done. Excel: data/results/DISCO_master_sheet.xlsx"
echo "=== run_disco_pipeline.sh ${VERSION} finished ==="
