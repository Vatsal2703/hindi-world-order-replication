# Bounded Rationality — Surprisal Pipeline

Reads `data/processed/reference_sentences.pkl` and `full_hutb_sentences.pkl`,
produced upstream by the main pipeline's `02_data_preparation.py`.

## Pipeline order
```
per_contituent.py/01_variant_generation.py
      ↓
per_contituent.py/02_train_traingram_br.py     ← needs EMILLE corpus
      ↓
per_contituent.py/03_per_conti.py
      ↓
per_contituent.py/04_analysys_per_constituent.py   ← produces all plots & stats (v2)
```

Note: there used to be a separate `05_analyse.py` plotting step; its v1 position
plot was methodologically confounded (see `04`'s own docstring) and it has been
removed. `04` now folds in the corrected version of that analysis directly, so
it is the current final step.

## Key outputs
| File | Created by | Used by |
|------|-----------|---------|
| `data/processed/reference_sentences.pkl` | main pipeline `02_data_preparation.py` | 01, 03 |
| `data/processed/full_hutb_sentences.pkl` | main pipeline `02_data_preparation.py` | 02 |
| `data/processed/bounded_rationality_all_variants_final.pkl` | 01 | 03 |
| `data/models/br_trigram_model_blind.pkl` | 02 | 03 |
| `data/features/per_constituent_surprisal.csv` | 03 | 04 |
| `data/figures/position_ref_vs_var_panels_{PRIMARY}.png` | 04 | presentation |
| `data/figures/verb_adjacent_ref_vs_var_{PRIMARY}.png` | 04 | presentation |
| `data/figures/surprisal_gap_by_k_{PRIMARY}.png` | 04 | presentation |

`aggregate.py` sits alongside `03_per_conti.py` as a plain importable module
(not a pipeline step) — `03` imports `aggregate` from it directly.

## Notes
- In `04_analysys_per_constituent.py`, toggle `PRIMARY = 'mean'` or `PRIMARY = 'sum'` to switch aggregation.
- Expected dataset size: ~7,620 reference sentences, ~154,948 variants.
- Known k=3 anomaly in by-k gap table is real, confirmed across two datasets.
