# Bounded Rationality — Surprisal Pipeline (Colab Notebooks)

## Pipeline order
```
01_data_preparation.ipynb
      ↓
02_variant_generation.ipynb
      ↓
03_train_trigram.ipynb          ← needs EMILLE corpus
      ↓
04_per_conti.ipynb
      ↓
05_analyse.ipynb                ← produces all plots & stats
```

## Key outputs
| File | Created by | Used by |
|------|-----------|---------|
| `data/processed/reference_sentences.pkl` | 01 | 02, 03, 04 |
| `data/processed/full_hutb_sentences.pkl` | 01 | — |
| `data/processed/bounded_rationality_all_variants_final.pkl` | 02 | 04 |
| `data/models/br_trigram_model_blind.pkl` | 03 | 04 |
| `data/features/per_constituent_surprisal.csv` | 04 | 05 |
| `data/figures/*.png` | 05 | presentation |

## Notes
- Each notebook re-mounts Drive at the top — run that cell every new Colab session.
- In `05_analyse.ipynb`, toggle `PRIMARY = 'mean'` or `PRIMARY = 'sum'` to switch aggregation.
- Expected dataset size: ~7,620 reference sentences, ~154,948 variants.
- Known k=3 anomaly in by-k gap table is real, confirmed across two datasets.
