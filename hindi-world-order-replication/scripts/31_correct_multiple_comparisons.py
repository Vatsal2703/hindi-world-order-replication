#!/usr/bin/env python3
# v1.3 (2026-11-02): renumbered from 42_correct_multiple_comparisons.py; naming only, no logic change.
# v1.2 (2026-11-02): renamed from 42_multiple_comparisons.py; naming only, no logic change.
"""
31_correct_multiple_comparisons.py  (v1.3)

Corrects the MTP-II regression p-values for multiple comparisons, so that
the "many specifications" limitation becomes a tested result.

Input   data/results/lstm_v2_regressions.csv  (script 30)
Output  data/results/mtp2_corrected.csv

Primary family (18 tests) = the rows of the report's Table 5.1, per corpus:
  3 emotions x {DL (from the LSTM model), LSTM surprisal, trigram surprisal}.
Wide family (all rows of the file, incl. the joint "both" models) is also
corrected, as a stricter check.
Corrections: Holm (family-wise error) and Benjamini-Hochberg (FDR), alpha .05.
Note: p in the input is rounded to 4 decimals, so p = .0001 means p <= .0001;
this makes the correction slightly conservative, never lenient.

RUN:  python scripts/31_correct_multiple_comparisons.py   (needs pandas, statsmodels)
"""
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests

VERSION = "1.3"
print(f"=== 31_correct_multiple_comparisons.py v{VERSION} ===")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(BASE, "data/results/lstm_v2_regressions.csv")
OUT = os.path.join(BASE, "data/results/mtp2_corrected.csv")

d = pd.read_csv(IN)
prim = d[((d.model == "LSTMv2") & d.predictor.isin(["lstm", "dl"])) |
         ((d.model == "trigram") & (d.predictor == "tri"))].copy()
prim["family"] = "primary (18)"
wide = d.copy(); wide["family"] = f"wide ({len(d)})"

out = []
for fam in (prim, wide):
    fam = fam.copy()
    for meth, col in (("holm", "p_holm"), ("fdr_bh", "p_fdr")):
        rej, padj, _, _ = multipletests(fam.p.values, alpha=0.05, method=meth)
        fam[col] = padj.round(4); fam["sig_holm" if meth == "holm" else "sig_fdr"] = rej
    out.append(fam)
res = pd.concat(out)
res.to_csv(OUT, index=False)

p = out[0].sort_values(["predictor", "emotion", "corpus"])
print(f"\nPrimary family: {len(p)} tests")
print(p[["corpus", "emotion", "predictor", "beta", "p", "p_holm", "sig_holm", "p_fdr", "sig_fdr"]].to_string(index=False))
print(f"\nSignificant raw / Holm / FDR: {(p.p < .05).sum()} / {p.sig_holm.sum()} / {p.sig_fdr.sum()}")
w = out[1]
print(f"Wide family ({len(w)} tests) significant raw / Holm / FDR: {(w.p < .05).sum()} / {w.sig_holm.sum()} / {w.sig_fdr.sum()}")
for (e, pr, lab) in (("dominance", "dl", "Dominance ~ DL"), ("valence", "lstm", "Valence ~ LSTM")):
    s = w[(w.emotion == e) & (w.predictor == pr)]
    print(f"  {lab}, every model row in the wide family: Holm-significant {s.sig_holm.sum()} of {len(s)}")
print(f"wrote {OUT}\nDONE v{VERSION}")
