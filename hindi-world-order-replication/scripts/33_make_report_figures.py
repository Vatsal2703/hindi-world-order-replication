#!/usr/bin/env python3
# v1.3 (2026-11-02): renumbered from 41_make_report_figures.py; naming only, no logic change.
# v1.2 (2026-11-02): renamed from 41_report_figures.py; naming only, no logic change.
"""
33_make_report_figures.py  (v1.3)  -- v1.1: legends moved below the plots, title fits, DL footnote

Builds the figures and tables for the final MTP report straight from the
results files, so every number in the report can be traced to a file.

Inputs
  data/results/table4_results.csv        (script 15: MTP-I accuracy, ours vs paper)
  data/results/table2_regression.csv     (script 15: MTP-I full-data regression)
  data/results/table4_results_v2lstm.csv (script 15 with the retrained LSTM)
  data/results/lstm_v2_regressions.csv   (script 30: MTP-II final table)
  toolkit/ranker.json                    (toolkit word-order ranker)
Outputs (data/report/)
  fig1_mtp1_accuracy.png   MTP-I accuracy per feature set, ours vs paper
  fig2_mtp2_effects.png    MTP-II standardised effects, HUTB vs DISCO
  report_tables.md         Tables 1-5 in Markdown (paste into Word/Docs)

RUN:  python scripts/33_make_report_figures.py      (needs pandas + matplotlib)
"""
import os, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "1.3"
print(f"=== 33_make_report_figures.py v{VERSION} ===")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(BASE, *p)
OUT = R("data", "report"); os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
C_OURS, C_PAPER = "#2563eb", "#9ca3af"
C_HUTB, C_DISCO = "#2563eb", "#d97706"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})

NAMES = {"a": "IS score", "b": "Dependency length", "c": "PCFG surprisal", "d": "Lex-rept surprisal",
         "e": "Trigram surprisal", "f": "LSTM surprisal", "g": "Adaptive LSTM surprisal",
         "base1": "base1 (a–f)", "base1+g": "base1 + g (full)", "base2": "base2 (no lex-rept)",
         "base2+g": "base2 + g"}

# ---------------------------------------------------------------- MTP-I ----
t4 = pd.read_csv(R("data/results/table4_results.csv"))
t4["name"] = t4.model.map(NAMES)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
y = range(len(t4))[::-1]
h = 0.38
ax.barh([i + h / 2 for i in y], t4.full_accuracy, h, color=C_OURS, label="This replication")
ax.barh([i - h / 2 for i in y], t4.paper_full, h, color=C_PAPER, label="Ranjan et al. (2022)")
for i, (o, p) in zip(y, zip(t4.full_accuracy, t4.paper_full)):
    ax.text(o + 0.4, i + h / 2, f"{o:.2f}", va="center", fontsize=8, color=INK)
    ax.text(p + 0.4, i - h / 2, f"{p:.2f}", va="center", fontsize=8, color=MUTED)
ax.set_yticks(list(y)); ax.set_yticklabels(t4.name)
ax.set_xlim(45, 100); ax.set_xlabel("10-fold CV accuracy, full data (%)")
ax.axvline(50, color=GRID, lw=1, zorder=0)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False, fontsize=9)
ax.set_title("MTP-I: predicting the attested Hindi word order (178,722 pairs)", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_mtp1_accuracy.png"), dpi=200); plt.close(fig)
print("wrote fig1_mtp1_accuracy.png")

# ---------------------------------------------------------------- MTP-II ---
rg = pd.read_csv(R("data/results/lstm_v2_regressions.csv"))
rows = [("dominance", "LSTMv2", "dl", "Dominance ~ dependency length"),
        ("valence", "LSTMv2", "lstm", "Valence ~ LSTM surprisal"),
        ("dominance", "LSTMv2", "lstm", "Dominance ~ LSTM surprisal"),
        ("arousal", "LSTMv2", "lstm", "Arousal ~ LSTM surprisal"),
        ("valence", "trigram", "tri", "Valence ~ trigram surprisal"),
        ("dominance", "trigram", "tri", "Dominance ~ trigram surprisal"),
        ("arousal", "trigram", "tri", "Arousal ~ trigram surprisal")]
def get(c, e, m, p):
    r = rg[(rg.corpus == c) & (rg.emotion == e) & (rg.model == m) & (rg.predictor == p)].iloc[0]
    return float(r.beta), float(r.p)
tab = []
for e, m, p, lab in rows:
    bh, ph = get("HUTB", e, m, p); bd, pd_ = get("DISCO", e, m, p)
    tab.append((lab, bh, ph, bd, pd_))
fig, ax = plt.subplots(figsize=(7.6, 4.8))
y = range(len(tab))[::-1]; h = 0.36
for i, (lab, bh, ph, bd, pdv) in zip(y, tab):
    for off, b, pv, col in ((h / 2, bh, ph, C_HUTB), (-h / 2, bd, pdv, C_DISCO)):
        sig = pv < 0.05
        ax.barh(i + off, b, h, color=col if sig else "white", edgecolor=col, lw=1.2)
        ax.text(b + (0.006 if b >= 0 else -0.006), i + off, f"{b:+.3f}" + ("" if sig else " n.s."),
                va="center", ha="left" if b >= 0 else "right", fontsize=8, color=INK)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(list(y)); ax.set_yticklabels([t[0] for t in tab])
ax.set_xlim(-0.26, 0.26); ax.set_xlabel("standardised β (length and frequency controlled; HC3 SE)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_HUTB, label="HUTB (n = 2,779)"), Patch(color=C_DISCO, label="DISCO (n = 3,171)"),
                   Patch(facecolor="white", edgecolor=MUTED, label="not significant (p ≥ .05)")],
          loc="upper center", bbox_to_anchor=(0.4, -0.14), ncol=3, frameon=False, fontsize=8)
ax.set_title("MTP-II: difficulty and emotion, HUTB vs DISCO", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_mtp2_effects.png"), dpi=200); plt.close(fig)
print("wrote fig2_mtp2_effects.png")

# ---------------------------------------------------------------- tables ---
def pfmt(p):
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")
md = [f"<!-- generated by scripts/33_make_report_figures.py v{VERSION}; do not edit numbers by hand -->\n"]
md.append("## Table 1. MTP-I prediction accuracy (10-fold CV, %), this replication vs Ranjan et al. (2022)\n")
md.append("| Model | Full (ours) | Full (paper) | DO-fronted (ours) | DO (paper) | IO-fronted (ours) | IO (paper) |")
md.append("|---|---|---|---|---|---|---|")
for _, r in t4.iterrows():
    md.append(f"| {r['name']} | {r.full_accuracy:.2f} | {r.paper_full:.2f} | {r.do_accuracy:.2f} | {r.paper_do:.2f} | "
              f"{r.io_accuracy:.2f} | {r.paper_io:.2f} |")
md.append("\nFull data: 178,722 pairs (2,753 reference sentences). DO-fronted: 6,622 pairs (145 sentences). "
          "IO-fronted: 5,080 pairs (82 sentences).\n")

t2 = pd.read_csv(R("data/results/table2_regression.csv"))
md.append("## Table 2. MTP-I logistic regression, full data (N = 178,722)\n")
md.append("| Predictor | β | SE | t |\n|---|---|---|---|")
for _, r in t2.iterrows():
    md.append(f"| {r.predictor} | {r.beta:+.4f} | {r.se:.4f} | {r.t:.2f} |")

v2p = R("data/results/table4_results_v2lstm.csv")
if os.path.exists(v2p):
    v2 = pd.read_csv(v2p).set_index("model")
    md.append("\n## Table 3. Sensitivity of MTP-I to the LSTM used (full data, %)\n")
    md.append("| Model | Original LSTM | Retrained LSTM | Paper |\n|---|---|---|---|")
    for m in ["f", "base1+g", "base2+g"]:
        o = t4.set_index("model").loc[m]
        md.append(f"| {NAMES[m]} | {o.full_accuracy:.2f} | {v2.loc[m, 'full_accuracy']:.2f} | {o.paper_full:.2f} |")

md.append("\n## Table 4. MTP-II standardised effects (β, p), length and frequency controlled\n")
md.append("| Emotion ~ predictor | HUTB β | HUTB p | DISCO β | DISCO p | Same direction and both significant? |")
md.append("|---|---|---|---|---|---|")
for lab, bh, ph, bd, pdv in tab:
    ok = "yes" if (ph < .05 and pdv < .05 and (bh > 0) == (bd > 0)) else "no"
    md.append(f"| {lab} | {bh:+.3f} | {pfmt(ph)} | {bd:+.3f} | {pfmt(pdv)} | {ok} |")
md.append("\nOne surprisal measure per model, plus dependency length and controls (HUTB: log length, EMILLE log "
          "frequency; DISCO: log length, mean Zipf frequency, task domain). OLS on z-scored variables, HC3 SE. "
          "The dependency-length row is from the LSTM model; in the trigram model it is -0.070 (HUTB) and -0.086 (DISCO).\n")

rk = json.load(open(R("toolkit/ranker.json"), encoding="utf-8"))
md.append("## Table 5. Toolkit word-order ranker (10-fold CV accuracy, %)\n")
md.append("| Features | Accuracy |\n|---|---|")
for k, a in rk["cv_accuracy"].items():
    md.append(f"| {k} | {a:.2f} |")
md.append(f"\nTrained on {rk['n_sentences']:,} HUTB sentences ({rk['n_pairs']:,} pairs).")
open(os.path.join(OUT, "report_tables.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("wrote report_tables.md")
print(f"DONE v{VERSION}")
