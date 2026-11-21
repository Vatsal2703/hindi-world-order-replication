#!/usr/bin/env python3
# v1.3 (2026-11-02): renumbered from 35_check_frequency_control.py; naming only, no logic change.
# v1.2 (2026-11-02): renamed from 35_emille_freq_control.py; naming only, no logic change.
"""
Script 27: EMILLE Frequency Control (from our own trigram model)  — v1.3

Supervisor request (6 Oct): compute word frequencies from the corpus the
surprisal models were trained on, rather than an external database. The
unigram counts are already inside trigram_model_emille.pkl, so no new data
is needed.

Why it matters: if surprisal comes from EMILLE and the frequency control
comes from a different corpus, part of what the control removes is the gap
between two corpora rather than frequency itself. A cross-check run with an
external database (53% vocabulary coverage) produced a spurious valence
effect that disappears under same-corpus control.

What it reports:
  [1] vocabulary coverage of our reference sentences
  [2] tertile means (uncontrolled, for reference)
  [3] partial Spearman correlations, n_words + EMILLE frequency partialled
  [4] hierarchical OLS on dominance -- does the effect survive?
  [5] valence under each level of control
  [6] the positive/negative asymmetry interaction

Version history
  v1.0  first version
  v1.1  fixed script number in the banner and run command (was 30); no change to results

RUN IN `digit` CONDA ENV:
    /Users/vatsalagrawal/miniconda3/envs/digit/bin/python \\
        scripts/27_check_frequency_control.py

Needs: pandas, numpy, scipy, statsmodels   (no sklearn)

Input : data/models/trigram_model_emille.pkl
        data/results/MTP2_raw_dataset_full.csv
Output: console report
        data/results/MTP2_refs_with_emille_freq.csv
"""

VERSION = "1.3"

import os
import sys
import math
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_base():
    d = SCRIPT_DIR
    while d and d != os.path.dirname(d):
        if os.path.basename(d) == "hindi-world-order-replication":
            return d
        if os.path.isdir(os.path.join(d, "hindi-world-order-replication")):
            return os.path.join(d, "hindi-world-order-replication")
        d = os.path.dirname(d)
    return "."


BASE  = find_base()
MODEL = os.path.join(BASE, "data", "models", "trigram_model_emille.pkl")
DATA  = os.path.join(BASE, "data", "results", "MTP2_raw_dataset_full.csv")
OUT   = os.path.join(BASE, "data", "results", "MTP2_refs_with_emille_freq.csv")

EMOTIONS   = ["tgt_valence", "tgt_arousal", "tgt_dominance"]
DIFFICULTY = ["trigram_pw", "deplen_resid"]


def residualise(y, X):
    """Residuals of y after regressing on X (statsmodels, no sklearn)."""
    Xc = sm.add_constant(np.asarray(X, dtype=float))
    return np.asarray(y, dtype=float) - sm.OLS(y, Xc).fit().predict(Xc)


def partial_spearman(x, y, C):
    return spearmanr(residualise(x, C), residualise(y, C))


def main():
    print("\n" + "=" * 88)
    print(f" SCRIPT 35: EMILLE FREQUENCY CONTROL  (v{VERSION})")
    print("=" * 88)

    for p in (MODEL, DATA):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing required file: {p}")

    # ------------------------------------------------------ [0] frequencies
    print(f"\n[0] Reading unigram counts from {os.path.basename(MODEL)}")
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    uni = model["unigrams"]
    total = sum(uni.values())
    # log10 frequency per million words -- a Zipf-like scale. Raw counts are
    # heavy-tailed and would dominate any linear model.
    logfreq = {w: math.log10(c / total * 1e6) for w, c in uni.items() if c > 0}
    median_lf = float(np.median(list(logfreq.values())))
    print(f"    {len(logfreq):,} word types   {total:,} tokens"
          f"   median log10(fpmw) = {median_lf:.3f}")

    # ------------------------------------------------------ [1] sentences
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    r = df[df["is_reference"] == 1].copy().reset_index(drop=True)
    print(f"    {len(r):,} reference sentences loaded")

    r["trigram_pw"] = r["trigram_surprisal"] / r["n_words"]
    r["deplen_resid"] = residualise(r["dep_len"].values, r[["n_words"]].values)

    def sent_freq(s):
        toks = str(s).split()
        vals = [logfreq.get(t) for t in toks]
        hit = [v for v in vals if v is not None]
        imp = [v if v is not None else median_lf for v in vals]
        return pd.Series({
            "em_cov": len(hit) / len(toks) if toks else 0.0,
            "em_logf": float(np.mean(imp)) if imp else np.nan,
        })

    r = pd.concat([r, r["sentence"].apply(sent_freq)], axis=1)

    print(f"\n[1] Vocabulary coverage")
    print(f"    mean {r['em_cov'].mean():.3f}   median {r['em_cov'].median():.3f}"
          f"   10th pct {r['em_cov'].quantile(.10):.3f}   min {r['em_cov'].min():.3f}")
    print(f"    mean sentence log10(fpmw) {r['em_logf'].mean():.3f}"
          f"  (sd {r['em_logf'].std():.3f})")

    # ------------------------------------------------------ [2] tertiles
    print("\n\n[2] Tertile means on per-word trigram (uncontrolled)")
    print("-" * 88)
    r["tri_bin"] = pd.qcut(r["trigram_pw"], 3, labels=["low", "mid", "high"])
    print(f"{'emotion':<16} {'low':>9} {'mid':>9} {'high':>9}   {'F':>7} {'p':>11}")
    for emo in EMOTIONS:
        g = [r.loc[r.tri_bin == b, emo] for b in ["low", "mid", "high"]]
        F, p = stats.f_oneway(*g)
        print(f"{emo:<16} {g[0].mean():+9.4f} {g[1].mean():+9.4f} "
              f"{g[2].mean():+9.4f}   {F:7.2f} {p:11.2e}")

    # ------------------------------------------------------ [3] partials
    print("\n\n[3] Partial Spearman — n_words + EMILLE frequency partialled")
    print("-" * 88)
    C = r[["n_words", "em_logf"]].values
    print(f"{'difficulty':<16} {'emotion':<16} {'rho':>10} {'p':>12}")
    for diff in DIFFICULTY:
        for emo in EMOTIONS:
            rho, p = partial_spearman(r[diff].values, r[emo].values, C)
            star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
            print(f"{diff:<16} {emo:<16} {rho:>10.4f} {p:>12.2e} {star}")
        print()

    # ------------------------------------------------------ [4] hierarchy
    print("\n[4] Hierarchical OLS on tgt_dominance")
    print("-" * 88)
    steps = [
        ("bivariate",          ["trigram_pw"]),
        ("+ length",           ["trigram_pw", "n_words"]),
        ("+ EMILLE frequency", ["trigram_pw", "n_words", "em_logf"]),
        ("+ lexicon coverage", ["trigram_pw", "n_words", "em_logf", "tgt_coverage"]),
    ]
    base = None
    for label, cols in steps:
        m = sm.OLS(r["tgt_dominance"], sm.add_constant(r[cols])).fit()
        b = m.params["trigram_pw"]
        if base is None:
            base, att = b, ""
        else:
            att = f"   attenuation {100*(1 - b/base):5.1f}%"
        print(f"  {label:<22} beta={b:+.5f}  t={m.tvalues['trigram_pw']:+6.2f}"
              f"  p={m.pvalues['trigram_pw']:.3e}  R2={m.rsquared:.4f}{att}")

    # ------------------------------------------------------ [5] valence
    print("\n\n[5] Valence under each level of control")
    print("-" * 88)
    rho, p = spearmanr(r["trigram_pw"], r["tgt_valence"])
    print(f"  raw                               rho={rho:+.4f}  p={p:.3e}")
    rho, p = partial_spearman(r["trigram_pw"].values, r["tgt_valence"].values,
                              r[["n_words"]].values)
    print(f"  length partialled                 rho={rho:+.4f}  p={p:.3e}")
    rho, p = partial_spearman(r["trigram_pw"].values, r["tgt_valence"].values, C)
    print(f"  length + EMILLE freq partialled   rho={rho:+.4f}  p={p:.3e}")
    print(f"\n  r(em_logf, tgt_valence) = {r['em_logf'].corr(r['tgt_valence']):+.4f}")

    # ------------------------------------------------------ [6] asymmetry
    print("\n\n[6] Positive/negative asymmetry, EMILLE frequency in the model")
    print("-" * 88)
    sub = r[r["tgt_valence"] != 0].copy()
    sub["is_neg"] = (sub["tgt_valence"] < 0).astype(int)
    for diff in DIFFICULTY:
        m = smf.ols(f"tgt_dominance ~ {diff}*is_neg + n_words + em_logf",
                    data=sub).fit()
        k = f"{diff}:is_neg"
        verdict = "DIFFERENT" if m.pvalues[k] < .05 else "no difference"
        print(f"  {diff:<14} interaction beta={m.params[k]:+.5f}"
              f"  t={m.tvalues[k]:+5.2f}  p={m.pvalues[k]:.3e}  -> {verdict}")

    for name, grp in [("positive", sub[sub.is_neg == 0]),
                      ("negative", sub[sub.is_neg == 1])]:
        g = grp.copy()
        g["b"] = pd.qcut(g["trigram_pw"], 3, labels=["low", "mid", "high"])
        lo = g.loc[g.b == "low", "tgt_dominance"]
        hi = g.loc[g.b == "high", "tgt_dominance"]
        pooled = np.sqrt(((len(lo)-1)*lo.var(ddof=1) + (len(hi)-1)*hi.var(ddof=1))
                         / (len(lo)+len(hi)-2))
        print(f"  {name:<9} n={len(g):>5,}  low {lo.mean():+.4f}"
              f"  high {hi.mean():+.4f}   d={(hi.mean()-lo.mean())/pooled:+.3f}")

    # ------------------------------------------------------ save
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r.drop(columns=["tri_bin"]).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n  Saved -> {OUT}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()