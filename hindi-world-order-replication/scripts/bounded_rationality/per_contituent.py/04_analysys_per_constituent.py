#!/usr/bin/env python3
"""
Analyse per-constituent surprisal.
Bounded Rationality project — reads CSV from per_conti.py.

Pipeline position: step 5 of 5
  ... -> per_conti.py -> [THIS]

Input:  data/features/per_constituent_surprisal.csv
Output: data/features/surprisal_position_signature.png
        data/features/surprisal_gap_by_k.png

Run from project root:
  python scripts/bounded_rationality/per_contituent.py/analyse_per_constituent.py

Toggle PRIMARY below to switch between "sum" and "mean" aggregation.
"""
from __future__ import annotations
import math, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_IN   = "./data/features/per_constituent_surprisal.csv"
PLOT_OUT = "./data/features/surprisal_position_signature.png"
BYK_OUT  = "./data/features/surprisal_gap_by_k.png"

PRIMARY = "mean"   # "sum" or "mean"


def welch_t(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float("nan")
    se = math.sqrt(a.var(ddof=1)/na + b.var(ddof=1)/nb)
    return (a.mean() - b.mean()) / se if se else float("nan")


def position_means(frame: pd.DataFrame, max_dist: int = 6) -> dict[int, float]:
    acc = {-(d): [] for d in range(1, max_dist + 1)}
    for _, row in frame.iterrows():
        k = int(row["k"])
        for i in range(1, k + 1):
            c = f"const{i}_surp_{PRIMARY}"
            if c not in row or pd.isna(row[c]): continue
            p = -(k - i + 1)
            if p in acc: acc[p].append(row[c])
    return {p: (sum(v)/len(v) if v else float("nan")) for p, v in acc.items()}


def main():
    if not os.path.exists(CSV_IN):
        print(f"ERROR: {CSV_IN} not found. Run per_conti.py first."); return 1

    df = pd.read_csv(CSV_IN)
    df["is_reference"] = df["is_reference"].astype(bool)
    R, V = df[df.is_reference], df[~df.is_reference]
    col  = f"last_surp_{PRIMARY}"

    print("\n" + "=" * 70)
    print(" PER-CONSTITUENT SURPRISAL ANALYSIS")
    print("=" * 70)
    print(f"  rows={len(df):,}  references={len(R):,}  variants={len(V):,}")
    print(f"  aggregation: {PRIMARY}  (column: {col})")

    # 1. Verb-adjacent signal
    print("\n" + "-" * 70)
    print(" 1. VERB-ADJACENT SURPRISAL: reference vs variant")
    print("-" * 70)
    r_m, v_m = R[col].mean(), V[col].mean()
    t = welch_t(R[col].dropna(), V[col].dropna())
    print(f"  reference mean : {r_m:7.3f}")
    print(f"  variant   mean : {v_m:7.3f}")
    print(f"  difference     : {r_m - v_m:+7.3f}  (negative = refs more predictable)")
    print(f"  Welch t        : {t:7.2f}   p {'< 1e-4' if abs(t) > 3.9 else '> 1e-4'}")
    print(f"  => {'LOWER-surprisal verb-adjacent in references — least-effort confirmed.' if r_m < v_m else 'No signal.'}")
    rt, vt = R["sentence_total_surprisal"].mean(), V["sentence_total_surprisal"].mean()
    print(f"\n  contrast — sentence-total: ref {rt:.2f}  var {vt:.2f}  diff {rt-vt:+.2f}")

    # 2. By-k gap
    print("\n" + "-" * 70)
    print(" 2. VERB-ADJACENT GAP BY k")
    print("-" * 70)
    print(f"  {'k':>3} {'n_ref':>7} {'ref':>9} {'var':>9} {'gap':>8}")
    ks, gaps, ns = [], [], []
    for k in sorted(df["k"].unique()):
        rk = R[R["k"]==k][col].dropna()
        vk = V[V["k"]==k][col].dropna()
        if len(rk) < 5 or len(vk) < 5: continue
        gap = rk.mean() - vk.mean()
        ks.append(int(k)); gaps.append(gap); ns.append(len(rk))
        print(f"  {int(k):>3} {len(rk):>7,} {rk.mean():>9.3f} {vk.mean():>9.3f} {gap:>+8.3f}")

    # 3. Position signature
    print("\n" + "-" * 70)
    print(" 3. POSITION SIGNATURE")
    print("-" * 70)
    rs, vs = position_means(R), position_means(V)
    pos = sorted(rs.keys())
    print(f"  {'pos':>4} {'ref':>9} {'var':>9}")
    for p in pos: print(f"  {p:>4} {rs[p]:>9.3f} {vs[p]:>9.3f}")

    # Plot 1 — position signature
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pos, [rs[p] for p in pos], "o-", label="Reference", lw=2)
    ax.plot(pos, [vs[p] for p in pos], "s--", label="Variant", lw=2)
    ax.set_xlabel("Position from main verb  (-1 = verb-adjacent)")
    ax.set_ylabel(f"Avg constituent surprisal ({PRIMARY}, bits)")
    ax.set_title("Preverbal Constituent Surprisal Signature (trigram)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_OUT, dpi=130)
    print(f"\n  Saved -> {PLOT_OUT}")

    # Plot 2 — gap by k
    if ks:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.bar([str(k) for k in ks], gaps, color="#1E2761")
        for b, n, g in zip(bars, ns, gaps):
            ax2.text(b.get_x()+b.get_width()/2, g/2, f"n={n}",
                     ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_xlabel("Number of preverbal constituents (k)")
        ax2.set_ylabel(f"Verb-adjacent surprisal gap (ref - var, {PRIMARY})")
        ax2.set_title("Least-effort surprisal signal by sentence complexity")
        fig2.tight_layout(); fig2.savefig(BYK_OUT, dpi=130)
        print(f"  Saved -> {BYK_OUT}")

    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())