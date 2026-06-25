#!/usr/bin/env python3
"""
Step 5 — Analysis & Plots
Bounded Rationality project — surprisal extension.

Pipeline position: final step
  01_data_preparation -> 02_variant_generation -> 03_train_trigram -> 04_per_conti -> [THIS]

Input:
  data/features/per_constituent_surprisal.csv

Outputs:
  data/figures/position_signature_{PRIMARY}.png
  data/figures/total_surprisal_by_k_{PRIMARY}.png

Run from project root:
  python scripts/bounded_rationality/05_analyse.py

Toggle PRIMARY = 'mean' or 'sum' below.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# CONFIG — edit these paths if running locally (not Colab)
# ============================================================================
INPUT_CSV   = "./data/features/per_constituent_surprisal.csv"
FIGURES_DIR = "./data/figures"

PRIMARY = "mean"   # <-- toggle to 'sum' for sum-aggregation plots
# ============================================================================

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


# ── 1. Load CSV ──────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"  -> {len(df):,} rows loaded")
print(f"  References : {df['is_reference'].sum():,}")
print(f"  Non-ref    : {(df['is_reference']==0).sum():,}")


# ── 2. Verb-adjacent surprisal: reference vs variant ─────────────────────────
print(f"\n--- Verb-adjacent surprisal ({PRIMARY}) ---")
col = f"last_surp_{PRIMARY}"

ref_vals = df[df["is_reference"] == 1][col].dropna()
var_vals = df[df["is_reference"] == 0][col].dropna()

t, p = stats.ttest_ind(ref_vals, var_vals, equal_var=False)

print(f"  References : {ref_vals.mean():.4f} bits/word  (n={len(ref_vals):,})")
print(f"  Variants   : {var_vals.mean():.4f} bits/word  (n={len(var_vals):,})")
print(f"  Welch t    : {t:.2f}")
print(f"  p-value    : {p:.4e}")
print(f"  More predictable verb-adjacent in refs: {ref_vals.mean() < var_vals.mean()}")


# ── 3. By-k gap table ────────────────────────────────────────────────────────
print(f"\n--- By-k gap table ({PRIMARY}) ---")
col = f"last_surp_{PRIMARY}"

rows = []
for k in sorted(df["k"].unique()):
    sub = df[df["k"] == k]
    r = sub[sub["is_reference"] == 1][col].mean()
    v = sub[sub["is_reference"] == 0][col].mean()
    gap = v - r
    t_k, p_k = stats.ttest_ind(
        sub[sub["is_reference"] == 1][col].dropna(),
        sub[sub["is_reference"] == 0][col].dropna(),
        equal_var=False
    )
    rows.append({"k": k, "ref_mean": r, "var_mean": v, "gap (v-r)": gap, "t": t_k, "p": p_k})

gap_df = pd.DataFrame(rows)
print(gap_df.to_string(index=False, float_format="{:.4f}".format))


# ── 4. Position signature plot ────────────────────────────────────────────────
print(f"\n--- Generating position signature plot ({PRIMARY}) ---")

colors  = {2: "blue", 3: "red", 4: "orange", 5: "green", 6: "purple"}
markers = {2: "o",    3: "^",   4: "s",      5: "*",     6: "D"}

fig, ax = plt.subplots(figsize=(7, 5))

refs_only = df[df["is_reference"] == 1]

for k in [2, 3, 4, 5, 6]:
    sub = refs_only[refs_only["k"] == k]
    if len(sub) == 0:
        continue

    positions, means = [], []
    for pos_from_verb in range(1, k + 1):
        const_idx = k - pos_from_verb + 1
        c = f"const{const_idx}_surp_{PRIMARY}"
        if c in sub.columns:
            positions.append(pos_from_verb)
            means.append(sub[c].mean())

    ax.plot(positions, means,
            color=colors.get(k, "black"),
            marker=markers.get(k, "o"),
            label=str(k),
            linewidth=1.5)

ax.set_xlabel("Constituent position from main verb (1 = verb-adjacent)", fontsize=11)
ax.set_ylabel(f"Average constituent surprisal ({PRIMARY}, bits)", fontsize=11)
ax.set_title("Surprisal by position from verb (references only)", fontsize=12)
ax.legend(title="k (preverbal constituents)", fontsize=9)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

plt.tight_layout()
out = f"{FIGURES_DIR}/position_signature_{PRIMARY}.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")


# ── 5. By-k total surprisal plot ──────────────────────────────────────────────
print(f"\n--- Generating by-k total surprisal plot ({PRIMARY}) ---")

df["total_surp_norm"] = df["sentence_total_surprisal"] / df["k"]

k_vals    = sorted(df["k"].unique())
ref_means = [df[(df["k"]==k) & (df["is_reference"]==1)]["total_surp_norm"].mean() for k in k_vals]
var_means = [df[(df["k"]==k) & (df["is_reference"]==0)]["total_surp_norm"].mean() for k in k_vals]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_vals, ref_means, "o-",  color="darkred", label="Corpus (reference)", linewidth=2)
ax.plot(k_vals, var_means, "x--", color="gray",    label="Random Order (variants)", linewidth=1.5)

ax.set_xlabel("Number of preverbal constituents (k)", fontsize=11)
ax.set_ylabel("Avg total surprisal\n(normalized by k)", fontsize=11)
ax.set_title("Total surprisal by k: corpus vs random order", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{FIGURES_DIR}/total_surprisal_by_k_{PRIMARY}.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

print("\n[OK] Step 5 complete.")