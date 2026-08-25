#!/usr/bin/env python3
"""
Step 5 — Analysis & Plots  (v2 — corrected position plot)
Bounded Rationality project — surprisal extension.

Pipeline position: final analysis step
  01_data_preparation -> 02_variant_generation -> 03_train_trigram -> 04_per_conti -> [THIS]

Input:
  data/features/per_constituent_surprisal.csv

Outputs:
  data/figures/position_ref_vs_var_panels_{PRIMARY}.png   (per-k ref vs variant)
  data/figures/verb_adjacent_ref_vs_var_{PRIMARY}.png     (headline result)
  data/figures/surprisal_gap_by_k_{PRIMARY}.png           (gap bar chart)

Run from project root:
  python scripts/bounded_rationality/analyse_per_constituent.py

Toggle PRIMARY = 'mean' or 'sum' below.

CHANGE LOG
  v2: position signature now overlays REFERENCE vs VARIANT per k
      (v1 plotted references only, which cannot show the least-effort effect
       and was confounded by constituent length). Mapping const{k}=verb-adjacent
       verified by diagnostic_position_map.py.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# CONFIG
# ============================================================================
INPUT_CSV   = "./data/features/per_constituent_surprisal.csv"
FIGURES_DIR = "./data/figures"
PRIMARY     = "mean"   # 'mean' or 'sum'
VERSION     = "v2"
# ============================================================================

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print(f" PER-CONSTITUENT SURPRISAL ANALYSIS  ({VERSION})")
print("=" * 70)

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df["is_reference"] = df["is_reference"].astype(bool)
print(f"  rows={len(df):,}  references={df['is_reference'].sum():,}  "
      f"variants={(~df['is_reference']).sum():,}")
print(f"  aggregation: {PRIMARY}")

last_col = f"last_surp_{PRIMARY}"


# ── 2. Verb-adjacent surprisal: reference vs variant ─────────────────────────
print("\n" + "-" * 70)
print(" 1. VERB-ADJACENT SURPRISAL: reference vs variant")
print("-" * 70)

ref_vals = df[df["is_reference"]][last_col].dropna()
var_vals = df[~df["is_reference"]][last_col].dropna()
t, p = stats.ttest_ind(ref_vals, var_vals, equal_var=False)

print(f"  reference : {ref_vals.mean():.4f} bits  (n={len(ref_vals):,})")
print(f"  variant   : {var_vals.mean():.4f} bits  (n={len(var_vals):,})")
print(f"  Welch t   : {t:.2f}   p = {p:.3e}")
rt = df[df["is_reference"]]["sentence_total_surprisal"].mean()
vt = df[~df["is_reference"]]["sentence_total_surprisal"].mean()
print(f"  contrast — sentence-total: ref {rt:.2f}  var {vt:.2f}  diff {rt-vt:+.2f}")
print(f"  => {'least-effort confirmed (refs lower verb-adjacent)' if ref_vals.mean() < var_vals.mean() else 'NO SIGNAL'}")


# ── 3. By-k gap table ────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print(" 2. VERB-ADJACENT GAP BY k")
print("-" * 70)
print(f"  {'k':>3} {'n_ref':>7} {'ref':>9} {'var':>9} {'gap(v-r)':>10}")

ks, gaps, ns = [], [], []
for k in sorted(df["k"].unique()):
    rk = df[df["is_reference"] & (df["k"] == k)][last_col].dropna()
    vk = df[~df["is_reference"] & (df["k"] == k)][last_col].dropna()
    if len(rk) < 5 or len(vk) < 5:
        continue
    gap = vk.mean() - rk.mean()
    ks.append(int(k)); gaps.append(gap); ns.append(len(rk))
    print(f"  {int(k):>3} {len(rk):>7,} {rk.mean():>9.3f} {vk.mean():>9.3f} {gap:>+10.3f}")


# ── 4. Position signature — REF vs VARIANT per k (CORRECTED) ──────────────────
print("\n" + "-" * 70)
print(" 3. POSITION SIGNATURE — reference vs variant, per k")
print("-" * 70)


def position_means(sub, k, is_ref):
    """position 1 = verb-adjacent. const{k}=verb-adjacent (verified)."""
    grp = sub[sub["is_reference"] == is_ref]
    positions, means = [], []
    for pos_from_verb in range(1, k + 1):
        const_idx = k - pos_from_verb + 1
        col = f"const{const_idx}_surp_{PRIMARY}"
        if col in grp.columns and len(grp) > 0:
            positions.append(pos_from_verb)
            means.append(grp[col].mean())
    return positions, means


panel_ks = [k for k in [2, 3, 4, 5, 6] if (df["k"] == k).any()]
n = len(panel_ks)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), sharey=True)
if n == 1:
    axes = [axes]

for ax, k in zip(axes, panel_ks):
    sub = df[df["k"] == k]
    rp, rm = position_means(sub, k, True)
    vp, vm = position_means(sub, k, False)
    ax.plot(rp, rm, "o-",  color="darkred", label="Reference", linewidth=2)
    ax.plot(vp, vm, "x--", color="gray",    label="Variant",   linewidth=1.5)
    ax.set_title(f"k = {k}", fontsize=11)
    ax.set_xlabel("Position from verb\n(1 = verb-adjacent)", fontsize=9)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel(f"Mean constituent surprisal ({PRIMARY}, bits)", fontsize=10)
fig.suptitle("Surprisal by position: Reference vs Variant (per k)", fontsize=13)
plt.tight_layout()
out1 = f"{FIGURES_DIR}/position_ref_vs_var_panels_{PRIMARY}.png"
plt.savefig(out1, dpi=150); plt.close()
print(f"  Saved -> {out1}")


# ── 5. Verb-adjacent headline plot ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ref_adj, var_adj = [], []
for k in panel_ks:
    sub = df[df["k"] == k]
    ref_adj.append(sub[sub["is_reference"]][last_col].mean())
    var_adj.append(sub[~sub["is_reference"]][last_col].mean())

ax.plot(panel_ks, ref_adj, "o-",  color="darkred", label="Reference", linewidth=2)
ax.plot(panel_ks, var_adj, "x--", color="gray",    label="Variant",   linewidth=1.5)
ax.set_xlabel("Number of preverbal constituents (k)", fontsize=11)
ax.set_ylabel(f"Verb-adjacent surprisal ({PRIMARY}, bits)", fontsize=11)
ax.set_title("Verb-adjacent surprisal: Reference vs Variant", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
out2 = f"{FIGURES_DIR}/verb_adjacent_ref_vs_var_{PRIMARY}.png"
plt.savefig(out2, dpi=150); plt.close()
print(f"  Saved -> {out2}")


# ── 6. Gap-by-k bar chart ────────────────────────────────────────────────────
if ks:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(k) for k in ks], gaps, color="#1E2761")
    for b, nn, g in zip(bars, ns, gaps):
        ax.text(b.get_x() + b.get_width() / 2, g / 2, f"n={nn}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Number of preverbal constituents (k)", fontsize=11)
    ax.set_ylabel(f"Verb-adjacent gap (var - ref, {PRIMARY})", fontsize=11)
    ax.set_title("Least-effort surprisal signal by sentence complexity", fontsize=12)
    plt.tight_layout()
    out3 = f"{FIGURES_DIR}/surprisal_gap_by_k_{PRIMARY}.png"
    plt.savefig(out3, dpi=150); plt.close()
    print(f"  Saved -> {out3}")

print("\n" + "=" * 70)
print(f" [OK] Analysis complete ({VERSION}).")
print(" Least-effort signal = reference line BELOW variant line at each position.")
print("=" * 70 + "\n")