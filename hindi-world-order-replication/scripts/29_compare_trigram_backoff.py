#!/usr/bin/env python3
"""
Script 29: Compare Trigram Retraining vs Backoff Fix  — v1.0

MTP-II: the trigram surprisal gap (reference-minus-variant) is a clear
outlier relative to lstm/adaptive/pcfg. Two suspected causes:
  (a) the trigram model (data/models/trigram_model.pkl, 401K tokens) is
      trained on far less data than the LSTM (EMILLE, ~11M+ tokens);
  (b) trigram_surprisal()'s backoff jumps straight from trigram to unigram,
      skipping the bigram level entirely when the trigram is unseen.

This script isolates how much each fix contributes by scoring the SAME
sample of sentences (reference + all its variants) four ways:
  1. old model (trigram_model.pkl)         + old backoff (tri -> uni)
  2. old model                             + new backoff (tri -> bi -> uni)
  3. new model (trigram_model_emille.pkl)  + old backoff
  4. new model                             + new backoff

For each combination it reports mean reference surprisal, mean variant
surprisal, and the ref-minus-var gap. Read-only: writes nothing to disk,
does not touch scripts/26_raw_dataset_full.py's own trigram_model.pkl input
or either model file.

RUN IN ANY ENV (pure Python, no torch needed):
    python scripts/29_compare_trigram_backoff.py --limit 200

Output: printed comparison table only.
"""

VERSION = "1.0"

import os
import sys
import math
import pickle
import argparse
from itertools import permutations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
from utils.paths import find_base
from parsers.ud_parser import Sentence, Word  # noqa: F401  (needed to unpickle)

BASE           = find_base(__file__)
SENTS_IN       = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
OLD_MODEL_IN   = os.path.join(BASE, "data", "models", "trigram_model.pkl")
NEW_MODEL_IN   = os.path.join(BASE, "data", "models", "trigram_model_emille.pkl")

MAX_VARIANTS = 99

# ============================================================================
# ORDERING RECONSTRUCTION (duplicated from scripts/26_raw_dataset_full.py so
# this script is self-contained)
# ============================================================================

def get_subtree(sent, head_idx):
    result = {head_idx}
    for w in sent.words:
        if w.head == head_idx and w.idx != head_idx:
            result |= get_subtree(sent, w.idx)
    return result


def build_orders(sent):
    if getattr(sent, "root_idx", None) is None:
        return None, []
    children = sent.get_children(sent.root_idx)
    preverbal = sorted([w for w in children
                        if w.idx < sent.root_idx and w.upos != "PUNCT"],
                       key=lambda w: w.idx)
    if len(preverbal) < 2:
        return None, []
    blocks = [sorted(get_subtree(sent, pw.idx)) for pw in preverbal]
    all_pre = set(i for b in blocks for i in b)
    post_v = [w.idx for w in sent.words if w.idx not in all_pre]
    ref = [i for b in blocks for i in b] + post_v
    variants, seen = [], {tuple(ref)}
    for perm in permutations(range(len(blocks))):
        order = [i for k in perm for i in blocks[k]] + post_v
        if tuple(order) not in seen:
            seen.add(tuple(order))
            variants.append(order)
            if len(variants) >= MAX_VARIANTS:
                break
    return ref, variants

# ============================================================================
# TRIGRAM SCORING — old backoff (tri -> uni) vs new backoff (tri -> bi -> uni)
# ============================================================================

def trigram_total_tokens(model):
    return max(sum(model.get("unigrams", {}).values()), 1)


def _word_prob_old(model, w1, w2, tok, total_uni):
    tri = model["trigrams"].get((w1, w2, tok), 0)
    bi  = model["bigrams"].get((w1, w2), 0)
    uni = model["unigrams"].get(tok, 0)
    if tri > 0 and bi > 0:
        return tri / bi
    if uni > 0:
        return uni / total_uni
    return 1e-10


def _word_prob_new(model, w1, w2, w3, total_uni):
    tri_num = model["trigrams"].get((w1, w2, w3), 0)
    tri_den = model["bigrams"].get((w1, w2), 0)
    if tri_num > 0 and tri_den > 0:
        return tri_num / tri_den

    bi_num = model["bigrams"].get((w2, w3), 0)
    bi_den = model["unigrams"].get(w2, 0)
    if bi_num > 0 and bi_den > 0:
        return bi_num / bi_den

    uni = model["unigrams"].get(w3, 0)
    if uni > 0:
        return uni / total_uni

    return 1e-10


def trigram_surprisal(model, tokens, total_uni, word_prob_fn):
    total, hist = 0.0, ["<s>", "<s>"]
    for tok in tokens:
        p = word_prob_fn(model, hist[-2], hist[-1], tok, total_uni)
        total += -math.log2(min(max(p, 1e-10), 1.0))
        hist.append(tok)
    return total

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="first N sentences")
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print(f" SCRIPT 29: TRIGRAM RETRAIN vs BACKOFF FIX COMPARISON  (v{VERSION})")
    print("=" * 78)

    with open(OLD_MODEL_IN, "rb") as f:
        old_model = pickle.load(f)
    with open(NEW_MODEL_IN, "rb") as f:
        new_model = pickle.load(f)
    old_total = trigram_total_tokens(old_model)
    new_total = trigram_total_tokens(new_model)
    print(f"\n  Old model : {old_total:,} tokens, vocab {old_model['vocab_size']:,}  ({OLD_MODEL_IN})")
    print(f"  New model : {new_total:,} tokens, vocab {new_model['vocab_size']:,}  ({NEW_MODEL_IN})")

    with open(SENTS_IN, "rb") as f:
        sentences = pickle.load(f)
    sentences = sentences[: args.limit]
    print(f"\n  Sample    : first {len(sentences)} sentences from {os.path.basename(SENTS_IN)}")

    ref_rows, var_rows = [], []
    for sent in sentences:
        ref_order, variants = build_orders(sent)
        if ref_order is None:
            continue
        n_words = len(sent.words)
        ref_toks = [sent.words[idx - 1].form for idx in ref_order if 0 < idx <= n_words]
        ref_rows.append(ref_toks)
        for order in variants:
            var_toks = [sent.words[idx - 1].form for idx in order if 0 < idx <= n_words]
            var_rows.append(var_toks)

    print(f"  Rows      : {len(ref_rows)} reference, {len(var_rows)} variant\n")

    combos = [
        ("1. old model + old backoff (tri->uni)      ", old_model, old_total, _word_prob_old),
        ("2. old model + new backoff (tri->bi->uni)   ", old_model, old_total, _word_prob_new),
        ("3. new EMILLE model + old backoff           ", new_model, new_total, _word_prob_old),
        ("4. new EMILLE model + new backoff           ", new_model, new_total, _word_prob_new),
    ]

    print("-" * 78)
    print(f"  {'combination':<45s} {'mean ref':>10s} {'mean var':>10s} {'gap':>10s}")
    print("-" * 78)
    for label, model, total_uni, fn in combos:
        ref_scores = [trigram_surprisal(model, toks, total_uni, fn) for toks in ref_rows]
        var_scores = [trigram_surprisal(model, toks, total_uni, fn) for toks in var_rows]
        mean_ref = sum(ref_scores) / len(ref_scores)
        mean_var = sum(var_scores) / len(var_scores)
        gap = mean_ref - mean_var
        print(f"  {label:<45s} {mean_ref:>10.2f} {mean_var:>10.2f} {gap:>10.2f}")
    print("-" * 78)
    print("  (gap = mean reference surprisal - mean variant surprisal;")
    print("   target range roughly -20 to -30, same order of magnitude as lstm's -13.8)")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
