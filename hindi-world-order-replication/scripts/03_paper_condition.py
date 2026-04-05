#!/usr/bin/env python3
"""
Step 3: Filter sentences based on the replication conditions from the paper.

Paper: "Discourse Context Predictability Effects in Hindi Word Order"
       Ranjan, van Schijndel, Agarwal, Rajkumar (EMNLP 2022)

Conditions applied (Section 3 of the paper):
  (a) Both subject AND object as DIRECT dependents of root verb
  (b) Projective dependency tree (no crossing arcs)
  (c) Declarative sentence (Stype=declarative in MISC — trusting dataset)
  (d) Root is a finite verb (VERB or AUX)
  (e) No negative markers (नहीं, न, मत)
  (f) At least 2 preverbal direct dependents of root (excluding punct)
      NO upper cap — as per professor's instruction.

Why no passive subjects (nsubj:pass, csubj:pass)?
  - Paper studies ACTIVE transitive sentences where S and O can
    freely reorder. Passives have different word order constraints.

Input:  ./data/processed/valid_sentences.pkl
        (from step 01 — has subject+object ANYWHERE in tree,
         includes passives, train split only)

Output: ./data/processed/replication_filtered_sentences.pkl
"""

import sys
import os
import pickle
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parsers.ud_parser import Sentence, Word

# ============================================================================
# CONSTANTS
# ============================================================================

# Active subjects only — paper studies active transitive sentences.
SUBJ_TAGS = {'nsubj', 'csubj'}

# Direct and indirect objects (same as paper)
OBJ_TAGS = {'obj', 'iobj'}

# Hindi negative markers to exclude (paper condition)
NEGATIVE_MARKERS = {'नहीं', 'न', 'मत'}

# Minimum preverbal dependents — need at least 2 to permute
MIN_PREVERBAL = 2

# ============================================================================
# CONDITION CHECKERS
# ============================================================================

def is_finite_verb_root(sentence: Sentence) -> bool:
    """
    Condition (d): Root must be VERB or AUX.
    AUX covers compound verb constructions in Hindi.
    """
    if sentence.root_word is None:
        return False
    return sentence.root_word.upos in {'VERB', 'AUX'}


def has_negative_marker(sentence: Sentence) -> bool:
    """
    Condition (e): Reject sentences with नहीं, न, मत.
    NOTE: Word class uses .form for surface form.
    """
    return any(w.form in NEGATIVE_MARKERS for w in sentence.words)


def is_declarative(sentence: Sentence) -> bool:
    """
    Condition (c): Sentence must be declarative.
    Uses Stype=declarative from MISC field — trusting dataset annotations.
    """
    if sentence.root_word is None:
        return False
    misc = sentence.root_word.misc or ""
    return 'Stype=declarative' in misc


def has_direct_subject_and_object(sentence: Sentence) -> bool:
    """
    Condition (a): Both subject AND object must be DIRECT
    dependents of the root verb (head == root_idx).
    """
    if sentence.root_word is None:
        return False

    root_idx = sentence.root_word.idx
    root_deps = [w for w in sentence.words if w.head == root_idx]

    has_subj = any(w.deprel in SUBJ_TAGS for w in root_deps)
    has_obj = any(w.deprel in OBJ_TAGS for w in root_deps)
    return has_subj and has_obj


def get_preverbal_deps(sentence: Sentence) -> list:
    """
    Get preverbal direct dependents of root, excluding punct.
    Preverbal = word.idx < root_word.idx
    Direct dependent = word.head == root_word.idx
    """
    if sentence.root_word is None:
        return []

    root_idx = sentence.root_word.idx
    return [
        w for w in sentence.words
        if w.head == root_idx
        and w.idx < root_idx
        and w.deprel != 'punct'
    ]


def has_enough_preverbal(sentence: Sentence) -> bool:
    """
    Condition (f): At least MIN_PREVERBAL preverbal direct dependents.
    No upper cap — as per professor's instruction.
    """
    return len(get_preverbal_deps(sentence)) >= MIN_PREVERBAL


def is_projective(sentence: Sentence) -> bool:
    """
    Condition (b): No crossing dependency arcs.
    O(n²) so checked last.
    """
    arcs = [(w.idx, w.head) for w in sentence.words if w.head != 0]

    for i in range(len(arcs)):
        a, b = min(arcs[i]), max(arcs[i])
        for j in range(i + 1, len(arcs)):
            c, d = min(arcs[j]), max(arcs[j])
            if (a < c < b < d) or (c < a < d < b):
                return False
    return True


# ============================================================================
# MASTER FILTER
# ============================================================================

def passes_all_conditions(sentence: Sentence) -> tuple:
    """
    Apply all 6 conditions. Returns (passed: bool, reason: str).
    Ordered cheapest-first, projective last.
    """
    if not is_finite_verb_root(sentence):
        return False, 'non_finite_root'

    if has_negative_marker(sentence):
        return False, 'negative_marker'

    if not is_declarative(sentence):
        return False, 'non_declarative'

    if not has_direct_subject_and_object(sentence):
        return False, 'no_direct_subj_obj'

    if not has_enough_preverbal(sentence):
        return False, 'preverbal_too_few'

    if not is_projective(sentence):
        return False, 'non_projective'

    return True, 'passed'


# ============================================================================
# MAIN
# ============================================================================

INPUT_FILE = "./data/processed/valid_sentences.pkl"
OUTPUT_FILE = "./data/processed/replication_filtered_sentences.pkl"


def main():
    print("\n" + "=" * 70)
    print(" STEP 3: REPLICATION FILTERING — PAPER CONDITIONS")
    print("=" * 70 + "\n")

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found!")
        print("Run 01_parse_hutb.py first.")
        sys.exit(1)

    with open(INPUT_FILE, 'rb') as f:
        sentences = pickle.load(f)

    print(f"Loaded {len(sentences):,} sentences from {INPUT_FILE}")
    print(f"(Step 01 pre-filter: subject + object anywhere in tree)\n")

    # ── Track exclusion reasons ──
    stats = {
        'total': len(sentences),
        'non_finite_root': 0,
        'negative_marker': 0,
        'non_declarative': 0,
        'no_direct_subj_obj': 0,
        'preverbal_too_few': 0,
        'non_projective': 0,
        'passed': 0,
    }

    filtered = []

    for sent in tqdm(sentences, desc="Filtering"):
        passed, reason = passes_all_conditions(sent)
        stats[reason] += 1
        if passed:
            filtered.append(sent)

    # ── Results ──
    print("\n" + "=" * 70)
    print(" FILTERING RESULTS")
    print("=" * 70)
    print(f"  Total input sentences       : {stats['total']:,}")
    print(f"  Passed all conditions       : {stats['passed']:,}")
    print(f"  Filter pass rate            : {100 * stats['passed'] / stats['total']:.1f}%")
    print()
    print("  Exclusion breakdown:")
    print(f"    (d) Non-finite root       : {stats['non_finite_root']:,}")
    print(f"    (e) Negative marker       : {stats['negative_marker']:,}")
    print(f"    (c) Non-declarative       : {stats['non_declarative']:,}")
    print(f"    (a) No direct subj+obj    : {stats['no_direct_subj_obj']:,}")
    print(f"    (f) Preverbal < {MIN_PREVERBAL}        : {stats['preverbal_too_few']:,}")
    print(f"    (b) Non-projective        : {stats['non_projective']:,}")
    print("=" * 70)

    # ── Preverbal distribution diagnostic ──
    print("\n  Preverbal dep count distribution (passed sentences):")
    preverbal_dist = {}
    for sent in filtered:
        count = len(get_preverbal_deps(sent))
        preverbal_dist[count] = preverbal_dist.get(count, 0) + 1

    for k in sorted(preverbal_dist.keys()):
        perms = 1
        for i in range(2, k + 1):
            perms *= i
        perms -= 1  # subtract the reference itself
        print(f"    {k} preverbal deps : {preverbal_dist[k]:>5,} sentences  (up to {perms} variants each)")

    total_max_variants = sum(
        count * (factorial(k) - 1)
        for k, count in preverbal_dist.items()
    )
    print(f"    ──────────────────────────")
    print(f"    Max possible variants     : {total_max_variants:,}")
    print(f"    Paper target              : ~72,833")

    # ── Paper comparison ──
    print(f"\n  Paper target  : ~1,996 reference sentences")
    print(f"  Your result   : {len(filtered):,} reference sentences")
    diff_pct = abs(len(filtered) - 1996) / 1996 * 100
    if diff_pct < 15:
        print(f"  ✓  Within {diff_pct:.0f}% of paper target — good match")
    elif len(filtered) < 1996:
        print(f"  ⚠  {diff_pct:.0f}% below target — filters may be too strict")
    else:
        print(f"  ⚠  {diff_pct:.0f}% above target — filters may be too loose")

    # ── Save ──
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(filtered, f)
    print(f"\n  Saved {len(filtered):,} sentences → {OUTPUT_FILE}")

    # ── Stype diagnostic ──
    print("\n" + "-" * 70)
    print(" DIAGNOSTIC: Stype annotation coverage in input")
    print("-" * 70)

    has_stype = sum(1 for s in sentences
                    if s.root_word and s.root_word.misc and 'Stype=' in s.root_word.misc)
    has_decl = sum(1 for s in sentences
                   if s.root_word and s.root_word.misc and 'Stype=declarative' in s.root_word.misc)
    has_q = sum(1 for s in sentences if any(w.form == '?' for w in s.words))

    n = len(sentences)
    print(f"  Any Stype tag        : {has_stype:,} / {n:,} ({100*has_stype/n:.1f}%)")
    print(f"  Stype=declarative    : {has_decl:,} / {n:,} ({100*has_decl/n:.1f}%)")
    print(f"  Has '?' token        : {has_q:,} / {n:,} ({100*has_q/n:.1f}%)")
    print(f"  No '?' token         : {n - has_q:,}")
    print("-" * 70 + "\n")


def factorial(n):
    """Simple factorial for display purposes."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    main()