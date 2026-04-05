#!/usr/bin/env python3
"""
Step 4: Subtree-aware variant generation.

Paper: "Discourse Context Predictability Effects in Hindi Word Order"
       Ranjan, van Schijndel, Agarwal, Rajkumar (EMNLP 2022)

For each reference sentence:
  1. Identify preverbal direct dependents of root (excluding punct)
  2. Extract complete subtrees for each dependent (DFS)
  3. Permute the subtree blocks (not individual words)
  4. Keep root verb + postverbal elements in original position
  5. Deduplicate by surface string (skip variants identical to reference)
  6. If > 99 unique variants, randomly sample exactly 99
     (Paper Appendix A footnote 8: "if total > 100, chose 99 
      non-reference variants randomly along with the reference")

Output structure per entry:
  - sentence_id:       index of reference sentence
  - sent_id:           CoNLL-U sent_id string
  - reference:         reference surface string
  - variant:           variant surface string  
  - variant_order:     list of token indices in variant order
  - is_reference:      True if this entry IS the reference
  - construction_type: SOV / DOSV / IOSV (for Table 3 analyses)
  - tokens:            original token list (for DL/IS computation)

Paper target: ~72,833 variant pairs from ~1,996 references
"""

import sys
import os
import pickle
import itertools
import random
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from parsers.ud_parser import Sentence, Word

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "./data/processed/replication_filtered_sentences.pkl"
OUTPUT_FILE = "./data/processed/all_variants_final.pkl"

# Paper: "if > 100 variants, randomly sample exactly 99"
MAX_NON_REF_VARIANTS = 99

# Relations for construction type labeling
SUBJ_TAGS = {'nsubj', 'csubj'}
DIRECT_OBJ_TAGS = {'obj'}
INDIRECT_OBJ_TAGS = {'iobj'}


# ============================================================================
# SUBTREE EXTRACTION
# ============================================================================

def get_subtree_indices(word_idx, sentence):
    """
    Iterative DFS to find all indices in a word's subtree.
    Returns sorted list of indices (preserves phrase-internal order).
    """
    visited = set()
    stack = [word_idx]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        # Push all children
        for w in sentence.words:
            if w.head == current:
                stack.append(w.idx)

    return sorted(visited)


# ============================================================================
# CONSTRUCTION TYPE
# ============================================================================

def get_construction_type(sentence):
    """
    Label reference sentence by word-order type (Table 3 of paper):
      DOSV — direct object precedes subject
      IOSV — indirect object precedes subject
      SOV  — canonical; subject before object
    """
    root = sentence.root_word
    if root is None:
        return "unknown"

    root_idx = root.idx

    # Preverbal direct deps, sorted left to right
    preverbal = sorted(
        [w for w in sentence.words
         if w.head == root_idx
         and w.idx < root_idx
         and w.deprel != 'punct'],
        key=lambda w: w.idx
    )

    if not preverbal:
        return "unknown"

    subj_pos = next(
        (w.idx for w in preverbal if w.deprel in SUBJ_TAGS), None
    )
    dobj_pos = next(
        (w.idx for w in preverbal if w.deprel in DIRECT_OBJ_TAGS), None
    )
    iobj_pos = next(
        (w.idx for w in preverbal if w.deprel in INDIRECT_OBJ_TAGS), None
    )

    # DOSV: direct object appears before subject
    if dobj_pos is not None:
        if subj_pos is None or dobj_pos < subj_pos:
            return "DOSV"

    # IOSV: indirect object appears before subject
    if iobj_pos is not None:
        if subj_pos is None or iobj_pos < subj_pos:
            return "IOSV"

    return "SOV"


# ============================================================================
# VARIANT GENERATION
# ============================================================================

def generate_variants(sentence, sentence_idx):
    """
    Generate all valid permutations of preverbal subtree blocks.

    Returns list of dicts, each containing reference-variant pair info.
    The reference itself is included with is_reference=True.
    """
    root = sentence.root_word
    if root is None:
        return []

    root_idx = root.idx
    reference_forms = [w.form for w in sentence.words]
    reference_str = " ".join(reference_forms)

    # ── Identify preverbal deps (exclude punct) ──
    preverbal_deps = sorted(
        [w for w in sentence.words
         if w.head == root_idx
         and w.idx < root_idx
         and w.deprel != 'punct'],
        key=lambda w: w.idx
    )

    if len(preverbal_deps) < 2:
        return []

    # ── Build subtree blocks ──
    blocks = []
    for dep in preverbal_deps:
        block_indices = get_subtree_indices(dep.idx, sentence)
        block_tokens = sorted(
            [w for w in sentence.words if w.idx in block_indices],
            key=lambda w: w.idx
        )
        blocks.append(block_tokens)

    # ── Identify fixed tokens (root + postverbal + anything not in blocks) ──
    all_block_indices = set()
    for block in blocks:
        for w in block:
            all_block_indices.add(w.idx)

    fixed_tokens = sorted(
        [w for w in sentence.words if w.idx not in all_block_indices],
        key=lambda w: w.idx
    )

    # ── Construction type of the reference ──
    construction = get_construction_type(sentence)

    # ── Generate all permutations ──
    results = []
    seen_strings = set()

    for perm in itertools.permutations(blocks):
        # Build the reordered token list
        new_tokens = []
        for block in perm:
            new_tokens.extend(block)
        new_tokens.extend(fixed_tokens)

        variant_str = " ".join(w.form for w in new_tokens)
        variant_order = [w.idx for w in new_tokens]

        # Deduplicate by surface string
        if variant_str in seen_strings:
            continue
        seen_strings.add(variant_str)

        is_ref = (variant_str == reference_str)

        results.append({
            'sentence_id': sentence_idx,
            'sent_id': sentence.sent_id,
            'reference': reference_str,
            'variant': variant_str,
            'variant_order': variant_order,
            'is_reference': is_ref,
            'construction_type': construction,
            'tokens': sentence.words,
        })

    # ── Separate reference from non-reference variants ──
    ref_entries = [r for r in results if r['is_reference']]
    nonref_entries = [r for r in results if not r['is_reference']]

    # ── Sample if too many non-reference variants ──
    if len(nonref_entries) > MAX_NON_REF_VARIANTS:
        nonref_entries = random.sample(nonref_entries, MAX_NON_REF_VARIANTS)

    # Return reference + sampled variants
    return ref_entries + nonref_entries


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" STEP 4: SUBTREE-AWARE VARIANT GENERATION")
    print("=" * 70 + "\n")

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run step 03 first!")
        return 1

    with open(INPUT_FILE, 'rb') as f:
        sentences = pickle.load(f)

    print(f"Loaded {len(sentences):,} reference sentences")
    print(f"Max non-reference variants per sentence: {MAX_NON_REF_VARIANTS}\n")

    all_results = []
    refs_with_variants = 0
    total_variants_only = 0

    for idx, sent in enumerate(tqdm(sentences, desc="Generating variants")):
        entries = generate_variants(sent, idx)
        if entries:
            refs_with_variants += 1
            nonref = [e for e in entries if not e['is_reference']]
            total_variants_only += len(nonref)
            all_results.extend(entries)

    # ── Save ──
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_results, f)

    # ── Summary ──
    print("\n" + "=" * 70)
    print(" VARIANT GENERATION RESULTS")
    print("=" * 70)
    print(f"  Reference sentences with variants : {refs_with_variants:,}")
    print(f"  Total entries (ref + variants)    : {len(all_results):,}")
    print(f"  Non-reference variants only       : {total_variants_only:,}")
    print(f"  Avg variants per reference        : {total_variants_only / max(refs_with_variants, 1):.1f}")
    print()
    print(f"  Paper target: ~72,833 non-reference variants")
    print(f"  Your result:  {total_variants_only:,} non-reference variants")
    diff_pct = abs(total_variants_only - 72833) / 72833 * 100
    if diff_pct < 15:
        print(f"  ✓  Within {diff_pct:.0f}% of paper target — good match")
    else:
        print(f"  ⚠  {diff_pct:.0f}% {'below' if total_variants_only < 72833 else 'above'} target")

    # ── Construction type distribution ──
    print("\n  Construction type distribution (reference sentences):")
    ref_entries = [r for r in all_results if r['is_reference']]
    type_counts = Counter(r['construction_type'] for r in ref_entries)
    for ctype, count in sorted(type_counts.items()):
        pct = count / len(ref_entries) * 100
        print(f"    {ctype:<8} : {count:>5,}  ({pct:.1f}%)")

    print("=" * 70)
    print(f"\n  Saved → {OUTPUT_FILE}\n")

    return 0


if __name__ == "__main__":
    exit(main())