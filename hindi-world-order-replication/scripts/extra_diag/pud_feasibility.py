#!/usr/bin/env python3
"""
pud_feasibility_check.py

Before running the full MTP-II pipeline on Hindi-PUD, this script checks
how many sentences survive the 6 MTP-I filtering conditions.

Run from project root (digit env):
    python scripts/pud_feasibility_check.py

If too few sentences survive (<30), the full pipeline won't produce
meaningful results and you should discuss with your professor.
"""

import os
import sys
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "src")))
from parsers.ud_parser import UDParser, Sentence

# ============================================================================
# PATH TO PUD FILE
# ============================================================================
PUD_FILE = os.path.expanduser(
    "~/Downloads/UD_Hindi-PUD-master/hi_pud-ud-test.conllu")

NEGATION = {"नहीं", "न", "मत"}
SUBJECT_RELS = {"nsubj", "csubj"}
OBJECT_RELS  = {"obj", "iobj"}

# ============================================================================
# FILTER CONDITIONS (same as Script 03)
# ============================================================================

def is_projective(sent):
    """No crossing dependency arcs."""
    arcs = [(min(w.idx, w.head), max(w.idx, w.head))
            for w in sent.words if w.head > 0 and not w.is_root()]
    for i, (s1, e1) in enumerate(arcs):
        for s2, e2 in arcs[i+1:]:
            if s1 < s2 < e1 < e2 or s2 < s1 < e2 < e1:
                return False
    return True

def is_declarative(sent):
    """Stype=declarative in any MISC field."""
    return any("Stype=declarative" in (w.misc or "") for w in sent.words)

def is_finite_verb_root(sent):
    """Root is VERB or AUX."""
    rw = sent.root_word
    return rw is not None and rw.upos in ("VERB", "AUX")

def has_no_negation(sent):
    """No negation markers."""
    return not any(w.form in NEGATION for w in sent.words)

def has_subject_and_object(sent):
    """Active subject AND object as DIRECT dependents of root."""
    if sent.root_idx is None:
        return False
    children = sent.get_children(sent.root_idx)
    has_subj = any(w.deprel in SUBJECT_RELS for w in children)
    has_obj  = any(w.deprel in OBJECT_RELS for w in children)
    return has_subj and has_obj

def has_two_preverbal_dependents(sent):
    """At least 2 preverbal non-punct dependents of root."""
    if sent.root_idx is None:
        return False
    preverbal = [w for w in sent.get_children(sent.root_idx)
                 if w.idx < sent.root_idx and w.upos != "PUNCT"]
    return len(preverbal) >= 2

def apply_filters(sentences):
    results = {
        "total":          len(sentences),
        "has_subj_obj":   0,
        "projective":     0,
        "declarative":    0,
        "finite_root":    0,
        "no_negation":    0,
        "two_preverbal":  0,
        "all_pass":       0,
    }
    passed = []
    for s in sentences:
        if not has_subject_and_object(s):   continue
        results["has_subj_obj"] += 1
        if not is_projective(s):            continue
        results["projective"] += 1
        if not is_declarative(s):           continue
        results["declarative"] += 1
        if not is_finite_verb_root(s):      continue
        results["finite_root"] += 1
        if not has_no_negation(s):          continue
        results["no_negation"] += 1
        if not has_two_preverbal_dependents(s): continue
        results["two_preverbal"] += 1
        results["all_pass"] += 1
        passed.append(s)
    return results, passed

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" PUD FEASIBILITY CHECK")
    print("="*70 + "\n")

    if not os.path.exists(PUD_FILE):
        print(f"ERROR: PUD file not found at:\n  {PUD_FILE}")
        print("Update PUD_FILE path at the top of this script.")
        sys.exit(1)

    print(f"Parsing: {os.path.basename(PUD_FILE)}")
    parser = UDParser()
    sentences = parser.parse_file(PUD_FILE, verbose=False)
    print(f"  Total sentences parsed: {len(sentences):,}\n")

    print("Applying 6 filter conditions...")
    results, passed = apply_filters(sentences)

    print(f"\n{'Condition':<35} {'Remaining':>10}")
    print("-" * 47)
    print(f"{'(a) Subject + Object as root dependents':<35} {results['has_subj_obj']:>10,}")
    print(f"{'(b) Projective tree':<35} {results['projective']:>10,}")
    print(f"{'(c) Declarative sentence':<35} {results['declarative']:>10,}")
    print(f"{'(d) Finite verb root':<35} {results['finite_root']:>10,}")
    print(f"{'(e) No negation markers':<35} {results['no_negation']:>10,}")
    print(f"{'(f) >= 2 preverbal dependents':<35} {results['two_preverbal']:>10,}")
    print("-" * 47)
    print(f"{'FINAL (all conditions passed)':<35} {results['all_pass']:>10,}")

    n = results["all_pass"]
    total = results["total"]
    print(f"\n  Survival rate: {n}/{total} = {100*n/total:.1f}%")
    print(f"  (HUTB reference: 2,779/13,306 = 20.9%)")

    print("\n" + "="*70)
    if n >= 50:
        print(f"  VERDICT: {n} sentences — sufficient for pipeline validation.")
        print("  Proceed with pud_run_pipeline.py")
    elif n >= 20:
        print(f"  VERDICT: {n} sentences — marginal. Results will be suggestive.")
        print("  Recommend proceeding but noting the small sample in your report.")
    else:
        print(f"  VERDICT: Only {n} sentences — too few for reliable analysis.")
        print("  Recommend using BHAAV instead, or reporting the filter survival")
        print("  rate as a finding (PUD sentences are more complex/non-canonical).")
    print("="*70 + "\n")

    # Show a few filtered sentences
    if passed:
        print("Sample filtered sentences:\n")
        for s in passed[:3]:
            print(f"  [{s.sent_id}] {s.text[:80]}...")
            preverbal = [w for w in s.get_children(s.root_idx)
                         if w.idx < s.root_idx and w.upos != "PUNCT"]
            print(f"  Root: {s.root_word.form}  Preverbal deps: {len(preverbal)}")
            print()

if __name__ == "__main__":
    main()