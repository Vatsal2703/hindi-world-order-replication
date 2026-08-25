#!/usr/bin/env python3
"""
Script 19: Emotion Annotation Validation  — v3.0

MTP-II: Validates the (valence, arousal) annotations with fairer, more
rigorous metrics than the earlier hard-quadrant check.

v3.0 improvements
  1. Expanded seed list (~35 words with clear circumplex membership).
  2. Separate VALENCE-sign and AROUSAL-sign accuracy (not just quadrant),
     since valence is the more reliable dimension and a single number hides that.
  3. Neutral band around the arousal boundary: words whose arousal falls inside
     the band are reported as "borderline" and excluded from hard scoring,
     so near-midline words are not counted as outright errors.
  4. Data-driven arousal threshold = MEDIAN arousal of the lexicon, instead of
     an arbitrary 0.5 (the lexicon's arousal clusters near 0.5).
  5. Human-validation correlation: if the exported sample has been filled in by
     annotators, compute auto-vs-human Pearson r for valence and arousal.

RUN IN `digit` CONDA ENV.

Input : data/processed/Hindi-NRC-VAD-Lexicon.txt
        data/processed/preceding_emotions.pkl
        data/processed/replication_filtered_sentences.pkl
        data/results/human_validation_sample.csv   (optional, if filled in)
Output: data/results/circumplex_validation.csv
        data/results/human_validation_sample.csv   (created if absent)
        data/results/human_correlation.csv          (if annotations present)
"""

VERSION = "3.0"

import sys
import os
import csv
import pickle
import random
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src')))
from utils.paths import find_base
from utils.emotion import load_vad_lexicon as _load_vad_lexicon

# ============================================================================
# PATHS
# ============================================================================

BASE               = find_base(__file__)
VAD_LEXICON        = os.path.join(BASE, "data", "processed", "Hindi-NRC-VAD-Lexicon.txt")
EMOTIONS_IN        = os.path.join(BASE, "data", "processed", "preceding_emotions.pkl")
FILTERED_SENTENCES = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
CIRCUMPLEX_OUT     = os.path.join(BASE, "data", "results", "circumplex_validation.csv")
HUMAN_SAMPLE_OUT   = os.path.join(BASE, "data", "results", "human_validation_sample.csv")
HUMAN_CORR_OUT     = os.path.join(BASE, "data", "results", "human_correlation.csv")

from parsers.ud_parser import Sentence, Word  # noqa: F401  (needed to unpickle)

SAMPLE_SIZE   = 200
RANDOM_SEED   = 42
NEUTRAL_BAND  = 0.05   # +/- around the arousal threshold = "borderline"

# ============================================================================
# EXPANDED SEED LIST
# Each: word -> (expected valence sign '+'/'-', expected arousal 'H'/'L')
# H = activated/high arousal, L = calm/low arousal
# ============================================================================
SEED_WORDS = {
    # pleasant-activated (Q1: +, H)
    "खुश": ("+", "H"), "उत्साह": ("+", "H"), "आनंद": ("+", "H"),
    "उत्साहित": ("+", "H"), "प्रसन्न": ("+", "H"), "रोमांच": ("+", "H"),
    "जोश": ("+", "H"), "हर्ष": ("+", "H"),
    # unpleasant-activated (Q2: -, H)
    "गुस्सा": ("-", "H"), "डर": ("-", "H"), "क्रोध": ("-", "H"),
    "भय": ("-", "H"), "घबराहट": ("-", "H"), "आतंक": ("-", "H"),
    "चिंता": ("-", "H"), "तनाव": ("-", "H"),
    # unpleasant-calm (Q3: -, L)
    "दुखी": ("-", "L"), "निराश": ("-", "L"), "उदास": ("-", "L"),
    "थका": ("-", "L"), "बोरियत": ("-", "L"), "सुस्त": ("-", "L"),
    "अकेला": ("-", "L"), "हताश": ("-", "L"),
    # pleasant-calm (Q4: +, L)
    "शांत": ("+", "L"), "संतुष्ट": ("+", "L"), "आराम": ("+", "L"),
    "सुकून": ("+", "L"), "निश्चिंत": ("+", "L"), "शांति": ("+", "L"),
    "सहज": ("+", "L"), "तसल्ली": ("+", "L"),
}

# ============================================================================
# LEXICON (shared with Script 16)
# ============================================================================

def load_vad_lexicon(path):
    lexicon = _load_vad_lexicon(path)
    print(f"  Loaded {len(lexicon):,} unique Hindi words")
    return lexicon

# ============================================================================
# VALIDATION 1: CIRCUMPLEX (fairer metrics)
# ============================================================================

def circumplex_check(lexicon):
    print("\n" + "-" * 70)
    print(" CIRCUMPLEX VALIDATION (expanded seeds, fair metrics)")
    print("-" * 70)

    # Data-driven arousal threshold = median arousal across the lexicon
    arousal_median = statistics.median(a for _, a in lexicon.values())
    print(f"  Arousal threshold (lexicon median): {arousal_median:.3f}")
    print(f"  Neutral band: +/- {NEUTRAL_BAND}\n")

    rows = []
    val_correct = val_total = 0
    aro_correct = aro_total = aro_borderline = 0
    quad_correct = quad_total = 0

    for word, (exp_v, exp_a) in SEED_WORDS.items():
        if word not in lexicon:
            rows.append({"word": word, "valence": None, "arousal": None,
                         "exp_valence": exp_v, "exp_arousal": exp_a,
                         "valence_ok": "", "arousal_ok": "", "note": "not_in_lexicon"})
            print(f"  {word:>10s}: not in lexicon")
            continue

        v, a = lexicon[word]

        # --- valence sign ---
        got_v = "+" if v >= 0 else "-"
        v_ok = (got_v == exp_v)
        val_correct += v_ok
        val_total += 1

        # --- arousal with neutral band ---
        if abs(a - arousal_median) <= NEUTRAL_BAND:
            a_ok_str = "borderline"
            aro_borderline += 1
            quad_scored = False
        else:
            got_a = "H" if a > arousal_median else "L"
            a_ok = (got_a == exp_a)
            aro_correct += a_ok
            aro_total += 1
            a_ok_str = "✓" if a_ok else "✗"
            quad_scored = True

        # --- full quadrant (only when arousal not borderline) ---
        if quad_scored:
            quad_ok = v_ok and a_ok
            quad_correct += quad_ok
            quad_total += 1

        flag = "✓" if v_ok else "✗"
        print(f"  {word:>10s}: V={v:+.2f}({flag})  A={a:.2f}({a_ok_str})")
        rows.append({"word": word, "valence": round(v, 3), "arousal": round(a, 3),
                     "exp_valence": exp_v, "exp_arousal": exp_a,
                     "valence_ok": v_ok, "arousal_ok": a_ok_str, "note": ""})

    print("\n  " + "-" * 50)
    if val_total:
        print(f"  Valence-sign accuracy : {val_correct}/{val_total} = {val_correct/val_total:.1%}")
    if aro_total:
        print(f"  Arousal accuracy      : {aro_correct}/{aro_total} = {aro_correct/aro_total:.1%}"
              f"  ({aro_borderline} borderline excluded)")
    if quad_total:
        print(f"  Full-quadrant accuracy: {quad_correct}/{quad_total} = {quad_correct/quad_total:.1%}")
    print("  (Valence is the primary, reliable dimension; arousal is noisier near the median.)")

    os.makedirs(os.path.dirname(CIRCUMPLEX_OUT), exist_ok=True)
    with open(CIRCUMPLEX_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["word", "valence", "arousal",
                                          "exp_valence", "exp_arousal",
                                          "valence_ok", "arousal_ok", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved -> {CIRCUMPLEX_OUT}")

# ============================================================================
# VALIDATION 2: HUMAN SAMPLE (export if absent)
# ============================================================================

def build_human_sample(emotions, sentences):
    if os.path.exists(HUMAN_SAMPLE_OUT):
        print("\n  Human sample already exists — not overwriting.")
        return
    print("\n" + "-" * 70)
    print(" HUMAN VALIDATION SAMPLE")
    print("-" * 70)
    lookup  = {s.sent_id: s for s in sentences}
    ordered = [s.sent_id for s in sentences]
    idx     = {sid: i for i, sid in enumerate(ordered)}
    candidates = [sid for sid in ordered
                  if emotions.get(sid, {}).get("has_context", False)]
    random.seed(RANDOM_SEED)
    sample = random.sample(candidates, min(SAMPLE_SIZE, len(candidates)))

    rows = []
    for sid in sample:
        prev_sid  = ordered[idx[sid] - 1] if idx[sid] > 0 else ""
        prev_text = lookup[prev_sid].text if prev_sid in lookup else ""
        e = emotions[sid]
        rows.append({"sent_id": sid, "preceding_sentence": prev_text,
                     "auto_valence": round(e["valence"], 3),
                     "auto_arousal": round(e["arousal"], 3),
                     "human_valence": "", "human_arousal": ""})

    with open(HUMAN_SAMPLE_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["sent_id", "preceding_sentence",
                                          "auto_valence", "auto_arousal",
                                          "human_valence", "human_arousal"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Exported {len(rows)} sentences -> {HUMAN_SAMPLE_OUT}")

# ============================================================================
# VALIDATION 3: AUTO vs HUMAN CORRELATION (if annotations present)
# ============================================================================

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else None


def human_correlation():
    if not os.path.exists(HUMAN_SAMPLE_OUT):
        return
    va, vh, aa, ah = [], [], [], []
    with open(HUMAN_SAMPLE_OUT, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            hv, ha = r.get("human_valence", ""), r.get("human_arousal", "")
            if hv.strip() == "" or ha.strip() == "":
                continue
            try:
                vh.append(float(hv)); va.append(float(r["auto_valence"]))
                ah.append(float(ha)); aa.append(float(r["auto_arousal"]))
            except ValueError:
                continue

    if len(va) < 3:
        print("\n  Human annotations not yet filled in — skipping correlation.")
        print("  (Fill human_valence / human_arousal in the sample CSV, then re-run.)")
        return

    print("\n" + "-" * 70)
    print(f" AUTO vs HUMAN CORRELATION  (n = {len(va)} annotated)")
    print("-" * 70)
    rv = pearson(va, vh)
    ra = pearson(aa, ah)
    print(f"  Valence: Pearson r = {rv:+.3f}")
    print(f"  Arousal: Pearson r = {ra:+.3f}")

    with open(HUMAN_CORR_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["dimension", "pearson_r", "n"])
        w.writerow(["valence", round(rv, 3), len(va)])
        w.writerow(["arousal", round(ra, 3), len(aa)])
    print(f"  Saved -> {HUMAN_CORR_OUT}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(f" SCRIPT 19: EMOTION ANNOTATION VALIDATION  (v{VERSION})")
    print("=" * 70)

    lexicon = load_vad_lexicon(VAD_LEXICON)
    circumplex_check(lexicon)

    if os.path.exists(EMOTIONS_IN) and os.path.exists(FILTERED_SENTENCES):
        with open(EMOTIONS_IN, 'rb') as f:
            emotions = pickle.load(f)
        with open(FILTERED_SENTENCES, 'rb') as f:
            sentences = pickle.load(f)
        build_human_sample(emotions, sentences)
        human_correlation()
    else:
        print("\n  (Skipping human sample: run Script 16 first)")

    print("\n" + "=" * 70)
    print("  Validation complete.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()