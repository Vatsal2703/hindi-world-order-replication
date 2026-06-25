#!/usr/bin/env python3
"""
Script 19: Emotion Annotation Validation (Fixed for Hindi-NRC-VAD-Lexicon)

MTP-II: Validates the (valence, arousal) emotion annotations two ways:

  1. Circumplex check: verify that known emotion seed words fall in the correct
     Russell (1980) quadrants when scored by our Hindi VAD lexicon.

  2. Human validation prep: export a random sample of preceding sentences with
     their automatic (valence, arousal) scores so 2-3 native Hindi speakers can
     rate them, enabling an inter-annotator agreement / correlation analysis.

Input:
  ./data/processed/Hindi-NRC-VAD-Lexicon.txt    (same as Script 16b)
  ./data/processed/preceding_emotions.pkl
  ./data/processed/replication_filtered_sentences.pkl

Output:
  ./data/results/circumplex_validation.csv
  ./data/results/human_validation_sample.csv
"""

import sys
import os
import csv
import pickle
import random
from collections import defaultdict

# ============================================================================
# DYNAMIC PATH RESOLUTION (same as Script 16b)
# ============================================================================

def find_workspace_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = script_dir

    while repo_root and repo_root != os.path.dirname(repo_root):
        if os.path.basename(repo_root) == "hindi-world-order-replication":
            return repo_root
        if os.path.isdir(os.path.join(repo_root, "hindi-world-order-replication")):
            return os.path.join(repo_root, "hindi-world-order-replication")
        repo_root = os.path.dirname(repo_root)
    return "."

BASE = find_workspace_paths()
VAD_LEXICON = os.path.join(BASE, "data", "processed", "Hindi-NRC-VAD-Lexicon.txt")
EMOTIONS_IN = os.path.join(BASE, "data", "processed", "preceding_emotions.pkl")
FILTERED_SENTENCES = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
CIRCUMPLEX_OUT = os.path.join(BASE, "data", "results", "circumplex_validation.csv")
HUMAN_SAMPLE_OUT = os.path.join(BASE, "data", "results", "human_validation_sample.csv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src')))

try:
    from parsers.ud_parser import Sentence, Word
except ImportError:
    pass

SAMPLE_SIZE = 200
RANDOM_SEED = 42

# Russell seed emotions -> expected quadrant
# Q1=pleasant-activated, Q2=unpleasant-activated, Q3=unpleasant-calm, Q4=pleasant-calm
SEED_WORDS = {
    "खुश": "Q1",        # happy
    "उत्साह": "Q1",     # excitement
    "गुस्सा": "Q2",     # anger
    "डर": "Q2",         # fear
    "दुखी": "Q3",       # sad
    "निराश": "Q3",      # depressed
    "शांत": "Q4",       # calm
    "संतुष्ट": "Q4",    # content
}


def load_vad_lexicon(path):
    """
    Load Hindi-NRC-VAD-Lexicon.txt -- SAME format/scale logic as Script 16b.
    File is [0, 1] scaled, Hindi word in column 5.
    Returns: dict[hindi_word] -> (valence in [-1,+1], arousal in [0,1])
    """
    if not os.path.exists(path):
        print(f"ERROR: VAD lexicon not found at: {path}")
        sys.exit(1)

    raw_aggregates = defaultdict(list)

    with open(path, 'r', encoding='utf-8') as f:
        sample = f.readline()
        delimiter = '\t' if '\t' in sample else ' '
        f.seek(0)

        for line in f:
            parts = [p.strip() for p in line.strip().split(delimiter) if p.strip()]
            if not parts or len(parts) < 5:
                parts = [p.strip() for p in line.strip().split() if p.strip()]
                if len(parts) < 5:
                    continue

            # Skip header
            if parts[0].lower() == 'english' or parts[4] == 'Hindi':
                continue

            try:
                v_raw = float(parts[1])
                a_raw = float(parts[2])
                hindi_raw = parts[4].split()[0]

                # Recalibrate: [0,1] valence -> [-1,+1]; arousal stays in [0,1]
                valence_scaled = (v_raw - 0.5) * 2.0
                arousal_scaled = a_raw
            except (ValueError, IndexError):
                continue

            raw_aggregates[hindi_raw].append((valence_scaled, arousal_scaled))

    lexicon = {}
    for hi_word, scores in raw_aggregates.items():
        avg_v = sum(s[0] for s in scores) / len(scores)
        avg_a = sum(s[1] for s in scores) / len(scores)
        lexicon[hi_word] = (avg_v, avg_a)

    print(f"  Loaded {len(lexicon):,} unique Hindi words")
    return lexicon


def quadrant(valence, arousal):
    """Russell quadrant given valence [-1,+1] and arousal [0,1]."""
    if valence >= 0 and arousal >= 0.5:
        return "Q1"
    if valence < 0 and arousal >= 0.5:
        return "Q2"
    if valence < 0 and arousal < 0.5:
        return "Q3"
    return "Q4"


def circumplex_check(lexicon):
    print("\n" + "-" * 70)
    print(" CIRCUMPLEX VALIDATION (seed words)")
    print("-" * 70)
    rows = []
    correct = 0
    total = 0
    for word, expected_q in SEED_WORDS.items():
        if word not in lexicon:
            rows.append({"word": word, "expected": expected_q,
                         "valence": None, "arousal": None,
                         "actual": "NOT_IN_LEXICON", "match": False})
            print(f"  {word:>12s}: not in lexicon")
            continue
        valence, arousal = lexicon[word]  # already in correct scales
        actual_q = quadrant(valence, arousal)
        match = (actual_q == expected_q)
        correct += int(match)
        total += 1
        rows.append({"word": word, "expected": expected_q,
                     "valence": round(valence, 3), "arousal": round(arousal, 3),
                     "actual": actual_q, "match": match})
        flag = "✓" if match else "✗"
        print(f"  {word:>12s}: V={valence:+.2f} A={arousal:.2f} "
              f"-> {actual_q} (expected {expected_q}) {flag}")

    if total > 0:
        print(f"\n  Circumplex accuracy: {correct}/{total} = {correct/total:.1%}")

    os.makedirs(os.path.dirname(CIRCUMPLEX_OUT), exist_ok=True)
    with open(CIRCUMPLEX_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["word", "expected", "valence",
                                               "arousal", "actual", "match"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {CIRCUMPLEX_OUT}")


def build_human_sample(emotions, sentences):
    print("\n" + "-" * 70)
    print(" HUMAN VALIDATION SAMPLE")
    print("-" * 70)

    sent_lookup = {s.sent_id: s for s in sentences}
    ordered = [s.sent_id for s in sentences]
    idx = {sid: i for i, sid in enumerate(ordered)}

    candidates = [sid for sid in ordered
                  if emotions.get(sid, {}).get("has_context", False)]

    random.seed(RANDOM_SEED)
    sample = random.sample(candidates, min(SAMPLE_SIZE, len(candidates)))

    rows = []
    for sid in sample:
        disc_idx = idx[sid]
        prev_sid = ordered[disc_idx - 1] if disc_idx > 0 else ""
        prev_text = sent_lookup[prev_sid].text if prev_sid in sent_lookup else ""
        e = emotions[sid]
        rows.append({
            "sent_id": sid,
            "preceding_sentence": prev_text,
            "auto_valence": round(e["valence"], 3),
            "auto_arousal": round(e["arousal"], 3),
            "human_valence": "",
            "human_arousal": "",
        })

    os.makedirs(os.path.dirname(HUMAN_SAMPLE_OUT), exist_ok=True)
    with open(HUMAN_SAMPLE_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sent_id", "preceding_sentence", "auto_valence", "auto_arousal",
            "human_valence", "human_arousal"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Exported {len(rows)} sentences for manual annotation")
    print(f"  Annotators fill 'human_valence' (-1..+1) and 'human_arousal' (0..1)")
    print(f"  Saved -> {HUMAN_SAMPLE_OUT}")


def main():
    print("\n" + "=" * 70)
    print(" SCRIPT 19: EMOTION ANNOTATION VALIDATION")
    print("=" * 70)

    lexicon = load_vad_lexicon(VAD_LEXICON)
    circumplex_check(lexicon)

    if os.path.exists(EMOTIONS_IN) and os.path.exists(FILTERED_SENTENCES):
        with open(EMOTIONS_IN, 'rb') as f:
            emotions = pickle.load(f)
        with open(FILTERED_SENTENCES, 'rb') as f:
            sentences = pickle.load(f)
        build_human_sample(emotions, sentences)
    else:
        print("\n  (Skipping human sample: run Scripts 16b first)")

    print("\n" + "=" * 70)
    print("  Validation complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()