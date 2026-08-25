#!/usr/bin/env python3
"""
Script 16: Emotion Annotation (Valence-Arousal Scoring - Recalibrated)

MTP-II: How Do Emotions Influence Decision Making in Language?
Russell's (1980) Circumplex Model of Affect

Recalibrated to handle lexicons scaled between [0, 1] instead of [-1, +1].
"""

import sys
import os
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src')))

from utils.paths import find_base
from utils.emotion import load_vad_lexicon, score_sentence, NEUTRAL_VALENCE, NEUTRAL_AROUSAL

try:
    from parsers.ud_parser import Sentence, Word
except ImportError:
    class Word:
        def __init__(self, form, upos):
            self.form = form
            self.upos = upos

# ============================================================================
# PATHS
# ============================================================================

BASE = find_base(__file__)
FILTERED_SENTENCES = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
VAD_LEXICON = os.path.join(BASE, "data", "processed", "Hindi-NRC-VAD-Lexicon.txt")
OUTPUT_FILE = os.path.join(BASE, "data", "processed", "preceding_emotions.pkl")

# ============================================================================
# MAIN MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" SCRIPT 16: EMOTION ANNOTATION (RECALIBRATED SCALE)")
    print("=" * 70 + "\n")

    lexicon = load_vad_lexicon(VAD_LEXICON)
    print(f"  Loaded {len(lexicon):,} unique Hindi words from `{os.path.basename(VAD_LEXICON)}`")

    if not os.path.exists(FILTERED_SENTENCES):
        print(f"ERROR: Sentences file not found at: {FILTERED_SENTENCES}")
        sys.exit(1)
        
    with open(FILTERED_SENTENCES, 'rb') as f:
        all_ref_sents = pickle.load(f)

    sent_lookup = {s.sent_id: s for s in all_ref_sents}
    ordered_sent_ids = [s.sent_id for s in all_ref_sents]
    sent_id_to_disc_idx = {sid: i for i, sid in enumerate(ordered_sent_ids)}

    emotions = {}
    n_with_context = 0
    n_no_context = 0

    for sent_id in ordered_sent_ids:
        disc_idx = sent_id_to_disc_idx[sent_id]
        prev_sent = sent_lookup.get(ordered_sent_ids[disc_idx - 1]) if disc_idx > 0 else None

        if prev_sent is not None:
            score = score_sentence(prev_sent.words, lexicon)
            score['has_context'] = True
            n_with_context += 1
        else:
            score = {
                'valence': NEUTRAL_VALENCE,
                'arousal': NEUTRAL_AROUSAL,
                'coverage': 0.0,
                'n_matched': 0,
                'has_context': False,
            }
            n_no_context += 1

        emotions[sent_id] = score

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(emotions, f)

    print("\n" + "=" * 70)
    print(" ANNOTATION RESULTS")
    print("=" * 70)
    print(f"  Total sentences scored      : {len(emotions):,}")
    print(f"  With preceding context      : {n_with_context:,}")
    print(f"  Without context (neutral)   : {n_no_context:,}")

    coverages = [e['coverage'] for e in emotions.values() if e['has_context']]
    if coverages:
        print(f"  Avg lexicon coverage        : {sum(coverages) / len(coverages):.1%}")

    valences = [e['valence'] for e in emotions.values() if e['has_context']]
    arousals = [e['arousal'] for e in emotions.values() if e['has_context']]
    if valences:
        print(f"\n  Valence range  : [{min(valences):.3f}, {max(valences):.3f}]")
        print(f"  Valence mean   : {sum(valences)/len(valences):.3f}")
        print(f"  Arousal range  : [{min(arousals):.3f}, {max(arousals):.3f}]")
        print(f"  Arousal mean   : {sum(arousals)/len(arousals):.3f}")

        q1 = sum(1 for v, a in zip(valences, arousals) if v >= 0 and a >= 0.5)  
        q2 = sum(1 for v, a in zip(valences, arousals) if v < 0 and a >= 0.5)   
        q3 = sum(1 for v, a in zip(valences, arousals) if v < 0 and a < 0.5)    
        q4 = sum(1 for v, a in zip(valences, arousals) if v >= 0 and a < 0.5)   
        total = len(valences)
        print(f"\n  Circumplex quadrant distribution:")
        print(f"    Q1 pleasant-activated (excited/happy)  : {q1:>5,} ({100*q1/total:.1f}%)")
        print(f"    Q2 unpleasant-activated (angry/tense)  : {q2:>5,} ({100*q2/total:.1f}%)")
        print(f"    Q3 unpleasant-calm (sad/depressed)     : {q3:>5,} ({100*q3/total:.1f}%)")
        print(f"    Q4 pleasant-calm (content/relaxed)     : {q4:>5,} ({100*q4/total:.1f}%)")

    print("=" * 70)
    print(f"\n  Saved -> {OUTPUT_FILE}\n")

if __name__ == "__main__":
    main()