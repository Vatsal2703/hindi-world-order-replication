#!/usr/bin/env python3
"""
Script 16b: Emotion Annotation (Valence-Arousal Scoring - Recalibrated)

MTP-II: How Do Emotions Influence Decision Making in Language?
Russell's (1980) Circumplex Model of Affect

Recalibrated to handle lexicons scaled between [0, 1] instead of [-1, +1].
"""

import sys
import os
import pickle
import csv
from collections import defaultdict

# ============================================================================
# DYNAMIC PATH RESOLUTION
# ============================================================================

def find_workspace_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = script_dir
    
    while repo_root and repo_root != os.path.dirname(repo_root):
        if os.path.basename(repo_root) == "hindi-world-order-replication":
            base = repo_root
            return os.path.join(base, "data", "processed", "replication_filtered_sentences.pkl"), \
                   os.path.join(base, "data", "processed", "Hindi-NRC-VAD-Lexicon.txt"), \
                   os.path.join(base, "data", "processed", "preceding_emotions.pkl")
        if os.path.isdir(os.path.join(repo_root, "hindi-world-order-replication")):
            base = os.path.join(repo_root, "hindi-world-order-replication")
            return os.path.join(base, "data", "processed", "replication_filtered_sentences.pkl"), \
                   os.path.join(base, "data", "processed", "Hindi-NRC-VAD-Lexicon.txt"), \
                   os.path.join(base, "data", "processed", "preceding_emotions.pkl")
        repo_root = os.path.dirname(repo_root)
        
    return "./data/processed/replication_filtered_sentences.pkl", \
           "./data/processed/Hindi-NRC-VAD-Lexicon.txt", \
           "./data/processed/preceding_emotions.pkl"

FILTERED_SENTENCES, VAD_LEXICON, OUTPUT_FILE = find_workspace_paths()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src')))

try:
    from parsers.ud_parser import Sentence, Word
except ImportError:
    class Word:
        def __init__(self, form, upos):
            self.form = form
            self.upos = upos

# ============================================================================
# CONFIGURATION
# ============================================================================

NEGATION_MARKERS = {'नहीं', 'न', 'मत', 'ना'}
INTENSIFIERS = {'बहुत', 'अत्यधिक', 'ज़्यादा', 'ज्यादा', 'बेहद', 'अत्यंत', 'काफी', 'इतना'}

NEUTRAL_VALENCE = 0.0   # Center of [-1, +1]
NEUTRAL_AROUSAL = 0.5   # Center of [0, 1]
INTENSIFIER_BOOST = 1.2

# ============================================================================
# LEXICON LOADING
# ============================================================================

def load_vad_lexicon(path):
    """
    Parses your specific [0, 1] scaled 5-column lexicon format.
    Transforms Valence back to [-1, +1] to balance positive vs negative vectors.
    """
    if not os.path.exists(path):
        print(f"ERROR: VAD lexicon file not found at: {path}")
        sys.exit(1)

    raw_aggregates = defaultdict(list)

    with open(path, 'r', encoding='utf-8') as f:
        sample = f.readline()
        delimiter = '\t' if '\t' in sample else ' '
        f.seek(0)
        
        for line_num, line in enumerate(f, 1):
            parts = [p.strip() for p in line.strip().split(delimiter) if p.strip()]
            if not parts or len(parts) < 5:
                parts = [p.strip() for p in line.strip().split() if p.strip()]
                if len(parts) < 5:
                    continue

            if parts[0].lower() == 'english' or parts[4] == 'Hindi':
                continue  

            try:
                v_raw = float(parts[1])
                a_raw = float(parts[2])
                hindi_raw = parts[4].split()[0]
                
                # --- RECALIBRATION TO THE STANDARD PROJECT SCALE ---
                # Your file is [0, 1]. Map Valence to [-1, +1] (0.5 becomes neutral 0.0)
                valence_scaled = (v_raw - 0.5) * 2.0
                # Arousal is kept exactly as a [0, 1] intensity float
                arousal_scaled = a_raw
                
            except (ValueError, IndexError):
                continue  

            raw_aggregates[hindi_raw].append((valence_scaled, arousal_scaled))

    lexicon = {}
    for hi_word, scores in raw_aggregates.items():
        avg_v = sum(s[0] for s in scores) / len(scores)
        avg_a = sum(s[1] for s in scores) / len(scores)
        lexicon[hi_word] = (avg_v, avg_a)

    print(f"  Loaded {len(lexicon):,} unique Hindi words from `{os.path.basename(path)}`")
    return lexicon

# ============================================================================
# EMOTION SCORING
# ============================================================================

def score_sentence(words, lexicon):
    valence_vals = []
    arousal_vals = []
    n_words = len(words)
    n_matched = 0

    for i, w in enumerate(words):
        form = w.form.strip() if hasattr(w, 'form') else str(w).strip()
        
        if form not in lexicon:
            continue

        v_scaled, a_scaled = lexicon[form]
        n_matched += 1

        valence = v_scaled
        arousal = a_scaled

        # Apply negation modifier
        if i > 0:
            prev_form = words[i - 1].form.strip() if hasattr(words[i - 1], 'form') else str(words[i - 1]).strip()
            if prev_form in NEGATION_MARKERS:
                valence = -valence

        # Apply intensifier modifier
        if i > 0:
            prev_form = words[i - 1].form.strip() if hasattr(words[i - 1], 'form') else str(words[i - 1]).strip()
            if prev_form in INTENSIFIERS:
                arousal = min(1.0, arousal * INTENSIFIER_BOOST)

        valence_vals.append(valence)
        arousal_vals.append(arousal)

    if not valence_vals:
        return {
            'valence': NEUTRAL_VALENCE,
            'arousal': NEUTRAL_AROUSAL,
            'coverage': 0.0,
            'n_matched': 0,
        }

    return {
        'valence': sum(valence_vals) / len(valence_vals),
        'arousal': sum(arousal_vals) / len(arousal_vals),
        'coverage': n_matched / max(n_words, 1),
        'n_matched': n_matched,
    }

# ============================================================================
# MAIN MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" SCRIPT 16: EMOTION ANNOTATION (RECALIBRATED SCALE)")
    print("=" * 70 + "\n")

    lexicon = load_vad_lexicon(VAD_LEXICON)

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