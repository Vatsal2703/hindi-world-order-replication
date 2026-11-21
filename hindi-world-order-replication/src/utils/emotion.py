# v1.1 (2026-11-02): script numbers in comments and messages updated; naming only, no logic change.
"""Shared VAD-lexicon and emotion-scoring utilities for the MTP-II emotion
pipeline (scripts 16-20, 23 and the toolkit): loading the Hindi NRC-VAD lexicon, scoring a
sentence's valence/arousal/dominance, reconstructing SOV/DOSV/IOSV
construction type, and formatting significance stars.
"""

import os
import sys
from collections import defaultdict

NEGATION_MARKERS = {'नहीं', 'न', 'मत', 'ना'}
INTENSIFIERS = {'बहुत', 'अत्यधिक', 'ज़्यादा', 'ज्यादा', 'बेहद', 'अत्यंत', 'काफी', 'इतना'}
INTENSIFIER_BOOST = 1.2

NEUTRAL_VALENCE   = 0.0   # Center of [-1, +1]
NEUTRAL_AROUSAL   = 0.5   # Center of [0, 1]
NEUTRAL_DOMINANCE = 0.0   # Center of [-1, +1]

SUBJECT_RELS = {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}


def load_vad_lexicon(path):
    """
    Parse the [0, 1]-scaled 5-column Hindi-NRC-VAD lexicon (English Word,
    Valence, Arousal, Dominance, Hindi Word).
    Valence and dominance are both bipolar axes (pleasant/unpleasant,
    powerful/weak) so both are rescaled from [0, 1] to [-1, +1]
    (0.5 -> neutral 0.0); arousal (calm/activated) is kept as a [0, 1]
    intensity float.
    """
    if not os.path.exists(path):
        sys.exit(f"ERROR: VAD lexicon not found: {path}")

    raw_aggregates = defaultdict(list)

    with open(path, 'r', encoding='utf-8') as f:
        sample = f.readline()
        delimiter = '\t' if '\t' in sample else ' '
        f.seek(0)

        for line in f:
            parts = [p.strip() for p in line.strip().split(delimiter) if p.strip()]
            if len(parts) < 5:
                parts = [p.strip() for p in line.strip().split() if p.strip()]
                if len(parts) < 5:
                    continue

            if parts[0].lower() == 'english' or parts[4] == 'Hindi':
                continue

            try:
                v_raw = float(parts[1])
                a_raw = float(parts[2])
                d_raw = float(parts[3])
                hindi_word = parts[4].split()[0]
            except (ValueError, IndexError):
                continue

            valence_scaled = (v_raw - 0.5) * 2.0
            dominance_scaled = (d_raw - 0.5) * 2.0
            raw_aggregates[hindi_word].append((valence_scaled, a_raw, dominance_scaled))

    lexicon = {
        word: (sum(s[0] for s in scores) / len(scores),
               sum(s[1] for s in scores) / len(scores),
               sum(s[2] for s in scores) / len(scores))
        for word, scores in raw_aggregates.items()
    }
    return lexicon


def _form_of(w):
    return w.form.strip() if hasattr(w, 'form') else str(w).strip()


def score_sentence(words, lexicon, intensifier_boost=INTENSIFIER_BOOST):
    """
    Score a sentence's (valence, arousal, dominance) from its lexicon-matched
    words, applying negation flip and intensifier boost based on the
    preceding token. `words` may be Word objects (with `.form`) or plain
    strings.
    """
    valence_vals, arousal_vals, dominance_vals = [], [], []
    n_matched = 0

    for i, w in enumerate(words):
        form = _form_of(w)
        if form not in lexicon:
            continue

        valence, arousal, dominance = lexicon[form]
        n_matched += 1

        if i > 0:
            prev_form = _form_of(words[i - 1])
            if prev_form in NEGATION_MARKERS:
                # Negating an emotion word inverts its dominance too
                # (e.g. "not powerful"), same as it inverts valence.
                valence = -valence
                dominance = -dominance
            if prev_form in INTENSIFIERS:
                # Intensifiers scale intensity, which is the arousal axis;
                # dominance (in-control vs controlled) is left unchanged.
                arousal = min(1.0, arousal * intensifier_boost)

        valence_vals.append(valence)
        arousal_vals.append(arousal)
        dominance_vals.append(dominance)

    if not valence_vals:
        return {
            'valence': NEUTRAL_VALENCE,
            'arousal': NEUTRAL_AROUSAL,
            'dominance': NEUTRAL_DOMINANCE,
            'coverage': 0.0,
            'n_matched': 0,
        }

    return {
        'valence': sum(valence_vals) / len(valence_vals),
        'arousal': sum(arousal_vals) / len(arousal_vals),
        'dominance': sum(dominance_vals) / len(dominance_vals),
        'coverage': n_matched / max(len(words), 1),
        'n_matched': n_matched,
    }


def get_construction_type(sent):
    """
    Classify a parsed sentence as SOV / DOSV / IOSV / UNKNOWN from the
    preverbal order of subject vs. objects in its dependency parse.
    """
    if getattr(sent, "root_idx", None) is None:
        return "UNKNOWN"

    try:
        preverbal = sent.get_preverbal_constituents()
    except Exception:
        return "UNKNOWN"

    subj_pos = obj_pos = iobj_pos = None
    for i, w in enumerate(preverbal):
        if w.deprel in SUBJECT_RELS and subj_pos is None:
            subj_pos = i
        elif w.deprel == "obj" and obj_pos is None:
            obj_pos = i
        elif w.deprel == "iobj" and iobj_pos is None:
            iobj_pos = i

    if subj_pos is None:
        return "UNKNOWN"
    if obj_pos is not None and obj_pos < subj_pos:
        return "DOSV"
    if iobj_pos is not None and iobj_pos < subj_pos:
        return "IOSV"
    return "SOV"


def sig_mark(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
