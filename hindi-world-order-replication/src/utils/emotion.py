"""Shared VAD-lexicon and emotion-scoring utilities for the MTP-II emotion
pipeline (scripts 16-22): loading the Hindi NRC-VAD lexicon, scoring a
sentence's valence/arousal, reconstructing SOV/DOSV/IOSV construction type,
and formatting significance stars.
"""

import os
import sys
from collections import defaultdict

NEGATION_MARKERS = {'नहीं', 'न', 'मत', 'ना'}
INTENSIFIERS = {'बहुत', 'अत्यधिक', 'ज़्यादा', 'ज्यादा', 'बेहद', 'अत्यंत', 'काफी', 'इतना'}
INTENSIFIER_BOOST = 1.2

NEUTRAL_VALENCE = 0.0   # Center of [-1, +1]
NEUTRAL_AROUSAL = 0.5   # Center of [0, 1]

SUBJECT_RELS = {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}


def load_vad_lexicon(path):
    """
    Parse the [0, 1]-scaled 5-column Hindi-NRC-VAD lexicon.
    Valence is rescaled to [-1, +1] (0.5 -> neutral 0.0) so positive and
    negative vectors balance; arousal is kept as a [0, 1] intensity float.
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
                hindi_word = parts[4].split()[0]
            except (ValueError, IndexError):
                continue

            valence_scaled = (v_raw - 0.5) * 2.0
            raw_aggregates[hindi_word].append((valence_scaled, a_raw))

    lexicon = {
        word: (sum(s[0] for s in scores) / len(scores), sum(s[1] for s in scores) / len(scores))
        for word, scores in raw_aggregates.items()
    }
    return lexicon


def _form_of(w):
    return w.form.strip() if hasattr(w, 'form') else str(w).strip()


def score_sentence(words, lexicon, intensifier_boost=INTENSIFIER_BOOST):
    """
    Score a sentence's (valence, arousal) from its lexicon-matched words,
    applying negation flip and intensifier boost based on the preceding token.
    `words` may be Word objects (with `.form`) or plain strings.
    """
    valence_vals, arousal_vals = [], []
    n_matched = 0

    for i, w in enumerate(words):
        form = _form_of(w)
        if form not in lexicon:
            continue

        valence, arousal = lexicon[form]
        n_matched += 1

        if i > 0:
            prev_form = _form_of(words[i - 1])
            if prev_form in NEGATION_MARKERS:
                valence = -valence
            if prev_form in INTENSIFIERS:
                arousal = min(1.0, arousal * intensifier_boost)

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
