#!/usr/bin/env python3
from typing import Set, Dict, List

PUNCTUATION_TAGS = {'PUNCT', 'SYM', 'punc', '.', ',', ':', ';', '!', '?'}
PUNCTUATION_CHARS = set('।,.!?;:"\'()[]{}॥')

def is_punctuation(word) -> bool:
    # Safely check upos and xpos (using getattr to avoid attribute errors)
    upos = getattr(word, 'upos', '')
    xpos = getattr(word, 'xpos', '')
    
    if upos in PUNCTUATION_TAGS or xpos in PUNCTUATION_TAGS:
        return True
    if word.form in PUNCTUATION_CHARS:
        return True
    return False

def get_word_by_idx(sentence, idx):
    """Retrieves a word object from a sentence by its index."""
    for w in sentence.words:
        if w.idx == idx:
            return w
    return None

def calculate_dependency_length_temperley(sentence, word_order: List[int]) -> int:
    """
    Calculates DLM based on current word position in the variant.
    """
    # MAP ID -> POSITION (The fix for the zeros)
    pos_map = {idx: pos for pos, idx in enumerate(word_order)}
    total_length = 0
    
    for word in sentence.words:
        if word.head == 0 : # or is_punctuation(word):
            continue
        
        # Find where word and its head are in THIS variant
        curr_pos = pos_map.get(word.idx)
        head_pos = pos_map.get(word.head)
        
        if curr_pos is not None and head_pos is not None:
            # Distance is absolute difference in positions
            # We subtract 1 to get the number of 'intervening' words
            dist = abs(curr_pos - head_pos) - 1
            total_length += max(0, dist)
    
    return total_length

def calculate_information_status_score(sentence, word_order: List[int], context_sentence=None) -> int:
    """
    Scores Given-New (+1) vs New-Given (-1).
    """
    preverbal = sentence.get_preverbal_constituents()
    pre_idxs = [w.idx for w in preverbal]
    
    # Get current order of preverbal elements
    ordered_pre_idxs = [idx for idx in word_order if idx in pre_idxs]
    
    if len(ordered_pre_idxs) < 2:
        return 0
    
    context_lemmas = set()
    if context_sentence:
        context_lemmas = {w.lemma.lower() if w.lemma != '_' else w.form.lower() 
                         for w in context_sentence.words if not is_punctuation(w)}
    
    def check_given(idx):
        w = get_word_by_idx(sentence, idx)
        if not w: return False
        if w.upos == 'PRON': return True
        lemma = w.lemma.lower() if w.lemma != '_' else w.form.lower()
        return lemma in context_lemmas

    first_given = check_given(ordered_pre_idxs[0])
    second_given = check_given(ordered_pre_idxs[1])
    
    if first_given and not second_given: return 1   # Given-New
    if not first_given and second_given: return -1  # New-Given
    return 0

def extract_features_for_sentence(sentence, word_order=None, context_sentence=None) -> Dict:
    # Default to original order if none provided
    if word_order is None:
        word_order = [w.idx for w in sentence.words]
        
    return {
        'dep_len_temperley': calculate_dependency_length_temperley(sentence, word_order),
        'info_status_score': calculate_information_status_score(sentence, word_order, context_sentence),
        'sentence_length': len(sentence.words)
    }