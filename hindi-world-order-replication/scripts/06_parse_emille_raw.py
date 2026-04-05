#!/usr/bin/env python3
"""
Script 15b: Parse raw EMILLE W0037 corpus (UTF-16, CES XML format)
into tokenized sentences for LSTM training.

Paper: LSTM trained on the written section of the EMILLE Hindi Corpus
       (~1 million mixed genre sentences).

Input:  W0037/monoling/hindi/written/  (2224 .txt files, UTF-16 CES XML)
Output: data/processed/emille_written_tokenized.pkl
        List of sentences, each sentence is a list of word strings.
"""

import os
import re
import pickle
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
EMILLE_WRITTEN_DIR = "../../W0037/monoling/hindi/written"
OUTPUT_FILE        = "./data/processed/emille_written_tokenized.pkl"

# Resolve path relative to this script's location
script_dir = Path(__file__).parent
emille_dir = (script_dir / EMILLE_WRITTEN_DIR).resolve()

print("\n" + "="*70)
print(" PARSING RAW EMILLE HINDI WRITTEN CORPUS")
print("="*70)
print(f"\n  Source: {emille_dir}")

if not emille_dir.exists():
    print(f"\nERROR: EMILLE directory not found: {emille_dir}")
    print("Expected W0037 at: ../../../W0037/")
    exit(1)

# ============================================================================
# COLLECT ALL .txt FILES
# ============================================================================
all_files = list(emille_dir.rglob("*.txt"))
print(f"  Found {len(all_files):,} .txt files\n")

# ============================================================================
# PARSE EACH FILE
# ============================================================================
# EMILLE CES XML format:
#   - UTF-16 LE encoding (with BOM)
#   - Text content inside <p>...</p> tags (may span lines)
#   - Devanagari Unicode text, space-separated words

TAG_RE = re.compile(r'<[^>]+>')

def extract_sentences_from_file(filepath):
    """
    Parse one EMILLE CES file and return list of token lists.

    Strategy: extract all text lines inside <p>...</p> blocks.
    Each non-empty line is treated as one sentence unit — this matches
    the original EMILLE processing that produced ~1M sentences.
    We do NOT split on dandas (।) within a line, as that fragments
    naturally flowing text into sub-sentence fragments.
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-16')
    except (UnicodeDecodeError, UnicodeError):
        try:
            with open(filepath, 'r', encoding='utf-16-le') as f:
                text = f.read()
        except Exception:
            return []

    sentences = []

    # Extract all <p>...</p> blocks
    para_re = re.compile(r'<p>(.*?)</p>', re.DOTALL | re.IGNORECASE)
    paras = para_re.findall(text)

    if not paras:
        # Fallback: strip all tags, use non-empty lines
        clean = TAG_RE.sub(' ', text)
        paras = [clean]

    for para in paras:
        # Strip XML tags inside paragraph
        para = TAG_RE.sub(' ', para)

        # Split paragraph into lines — each line is a sentence unit
        lines = para.split('\n')
        for line in lines:
            line = re.sub(r'\s+', ' ', line).strip()
            if not line:
                continue

            # Tokenize by whitespace
            tokens = line.split()

            # Keep only lines with at least 3 Hindi-script tokens
            hindi_count = sum(1 for t in tokens
                              if any('\u0900' <= c <= '\u097F' for c in t))
            if hindi_count < 3:
                continue

            sentences.append(tokens)

    return sentences


all_sentences = []
errors = 0

for filepath in tqdm(all_files, desc="Parsing EMILLE files"):
    try:
        sents = extract_sentences_from_file(filepath)
        all_sentences.extend(sents)
    except Exception as e:
        errors += 1

print(f"\n  Parsed:  {len(all_sentences):,} sentences")
print(f"  Errors:  {errors}")
print(f"  Target:  ~1,000,000 (paper)")

# ============================================================================
# SAVE
# ============================================================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(all_sentences, f)

print(f"\n  Saved → {OUTPUT_FILE}")
print("\nNext: Run scripts/17_train_LSTM.py")
print("="*70 + "\n")
