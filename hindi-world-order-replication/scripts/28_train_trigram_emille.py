#!/usr/bin/env python3
"""
Script 28: Retrain Trigram on the Full EMILLE Corpus  — v1.0

MTP-II: the surprisal comparison across trigram/lstm/adaptive/pcfg is only
fair if the models are trained on comparable data. The trigram model in
data/models/trigram_model.pkl was trained on ~401K tokens (19K vocab); the
LSTM (scripts/08_train_LSTM.py) was trained on the full EMILLE written
corpus (~998K sentences, 32K vocab). This script retrains a plain trigram
model on that SAME EMILLE corpus, so the two can be compared on equal
training data.

No Kneser-Ney, no HUTB blending -- just raw unigram/bigram/trigram counts
over data/processed/emille_written_tokenized.pkl, same <s> <s> ... </s>
padding convention as scripts/07_train_trigram.py, saved in the same plain
dict structure the existing trigram model uses:
    {"unigrams": Counter, "bigrams": Counter, "trigrams": Counter,
     "vocab_size": int}

Does NOT touch data/models/trigram_model.pkl -- writes a separate file so
both models remain available for comparison.

RUN IN ANY ENV (pure Python, no torch needed):
    python scripts/28_train_trigram_emille.py

Output: data/models/trigram_model_emille.pkl
"""

VERSION = "1.0"

import os
import sys
import pickle
from collections import Counter
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
from utils.paths import find_base

BASE         = find_base(__file__)
EMILLE_IN    = os.path.join(BASE, "data", "processed", "emille_written_tokenized.pkl")
MODEL_OUTPUT = os.path.join(BASE, "data", "models", "trigram_model_emille.pkl")


def main():
    print("\n" + "=" * 70)
    print(f" SCRIPT 28: RETRAIN TRIGRAM ON FULL EMILLE CORPUS  (v{VERSION})")
    print("=" * 70)

    if not os.path.exists(EMILLE_IN):
        sys.exit(f"ERROR: EMILLE corpus not found: {EMILLE_IN}")

    with open(EMILLE_IN, "rb") as f:
        emille_sentences = pickle.load(f)
    print(f"\n[1] Loaded EMILLE corpus: {len(emille_sentences):,} sentences")

    unigrams, bigrams, trigrams = Counter(), Counter(), Counter()

    print("\n[2] Counting n-grams...")
    for token_list in tqdm(emille_sentences, desc="Counting n-grams (EMILLE)"):
        tokens = ["<s>", "<s>"] + list(token_list) + ["</s>"]
        for i in range(len(tokens)):
            unigrams[tokens[i]] += 1
            if i >= 1:
                bigrams[(tokens[i - 1], tokens[i])] += 1
            if i >= 2:
                trigrams[(tokens[i - 2], tokens[i - 1], tokens[i])] += 1

    total_tokens = sum(unigrams.values())
    vocab_size = len(unigrams)

    model_data = {
        "unigrams": unigrams,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "vocab_size": vocab_size,
    }

    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(model_data, f)

    print("\n" + "-" * 70)
    print(" RESULT")
    print("-" * 70)
    print(f"  Total tokens (incl. <s>/</s>) : {total_tokens:,}")
    print(f"  Vocabulary size (unigram types): {vocab_size:,}")
    print(f"  Bigram types                   : {len(bigrams):,}")
    print(f"  Trigram types                  : {len(trigrams):,}")
    print(f"\n  Saved -> {MODEL_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
