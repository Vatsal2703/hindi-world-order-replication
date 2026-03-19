import os
import pickle
import sys
from collections import Counter
from tqdm import tqdm

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Config
FULL_CORPUS = "./data/processed/full_hutb_sentences.pkl"
REF_SENTENCES = "./data/processed/reference_sentences.pkl"
MODEL_OUTPUT = "./data/models/trigram_model_blind.pkl"

def train_blind_model():
    print("\n" + "="*70)
    print(" TRAINING BLIND N-GRAM MODEL (PREVENTING DATA LEAK)")
    print("="*70)

    # Load data
    with open(FULL_CORPUS, 'rb') as f:
        full_corpus = pickle.load(f)
    with open(REF_SENTENCES, 'rb') as f:
        refs = pickle.load(f)

    # 1. Identify IDs to exclude (The Test Set)
    test_ids = {s.sent_id for s in refs}
    print(f" Excluding {len(test_ids)} test sentences from training...")

    # 2. Filter
    train_sentences = [s for s in full_corpus if s.sent_id not in test_ids]
    print(f" Training on remaining {len(train_sentences):,} sentences.")

    # 3. Build N-grams
    unigrams, bigrams, trigrams = Counter(), Counter(), Counter()

    for sent in tqdm(train_sentences):
        tokens = ['<s>', '<s>'] + [w.form for w in sent.words] + ['</s>']
        for i in range(len(tokens)):
            unigrams[tokens[i]] += 1
            if i >= 1: bigrams[(tokens[i-1], tokens[i])] += 1
            if i >= 2: trigrams[(tokens[i-2], tokens[i-1], tokens[i])] += 1

    # 4. Save
    model_data = {
        'unigrams': unigrams, 'bigrams': bigrams, 
        'trigrams': trigrams, 'vocab_size': len(unigrams)
    }
    
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Blind Model Saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_blind_model()