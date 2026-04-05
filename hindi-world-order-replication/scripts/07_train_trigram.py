import os
import pickle
import sys
from collections import Counter, defaultdict
from tqdm import tqdm

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Config
FULL_CORPUS = "./data/processed/full_hutb_sentences.pkl"
REF_SENTENCES = "./data/processed/replication_filtered_sentences.pkl"
EMILLE_CORPUS = "./data/processed/emille_written_tokenized.pkl"
MODEL_OUTPUT = "./data/models/trigram_model_blind.pkl"

def train_blind_model():
    print("\n" + "="*70)
    print(" TRAINING BLIND N-GRAM MODEL WITH KNESER-NEY SMOOTHING")
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

    # 3. Load EMILLE corpus (already tokenized as list-of-token-lists)
    with open(EMILLE_CORPUS, 'rb') as f:
        emille_sentences = pickle.load(f)
    print(f" EMILLE sentences: {len(emille_sentences):,}")

    # 4. Build raw N-gram counts (HUTB train + EMILLE)
    unigrams, bigrams, trigrams = Counter(), Counter(), Counter()

    for sent in tqdm(train_sentences, desc="Counting n-grams (HUTB)"):
        tokens = ['<s>', '<s>'] + [w.form for w in sent.words] + ['</s>']
        for i in range(len(tokens)):
            unigrams[tokens[i]] += 1
            if i >= 1: bigrams[(tokens[i-1], tokens[i])] += 1
            if i >= 2: trigrams[(tokens[i-2], tokens[i-1], tokens[i])] += 1

    for token_list in tqdm(emille_sentences, desc="Counting n-grams (EMILLE)"):
        tokens = ['<s>', '<s>'] + token_list + ['</s>']
        for i in range(len(tokens)):
            unigrams[tokens[i]] += 1
            if i >= 1: bigrams[(tokens[i-1], tokens[i])] += 1
            if i >= 2: trigrams[(tokens[i-2], tokens[i-1], tokens[i])] += 1

    print(f" Unigrams: {len(unigrams):,}  Bigrams: {len(bigrams):,}  Trigrams: {len(trigrams):,}")

    # 4. Compute Kneser-Ney continuation counts
    # D = standard discount
    D = 0.75

    # bigram_followers_count[(w1,w2)] = N1+(w1,w2,·) = distinct words following bigram (w1,w2)
    bigram_followers = defaultdict(set)
    # bigram_continuation_count[(w2,w3)] = N1+(·,w2,w3) = distinct left contexts for (w2,w3)
    bigram_continuation = defaultdict(set)

    print("Computing KN continuation counts...")
    for (w1, w2, w3) in trigrams:
        bigram_followers[(w1, w2)].add(w3)
        bigram_continuation[(w2, w3)].add(w1)

    bigram_followers_count   = {k: len(v) for k, v in bigram_followers.items()}
    bigram_continuation_count = {k: len(v) for k, v in bigram_continuation.items()}

    # bigram_cont_total[w2] = N1+(·,w2,·) = sum over w3 of N1+(·,w2,w3)
    # used as denominator at bigram level
    bigram_cont_total = defaultdict(int)
    for (w2, w3), cnt in bigram_continuation_count.items():
        bigram_cont_total[w2] += cnt

    # bigram_unique_followers[w2] = N1+(w2,·) in continuation sense
    # = number of distinct w3 such that bigram_continuation_count[(w2,w3)] > 0
    bigram_unique_followers = defaultdict(int)
    for (w2, w3) in bigram_continuation_count:
        bigram_unique_followers[w2] += 1

    # unigram_continuation_count[w3] = N1+(·,w3) = distinct bigrams ending in w3
    unigram_continuation = defaultdict(set)
    for (w2, w3) in bigrams:
        unigram_continuation[w3].add(w2)
    unigram_continuation_count = {k: len(v) for k, v in unigram_continuation.items()}

    # total distinct bigram types (denominator for unigram KN)
    total_bigram_types = len(bigrams)

    print(f" KN counts computed. D={D}")

    # 5. Save
    model_data = {
        'unigrams': unigrams,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'vocab_size': len(unigrams),
        # Kneser-Ney extras
        'D': D,
        'bigram_followers_count': dict(bigram_followers_count),
        'bigram_continuation_count': dict(bigram_continuation_count),
        'bigram_cont_total': dict(bigram_cont_total),
        'bigram_unique_followers': dict(bigram_unique_followers),
        'unigram_continuation_count': unigram_continuation_count,
        'total_bigram_types': total_bigram_types,
    }

    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nBlind Model (KN) Saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_blind_model()