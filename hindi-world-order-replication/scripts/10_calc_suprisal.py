import os
import math
from collections import Counter, defaultdict

def get_surprisal_metrics(silver_dir):
    tokens = []
    # 1. Collect all tokens from your Silver Standard
    for f in os.listdir(silver_dir):
        if f.endswith(".conllu"):
            with open(os.path.join(silver_dir, f), 'r') as file:
                for line in file:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) > 1:
                            tokens.append(parts[1].lower()) # Use words

    # 2. Build N-gram counts
    unigrams = Counter(tokens)
    bigrams = Counter(zip(tokens, tokens[1:]))
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
    
    total_words = len(tokens)
    total_surprisal = 0
    count = 0

    # 3. Calculate Average Trigram Surprisal (with simple add-1 smoothing)
    vocab_size = len(unigrams)
    for i in range(2, len(tokens)):
        context = (tokens[i-2], tokens[i-1])
        target = tokens[i]
        
        # P(target | context) = count(context + target) / count(context)
        context_count = bigrams[context]
        target_count = trigrams[context + (target,)]
        
        # Laplacian smoothing to avoid log(0)
        prob = (target_count + 1) / (context_count + vocab_size)
        total_surprisal += -math.log2(prob)
        count += 1

    return total_surprisal / count if count > 0 else 0

silver_dir = "./data/processed/silver_emille"
avg_s = get_surprisal_metrics(silver_dir)
print(f"Average Trigram Surprisal: {avg_s:.4f} bits")