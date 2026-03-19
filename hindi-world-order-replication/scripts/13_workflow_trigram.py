import sys
import os
import pickle
import math
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.basic_features import extract_features_for_sentence

# ============================================================================
# CONFIGURATION
# ============================================================================
VARIANTS_PICKLE = "./data/processed/all_variants_final.pkl"
INPUT_FILE = "./data/processed/reference_sentences.pkl"
MODEL_FILE = "./data/models/trigram_model_blind.pkl"

# Use a NEW filename for features so we don't overwrite the 60% baseline
OUTPUT_FEATURES_PKL = "./data/features/pairwise_features_trigram_blind.pkl"
OUTPUT_FEATURES_CSV = "./data/features/pairwise_features_trigram_blind.csv"

def calculate_trigram_surprisal(word_order, sentence, model):
    """
    Calculates total sentence surprisal as per paper: Sum(-log2 P(wi | wi-1, wi-2))
    """
    word_map = {w.idx: w.form for w in sentence.words}
    # <s> <s> are start tokens, </s> is end token
    tokens = ['<s>', '<s>'] + [word_map[idx] for idx in word_order] + ['</s>']
    
    total_s = 0.0
    v_size = model['vocab_size']
    
    for i in range(2, len(tokens)):
        w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
        context_count = model['bigrams'].get((w1, w2), 0)
        trigram_count = model['trigrams'].get((w1, w2, w3), 0)
        
        # Laplace Smoothing (Add-1) to avoid log(0)
        prob = (trigram_count + 1) / (context_count + v_size)
        total_s += -math.log2(prob)
    return total_s

def main():
    print("\n" + "="*70)
    print(" DL + IS + TRIGRAM SURPRISAL")
    print("="*70)

    # Load Model and Data
    with open(MODEL_FILE, 'rb') as f:
        trigram_model = pickle.load(f)
    with open(VARIANTS_PICKLE, 'rb') as f:
        all_variants = pickle.load(f)
    with open(INPUT_FILE, 'rb') as f:
        sentences = pickle.load(f)
    
    sent_lookup = {s.sent_id: s for s in sentences}
    
    # Group variants by sentence
    variants_by_sent = defaultdict(list)
    for v in all_variants:
        if v['sent_id'] in sent_lookup:
            variants_by_sent[v['sent_id']].append(v)

    pair_features = []
    
    # Using tqdm to track progress through the 200k+ variants
    for sent_id, sent_variants in tqdm(variants_by_sent.items(), desc="Processing"):
        original_sent = sent_lookup[sent_id]
        reference = next((v for v in sent_variants if v['is_reference']), None)
        non_refs = [v for v in sent_variants if not v['is_reference']]
        
        if not reference or not non_refs: continue

        # 1. Calculate Reference Features (The 'Human' Choice)
        ref_feat = extract_features_for_sentence(original_sent, word_order=reference['variant_order'])
        ref_surp = calculate_trigram_surprisal(reference['variant_order'], original_sent, trigram_model)

        for var in non_refs:
            # 2. Calculate Variant Features (The 'Computer' Choice)
            var_feat = extract_features_for_sentence(original_sent, word_order=var['variant_order'])
            var_surp = calculate_trigram_surprisal(var['variant_order'], original_sent, trigram_model)

            # 3. Create Balanced Pairwise Data
            # Label 1: Human Ref is better than Computer Var
            pair_features.append({
                'sent_id': sent_id,
                'label': 1,
                'dep_len_diff': ref_feat['dep_len_temperley'] - var_feat['dep_len_temperley'],
                'info_status_diff': ref_feat['info_status_score'] - var_feat['info_status_score'],
                'surprisal_diff': ref_surp - var_surp
            })
            # Label 0: Computer Var is 'better' than Human Ref (Mirror image)
            pair_features.append({
                'sent_id': sent_id,
                'label': 0,
                'dep_len_diff': var_feat['dep_len_temperley'] - ref_feat['dep_len_temperley'],
                'info_status_diff': var_feat['info_status_score'] - ref_feat['info_status_score'],
                'surprisal_diff': var_surp - ref_surp
            })

    # Save to NEW files to keep Experiment 1b separate from Experiment 1a
    df = pd.DataFrame(pair_features)
    os.makedirs("./data/features", exist_ok=True)
    with open(OUTPUT_FEATURES_PKL, 'wb') as f:
        pickle.dump(df, f)
    df.to_csv(OUTPUT_FEATURES_CSV, index=False)
    
    print(f"\nExperiment 1b complete. Features saved to {OUTPUT_FEATURES_PKL}")

if __name__ == "__main__":
    main()