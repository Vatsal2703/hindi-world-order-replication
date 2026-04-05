import pickle
import pandas as pd
import os

# 1. Load All Necessary Data
print("📂 Loading Models and Data...")
try:
    with open("./data/models/emille_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    reverse_vocab = {v: k for k, v in vocab.items()}

    with open("./data/results/final_scored_variants.pkl", 'rb') as f:
        scored_data = pickle.load(f)

    with open("./data/processed/all_variants_final.pkl", 'rb') as f:
        original_data = pickle.load(f)
except FileNotFoundError as e:
    print(f"❌ Error: Could not find file - {e}")
    exit()

df = pd.DataFrame(scored_data)
overlap_cases = []

# 2. Identify and Filter Overlaps
print("🔍 Analyzing rows for linguistic overlaps (filtering noise)...")
for sent_id, group in df.groupby('sent_id'):
    ref_row = group[group['is_reference'] == True]
    variants = group[group['is_reference'] == False]
    
    if ref_row.empty or variants.empty:
        continue
        
    ref_score = ref_row['avg_surprisal'].values[0]
    
    # Find the variant with the lowest surprisal (the one the model liked most)
    best_variant_idx = variants['avg_surprisal'].idxmin()
    best_var_score = variants.loc[best_variant_idx, 'avg_surprisal']
    
    # Check if the variant beat the reference
    if best_var_score < ref_score:
        # Get the word IDs for the reference to check for noise/special tokens
        ref_match = next((item for item in original_data if item['sent_id'] == sent_id and item['is_reference']), None)
        if not ref_match: continue
        
        words = [reverse_vocab.get(i, "<UNK>") for i in ref_match['variant_order']]
        
        # --- LINGUISTIC FILTERING ---
        # Skip sentences that are just noise (special tokens like <s> or </s>)
        noise_tokens = {'<s>', '</s>'}
        if any(t in words for t in noise_tokens):
            continue
            
        overlap_cases.append({
            'sent_id': sent_id,
            'ref_surprisal': ref_score,
            'variant_surprisal': best_var_score,
            'score_diff': ref_score - best_var_score,
            'hindi_text': " ".join(words),
            'best_var_idx': best_variant_idx # To find the winning variant text later
        })

# 3. Process and Display Results
if not overlap_cases:
    print("\n⚠️ No clean overlap cases found with current filters.")
    print("This might mean all overlaps were noise, or we need to relax the filters.")
else:
    overlap_df = pd.DataFrame(overlap_cases).sort_values(by='score_diff', ascending=False)

    print(f"\n✅ Found {len(overlap_df)} CLEAN linguistic overlapping cases.")
    print("\nTop 10 Actual Hindi Sentences where Scrambled > Reference:")
    print("-" * 100)

    for _, row in overlap_df.head(10).iterrows():
        print(f"ID: {row['sent_id']} | Diff: {row['score_diff']:.4f}")
        print(f"Reference: {row['hindi_text']}")
        
        # Get the actual winning variant text
        # We find the original data item that matches the sent_id and has the correct word order
        best_var_match = next((item for item in original_data if item['sent_id'] == row['sent_id'] and not item['is_reference']), None)
        
        if best_var_match:
            var_words = [reverse_vocab.get(i, "<UNK>") for i in best_var_match['variant_order']]
            print(f"Variant  : {' '.join(var_words)}")
        
        print("-" * 100)