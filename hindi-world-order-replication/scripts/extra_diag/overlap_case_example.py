import pickle

# 1. Load Vocab (Reverse it for ID -> Word)
with open("./data/models/emille_vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
reverse_vocab = {v: k for k, v in vocab.items()}

# 2. Load the original variants data (to get the word IDs)
with open("./data/processed/all_variants_final.pkl", 'rb') as f:
    all_data = pickle.load(f)

# 3. Target IDs from your top results
target_ids = ['train-s8168', 'train-s1089','train-s5703', 'train-s9825'] # Add more from your list!

print("🔍 Decoding Overlap Cases:")
print("-" * 50)

for target in target_ids:
    # Find the reference and the variants for this ID
    matching_items = [item for item in all_data if item['sent_id'] == target]
    
    for item in matching_items:
        # Convert IDs back to words
        words = [reverse_vocab.get(i, "<UNK>") for i in item['variant_order']]
        sentence_text = " ".join(words)
        
        label = "REFERENCE" if item['is_reference'] else "SCRAMBLED"
        print(f"[{target}] ({label}): {sentence_text}")
    print("-" * 50)