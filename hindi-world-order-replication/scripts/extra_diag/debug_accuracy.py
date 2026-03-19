import sys
import os

# 1. Get the absolute path of the directory where THIS script is
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to reach the project root (hindi-world-order-replication)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# 3. Add both the root and the src directory to Python's search path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pickle
# Now this should work!
# Load the data
VARIANTS_FILE = "./data/processed/all_variants_final.pkl"
SENTENCES_FILE = "./data/processed/replication_filtered_sentences.pkl"

with open(VARIANTS_FILE, 'rb') as f:
    all_variants = pickle.load(f)

with open(SENTENCES_FILE, 'rb') as f:
    sentences = pickle.load(f)
sent_lookup = {s.sent_id: s for s in sentences}

print("="*70)
print(" DEBUGGING 98% ACCURACY: SUBTREE INTEGRITY CHECK")
print("="*70)

# Pick one sentence with multiple variants
sample_id = all_variants[0]['sent_id']
sample_sent = sent_lookup[sample_id]
sample_variants = [v for v in all_variants if v['sent_id'] == sample_id]

print(f"\nOriginal Sentence: {sample_sent.text}")
print(f"Original Order (Indices): {[w.idx for w in sample_sent.words]}")

for i, v in enumerate(sample_variants[:3]):
    is_ref = " [REFERENCE]" if v['is_reference'] else ""
    print(f"\nVariant {i}{is_ref}:")
    print(f" - Order: {v['variant_order']}")
    
    # Reconstruct the text using the variant_order
    # variant_order is a list of word indices. We map them back to forms.
    try:
        word_map = {w.idx: w.form for w in sample_sent.words}
        variant_text = " ".join([word_map[idx] for idx in v['variant_order']])
        print(f" - Text: {variant_text}")
    except Exception as e:
        print(f" - Error mapping text: {e}")

# Check for "The Verb Tell"
print("\n" + "-"*30)
print("VERB POSITION CHECK")
last_word_is_verb = 0
for v in all_variants:
    last_idx = v['variant_order'][-1]
    last_word = next(w for w in sent_lookup[v['sent_id']].words if w.idx == last_idx)
    if last_word.upos in ['VERB', 'AUX', 'PUNCT']:
        last_word_is_verb += 1

print(f"Sentences where last word is Verb/Punct: {last_word_is_verb}/{len(all_variants)}")