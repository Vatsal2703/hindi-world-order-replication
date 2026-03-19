import sys
import os
import pickle
import itertools
import random
from tqdm import tqdm
from pathlib import Path

# Add project root to path for custom parsers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "./data/processed/reference_sentences.pkl"
OUTPUT_FILE = "./data/processed/all_variants_final.pkl"

def get_subtree_indices(word_idx, sentence):
    """
    Recursively finds all indices belonging to a word's subtree.
    Ensures that Nouns and their Case Markers/Adjectives stay together.
    """
    subtree = [word_idx]
    # Find all words that have this word_idx as their head
    children = [w.idx for w in sentence.words if w.head == word_idx]
    for child_idx in children:
        subtree.extend(get_subtree_indices(child_idx, sentence))
    return sorted(subtree)

def generate_subtree_variants(sentence, max_variants=100):
    """
    Generates variants by permuting preverbal phrase blocks (Subtrees).
    """
    # 1. Identify the Verb (Root) - usually the final word in Hindi
    verb = next((w for w in sentence.words if w.deprel == 'root'), None)
    if not verb:
        return []

    # 2. Find direct dependents of the verb that appear BEFORE the verb
    # These are the 'Heads' of our blocks (Subject head, Object head, etc.)
    preverbal_deps = [w for w in sentence.words if w.head == verb.idx and w.idx < verb.idx]
    
    if len(preverbal_deps) < 2:
        return [] 

    # 3. Create 'Subtree Blocks'
    # This prevents the 'Shredding' that caused 98% accuracy
    blocks = []
    for dep in preverbal_deps:
        block_indices = get_subtree_indices(dep.idx, sentence)
        blocks.append(block_indices)

    # 4. Identify Fixed indices (The Verb and anything after it, like Punctuation)
    all_block_indices = [idx for block in blocks for idx in block]
    fixed_indices = [w.idx for w in sentence.words if w.idx not in all_block_indices]

    # 5. Permute the BLOCKS (not individual words)
    all_perms = list(itertools.permutations(blocks))
    
    # Cap the variants as per the professor's 100-variant cutoff
    if len(all_perms) > max_variants:
        all_perms = random.sample(all_perms, max_variants)

    variants = []
    for perm in all_perms:
        # Flatten the permuted blocks + add fixed indices at the end
        new_order = [idx for block in perm for idx in block]
        new_order.extend(fixed_indices)
        
        # Check if this permutation is actually the original reference order
        is_ref = (new_order == [w.idx for w in sentence.words])
        
        variants.append({
            'sent_id': sentence.sent_id,
            'variant_order': new_order,
            'is_reference': is_ref
        })

    return variants

def main():
    print("\n" + "="*70)
    print(" STEP 2: SUBTREE-AWARE VARIANT GENERATION")
    print("="*70 + "\n")

    if not os.path.exists(INPUT_FILE):
        print(f" ERROR: {INPUT_FILE} not found. Run Step 1 first!")
        return

    with open(INPUT_FILE, 'rb') as f:
        sentences = pickle.load(f)

    all_results = []
    print(f" Generating variants for {len(sentences)} reference sentences...")

    for sent in tqdm(sentences):
        # Generate variants using the Subtree-Aware logic
        vars_for_sent = generate_subtree_variants(sent)
        all_results.extend(vars_for_sent)

    # Save for Step 3 (Feature Extraction)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n Success! Saved {len(all_results):,} variants.")
    return 0

if __name__ == "__main__":
    main()