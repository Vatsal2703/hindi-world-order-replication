import pickle
import os
import pandas as pd
from tqdm import tqdm

VARIANTS_PICKLE = "data/processed/all_variants_final.pkl"
OUTPUT_TEXT = "data/interim/variants_raw_text.txt"
OUTPUT_MAPPING = "data/interim/variants_mapping.csv"

def main():
    print("Exporting variants for Surprisal calculation...")
    
    if not os.path.exists(VARIANTS_PICKLE):
        print(f"Error: {VARIANTS_PICKLE} not found.")
        return

    with open(VARIANTS_PICKLE, 'rb') as f:
        all_variants = pickle.load(f)

    os.makedirs("data/interim", exist_ok=True)
    mapping_data = []
    
    with open(OUTPUT_TEXT, 'w', encoding='utf-8') as f:
        for i, v in enumerate(tqdm(all_variants, desc="Processing")):
            
            # --- ROBUST TEXT EXTRACTION ---
            # If 'preverbal_words' is missing, let's look for 'words' or 'variant_words'
            if 'preverbal_words' in v:
                words = v['preverbal_words']
            elif 'words' in v:
                words = v['words']
            else:
                # Fallback: manually construct from the provided data
                # Adjust this if your Script 05 uses a different key
                words = v.get('variant_words', [])

            root = v.get('root_form', '')
            
            # Combine preverbal constituents + root verb
            sentence_str = " ".join(words) + " " + root
            sentence_str = sentence_str.strip()
            
            f.write(sentence_str + "\n")
            
            mapping_data.append({
                'line_number': i,
                'sent_id': v.get('sent_id', 'unknown'),
                'is_reference': v.get('is_reference', False)
            })

    df_map = pd.DataFrame(mapping_data)
    df_map.to_csv(OUTPUT_MAPPING, index=False)
    
    print(f"\nSuccess! Exported {len(all_variants)} lines.")

if __name__ == "__main__":
    main()