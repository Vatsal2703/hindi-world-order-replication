import os
import pickle
from tqdm import tqdm

EMILLE_DIR = "./data/processed/silver_emille/"
OUTPUT_FILE = "./data/processed/emille_tokenized.pkl"

def preprocess_emille():
    print("\n" + "="*70)
    print(" PREPROCESSING EMILLE CORPUS (.CONLLU)")
    print("="*70)
    
    all_sentences = []
    conllu_files = [f for f in os.listdir(EMILLE_DIR) if f.endswith('.conllu')]
    
    for filename in tqdm(conllu_files, desc="Parsing Files"):
        filepath = os.path.join(EMILLE_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            current_sent = []
            for line in f:
                if line.startswith('#'): continue
                if line.strip() == "":
                    if current_sent:
                        all_sentences.append(current_sent)
                        current_sent = []
                    continue
                
                parts = line.split('\t')
                if len(parts) > 1:
                    word = parts[1] # The FORM field
                    current_sent.append(word)
            
            # Catch last sentence in file
            if current_sent:
                all_sentences.append(current_sent)

    print(f"\nExtracted {len(all_sentences):,} sentences.")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_sentences, f)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_emille()