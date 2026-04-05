import os

def calculate_dl(file_path):
    total_dl = 0
    word_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) > 6:
                    idx = int(parts[0])
                    head = int(parts[6])
                    # DL is the distance between a word and its head
                    if head != 0: # Skip root
                        total_dl += abs(idx - head)
                        word_count += 1
                        
    return total_dl / word_count if word_count > 0 else 0

# Run for the whole directory
silver_dir = "./data/processed/silver_emille"
all_dls = []

for f in os.listdir(silver_dir):
    if f.endswith(".conllu"):
        dl = calculate_dl(os.path.join(silver_dir, f))
        all_dls.append(dl)

avg_corpus_dl = sum(all_dls) / len(all_dls)
print(f"Average Dependency Length for EMILLE Hindi: {avg_corpus_dl:.3f}")