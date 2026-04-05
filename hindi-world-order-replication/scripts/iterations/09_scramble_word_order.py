import os
import random

def get_random_dl(file_path):
    sentence_dls = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            if line.strip() and not line.startswith('#'):
                sentence.append(line.split('\t'))
            elif sentence:
                # We have a full sentence. Let's shuffle the IDs.
                n = len(sentence)
                original_indices = list(range(1, n + 1))
                shuffled_indices = original_indices[:]
                random.shuffle(shuffled_indices)
                
                # Create a mapping from Old Position to New Position
                mapping = {old: new for old, new in zip(original_indices, shuffled_indices)}
                
                total_dl = 0
                for i, word_data in enumerate(sentence):
                    old_idx = int(word_data[0])
                    old_head = int(word_data[6])
                    
                    if old_head != 0:
                        new_idx = mapping[old_idx]
                        new_head = mapping[old_head]
                        total_dl += abs(new_idx - new_head)
                
                sentence_dls.append(total_dl / n)
                sentence = []
                
    return sum(sentence_dls) / len(sentence_dls) if sentence_dls else 0

# Run across the corpus
silver_dir = "./data/processed/silver_emille"
random_results = []

for f in os.listdir(silver_dir):
    if f.endswith(".conllu"):
        random_results.append(get_random_dl(os.path.join(silver_dir, f)))

print(f"Randomly Scrambled DL: {sum(random_results)/len(random_results):.3f}")
print(f"Real EMILLE DL: 3.662")