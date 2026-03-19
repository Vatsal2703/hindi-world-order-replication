import os

def analyze_sentence_efficiency(file_path):
    sentences = []
    current_sent = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                current_sent.append(line.split('\t'))
            elif current_sent:
                total_dl = 0
                for word in current_sent:
                    idx, head = int(word[0]), int(word[6])
                    if head != 0:
                        total_dl += abs(idx - head)
                
                avg_dl = total_dl / len(current_sent)
                text = " ".join([w[1] for w in current_sent])
                sentences.append((avg_dl, text))
                current_sent = []

    # Sort by DL efficiency
    sentences.sort()
    
    print("✅ MOST EFFICIENT (Low DL):")
    for dl, text in sentences[:2]:
        print(f"[{dl:.2f}] {text}")

    print("\n⚠️ LEAST EFFICIENT (High DL):")
    for dl, text in sentences[-2:]:
        print(f"[{dl:.2f}] {text}")

analyze_sentence_efficiency("./data/processed/silver_emille/hin-s-dem-monologue1.conllu")