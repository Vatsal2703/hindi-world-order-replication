import os
import stanza # type: ignore
import re
import gc # Garbage Collector

def process_corpus():
    # 1. Load model once - use 'tokenize' pre-segmented to save RAM if needed
    # but 'tokenize,pos,lemma,depparse' is standard for your needs.
    nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma,depparse', use_gpu=False)
    
    input_dir = "/Users/vatsalagrawal/Desktop/Mtech Thesis/W0037/monoling/hindi/spoken"
    output_dir = "./data/processed/silver_emille"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    for i, filename in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing: {filename}")
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".txt", ".conllu"))

        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue

        try:
            with open(input_path, 'r', encoding='utf-16') as f:
                text = f.read()
        except:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        # Extract only the body
        if "<body>" in text: text = text.split("<body>")[-1]
        text = re.sub(r'<[^>]+>', '', text).strip()
        
        if not text: continue

        # PROCESS IN CHUNKS to avoid RAM spikes
        doc = nlp(text)
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for sent in doc.sentences:
                for word in sent.words:
                    line = [str(word.id), word.text, word.lemma or "_", word.upos or "_", 
                            word.xpos or "_", word.feats or "_", str(word.head) if word.head is not None else "0", 
                            word.deprel or "_", "_", "_"]
                    out_f.write("\t".join(line) + "\n")
                out_f.write("\n")

        # CRITICAL FOR 8GB RAM: Clear variables and collect garbage
        del doc
        gc.collect() 

    print("\n ALL FILES PARSED.")

if __name__ == "__main__":
    process_corpus()