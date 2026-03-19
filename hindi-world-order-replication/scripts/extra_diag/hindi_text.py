import os

EMILLE_DIR = "/Users/vatsalagrawal/Desktop/Mtech Thesis/W0037/monoling/hindi/spoken"

def find_true_hindi():
    files = [f for f in os.listdir(EMILLE_DIR) if f.endswith('.txt')]
    print(f"Total files to check: {len(files)}")
    
    hindi_files = []
    for fname in files:
        fpath = os.path.join(EMILLE_DIR, fname)
        # Try UTF-16 then UTF-8
        try:
            with open(fpath, 'r', encoding='utf-16') as f:
                content = f.read(10000) # Check first 10k chars
        except:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)
        
        # Count Devanagari characters
        hindi_chars = len([c for c in content if '\u0900' <= c <= '\u097f'])
        
        if hindi_chars > 100: # Threshold: at least 100 Hindi characters
            hindi_files.append((fname, hindi_chars))

    print("\n✅ Found these files with high Devanagari density:")
    for name, count in sorted(hindi_files, key=lambda x: x[1], reverse=True):
        print(f"- {name} ({count} Hindi chars)")

if __name__ == "__main__":
    find_true_hindi()