import os

EMILLE_BASE_DIR = "/Users/vatsalagrawal/Desktop/Mtech Thesis/W0037/annotated/hindi"

def diagnostic():
    files = sorted([f for f in os.listdir(EMILLE_BASE_DIR) if f.lower().endswith('.txt')])
    
    for fname in files[:10]: # Check first 10 files
        fpath = os.path.join(EMILLE_BASE_DIR, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(500)
            # Check for actual Devanagari characters (range \u0900-\u097F)
            has_hindi = any('\u0900' <= char <= '\u097f' for char in sample)
            status = "✅ HINDI DETECTED" if has_hindi else "❌ NO HINDI / ENCODING ERROR"
            print(f"{fname}: {status} | Sample: {sample[:50].strip()}...")

if __name__ == "__main__":
    diagnostic()