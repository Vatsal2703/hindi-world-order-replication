#!/usr/bin/env python3
"""
Script 17: Add Emotion Features (with Pandas version compatibility)

Handles the StringDtype pickle compatibility issue between Pandas versions
without requiring an upgrade. Uses a safe unpickler shim.
"""

import sys
import os
import pickle
import io

# ============================================================================
# PANDAS COMPATIBILITY SHIM (handles StringDtype mismatch)
# ============================================================================

class CompatUnpickler(pickle.Unpickler):
    
    def find_class(self, module, name):
        # Intercept the new StringDtype that older Pandas can't handle
        if module == "pandas.core.arrays.string_" and name == "StringDtype":
            return self._string_dtype_stub
        if module == "pandas._libs.arrays" and name == "NDArrayBacked":
            return self._ndarray_backed_stub
        return super().find_class(module, name)

    @staticmethod
    def _string_dtype_stub(*args, **kwargs):
        """Return a plain Python object — works as a dtype placeholder."""
        return object

    @staticmethod
    def _ndarray_backed_stub(*args, **kwargs):
        return None


def safe_load_pickle(path):
    """
    Load a pickle file with StringDtype compatibility.
    Falls back to standard pd.read_pickle if the shim isn't needed.
    """
    import pandas as pd
    import numpy as np

    # Try standard load first
    try:
        return pd.read_pickle(path)
    except Exception:
        pass

    # Try with compatibility shim
    try:
        with open(path, 'rb') as f:
            raw = CompatUnpickler(f).load()
        if isinstance(raw, pd.DataFrame):
            return raw
        # If it came out as a dict or something else, convert
        return pd.DataFrame(raw)
    except Exception:
        pass

    # Last resort: use pandas pickle_compat directly with a patched module
    try:
        import pandas.compat.pickle_compat as pc

        class PatchedUnpickler(pc.Unpickler):
            def find_class(self, module, name):
                if "StringDtype" in name:
                    return lambda *a, **kw: object
                return super().find_class(module, name)

        with open(path, 'rb') as f:
            return PatchedUnpickler(f).load()
    except Exception as e:
        print(f"ERROR: Could not load {path}")
        print(f"  Reason: {e}")
        print("\nFix: Run this command and try again:")
        print("  conda install -c conda-forge 'pandas>=2.2' 'numpy>=2.0' -y")
        sys.exit(1)

# ============================================================================
# DYNAMIC PATH RESOLUTION
# ============================================================================

def find_workspace_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = script_dir

    while repo_root and repo_root != os.path.dirname(repo_root):
        if os.path.basename(repo_root) == "hindi-world-order-replication":
            base = repo_root
            return (os.path.join(base, "data", "features", "all_features_final.pkl"),
                    os.path.join(base, "data", "processed", "preceding_emotions.pkl"),
                    os.path.join(base, "data", "features", "all_features_with_emotion.pkl"))
        if os.path.isdir(os.path.join(repo_root, "hindi-world-order-replication")):
            base = os.path.join(repo_root, "hindi-world-order-replication")
            return (os.path.join(base, "data", "features", "all_features_final.pkl"),
                    os.path.join(base, "data", "processed", "preceding_emotions.pkl"),
                    os.path.join(base, "data", "features", "all_features_with_emotion.pkl"))
        repo_root = os.path.dirname(repo_root)

    return ("./data/features/all_features_final.pkl",
            "./data/processed/preceding_emotions.pkl",
            "./data/features/all_features_with_emotion.pkl")

FEATURES_IN, EMOTIONS_IN, FEATURES_OUT = find_workspace_paths()

# ============================================================================
# CONFIGURATION
# ============================================================================

ADAPTIVE_COL_CANDIDATES = [
    "adaptive_surprisal_diff",
    "adaptive_lstm_diff",
    "adaptive_surprisal",
    "adaptive_lstm_surprisal_diff",
]

NEUTRAL_VALENCE = 0.0
NEUTRAL_AROUSAL = 0.5


def find_adaptive_column(df):
    for c in ADAPTIVE_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "adaptive" in c.lower():
            return c
    return None


def main():
    import pandas as pd

    print("\n" + "=" * 70)
    print(" SCRIPT 17: ADD EMOTION FEATURES")
    print("=" * 70 + "\n")

    print(f"Target Features Input:  {FEATURES_IN}")
    print(f"Target Emotions Input:  {EMOTIONS_IN}")
    print(f"Target Output Path:     {FEATURES_OUT}\n")

    # --- Load existing features ---
    print("Loading MTP-I features...")
    if not os.path.exists(FEATURES_IN):
        print(f"ERROR: {FEATURES_IN} not found.")
        sys.exit(1)

    df = safe_load_pickle(FEATURES_IN)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # --- Load emotion annotations ---
    print("\nLoading emotion annotations...")
    if not os.path.exists(EMOTIONS_IN):
        print(f"ERROR: {EMOTIONS_IN} not found. Run Script 16b first.")
        sys.exit(1)
    with open(EMOTIONS_IN, 'rb') as f:
        emotions = pickle.load(f)
    print(f"  Loaded emotion scores for {len(emotions):,} sentences")

    if "sent_id" not in df.columns:
        print("ERROR: 'sent_id' column not found.")
        sys.exit(1)

    # --- Map valence and arousal onto each row by sent_id ---
    print("\nMerging emotion features...")
    df["prev_valence"] = df["sent_id"].map(
        lambda s: emotions.get(s, {}).get("valence", NEUTRAL_VALENCE))
    df["prev_arousal"] = df["sent_id"].map(
        lambda s: emotions.get(s, {}).get("arousal", NEUTRAL_AROUSAL))

    # --- Find adaptive LSTM column for interaction terms ---
    adaptive_col = find_adaptive_column(df)
    if adaptive_col is None:
        print("WARNING: adaptive LSTM column not found. Skipping interaction terms.")
    else:
        print(f"  Using '{adaptive_col}' for interaction terms")
        df["valence_x_adaptive"] = pd.to_numeric(df["prev_valence"], errors="coerce") * \
                                   pd.to_numeric(df[adaptive_col], errors="coerce")
        df["arousal_x_adaptive"] = pd.to_numeric(df["prev_arousal"], errors="coerce") * \
                                   pd.to_numeric(df[adaptive_col], errors="coerce")

    # --- Diagnostics ---
    n_matched = df["sent_id"].map(lambda s: s in emotions).sum()
    print(f"\n  Rows matched to an emotion score: {n_matched:,} / {len(df):,}")
    print(f"  prev_valence range: [{df['prev_valence'].min():.3f}, {df['prev_valence'].max():.3f}]")
    print(f"  prev_arousal range: [{df['prev_arousal'].min():.3f}, {df['prev_arousal'].max():.3f}]")

    # --- Save ---
    os.makedirs(os.path.dirname(FEATURES_OUT), exist_ok=True)
    df.to_pickle(FEATURES_OUT)

    new_cols = ["prev_valence", "prev_arousal"]
    if adaptive_col is not None:
        new_cols += ["valence_x_adaptive", "arousal_x_adaptive"]

    print("\n" + "=" * 70)
    print(f"  Added {len(new_cols)} new columns: {new_cols}")
    print(f"  Saved -> {FEATURES_OUT}")
    print("  Next: Run Script 18 (classify_with_emotion.py)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()