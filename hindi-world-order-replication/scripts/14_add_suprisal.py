import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FEATURES = "./data/features/pairwise_features_trigram_blind.pkl"

def run_evaluation():
    print("\n" + "="*70)
    print(" EXPERIMENT 1B: EVALUATING DL + IS + TRIGRAM SURPRISAL")
    print("="*70)

    # 1. Load the new Trigram-enabled features
    try:
        with open(INPUT_FEATURES, 'rb') as f:
            df = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FEATURES} not found. Run workflow_trigram.py first.")
        return

    # 2. Define Features (X) and Label (y)
    # Note: We now have 'surprisal_diff' in the mix
    features = ['dep_len_diff', 'info_status_diff', 'surprisal_diff']
    X = df[features]
    y = df['label']

    # 3. Standardize Features
    # Since Surprisal values are much larger than DL values, 
    # scaling is CRITICAL for the coefficients to be comparable.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 5. Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 6. Results
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nMODEL ACCURACY: {accuracy:.2%}")
    print("\n" + "-"*30)
    print("Feature Weights (Normalized):")
    for feat, coef in zip(features, model.coef_[0]):
        print(f" - {feat:18}: {coef:.4f}")
    print("-"*30)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Predicted Variant (0)', 'Predicted Human (1)']))

if __name__ == "__main__":
    run_evaluation()