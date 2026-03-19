#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("\n" + "="*70)
print(" EXPERIMENT 1: BASELINE LOGISTIC REGRESSION (DL + IS)")
print("="*70 + "\n")

# 1. Load the Data
INPUT_FILE = "./data/features/pairwise_features.csv"
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found!")
    exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} pairwise comparisons.")

# 2. Define Features (X) and Target Label (y)
# We are using the "Deltas" (Differences) calculated via Joachims' method
X = df[['dep_len_diff', 'info_status_diff']]
y = df['label'] # 1 means Sentence A is human, 0 means Sentence B is human

# 3. Train/Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train):,} pairs, Testing on {len(X_test):,} pairs...")

# 4. Train the Model
clf = LogisticRegression(random_state=42, solver='lbfgs')
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print(f"BASELINE MODEL ACCURACY: {accuracy * 100:.2f}%")
print("="*70)

# 6. Analyze Feature Importance (Coefficients)
# A negative coefficient for DL means the model learned that 
# SHORTER dependency lengths are more likely to be the human choice.
print("\nFeature Weights (Coefficients):")
for feature, coef in zip(X.columns, clf.coef_[0]):
    print(f" - {feature}: {coef:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Predicted Variant (0)', 'Predicted Human (1)']))