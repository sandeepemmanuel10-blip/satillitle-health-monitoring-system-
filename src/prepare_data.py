
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "dataset.csv")

# Identify metadata, labels and candidate predictors.
metadata = ["segment", "channel", "sampling", "train", "anomaly"]

candidate_features = [
    column for column in df.select_dtypes(include="number").columns
    if column not in metadata
]

print("DATASET PREPARATION REPORT")
print("=" * 50)

print(f"Rows: {len(df)}")
print(f"Candidate numeric features: {len(candidate_features)}")
print("\nCandidate features:")
print(candidate_features)

print("\nSplit and label counts:")
print(pd.crosstab(df["train"], df["anomaly"], margins=True))

print("\nLabel counts by channel:")
print(pd.crosstab(df["channel"], df["anomaly"]))

# Save a report of feature missingness and ranges.
summary = df[candidate_features].describe().T
summary["missing"] = df[candidate_features].isna().sum()
summary.to_csv(RESULTS_DIR / "feature_summary.csv")

print("\nSaved feature summary to results/feature_summary.csv")
print("No original dataset files were modified.")
