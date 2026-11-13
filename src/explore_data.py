
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

dataset_path = DATA_DIR / "dataset.csv"
segments_path = DATA_DIR / "segments.csv"

# Load the feature dataset
df = pd.read_csv(dataset_path)

print("=" * 60)
print("OPSSAT-AD DATA EXPLORATION")
print("=" * 60)

print(f"\nFeature dataset shape: {df.shape}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Total missing values: {df.isna().sum().sum()}")

print("\n--- Column data types ---")
print(df.dtypes.to_string())

print("\n--- Anomaly label counts ---")
print(df["anomaly"].value_counts(dropna=False).sort_index())

print("\n--- Train split counts ---")
print(df["train"].value_counts(dropna=False).sort_index())

print("\n--- Unique channels ---")
print(df["channel"].nunique())
print(df["channel"].value_counts().head(10).to_string())

print("\n--- Rows per anomaly label and split ---")
print(pd.crosstab(df["train"], df["anomaly"]))

print("\n--- Numeric feature summary ---")
print(df.select_dtypes(include="number").describe().T.to_string())

print("\n--- Segment dataset preview ---")
segments_preview = pd.read_csv(segments_path, nrows=5)
print(segments_preview.to_string(index=False))

print("\nExploration complete.")
