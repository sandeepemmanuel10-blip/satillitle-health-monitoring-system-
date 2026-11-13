
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from pyod.models.iforest import IForest

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "dataset.csv"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "mean", "var", "std", "len", "duration", "len_weighted",
    "gaps_squared", "n_peaks", "smooth10_n_peaks",
    "smooth20_n_peaks", "var_div_duration", "var_div_len",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
    "kurtosis", "skew",
]

df = pd.read_csv(DATA_PATH)

# Use the official split from the research notebook.
train_df = df[df["train"] == 1].copy()
test_df = df[df["train"] == 0].copy()

X_train = train_df[FEATURES]
y_train = train_df["anomaly"].astype(int)

X_test = test_df[FEATURES]
y_test = test_df["anomaly"].astype(int)

# Fit preprocessing on training data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Supervised anomaly classification.
ada = AdaBoostClassifier(random_state=2137)
ada.fit(X_train_scaled, y_train)

ada_pred = ada.predict(X_test_scaled)
ada_score = ada.predict_proba(X_test_scaled)[:, 1]

# 2. Unsupervised anomaly detection.
# Train on normal training segments only.
normal_mask = y_train == 0
X_train_normal = X_train_scaled[normal_mask]

iforest = IForest(
    random_state=2137,
    contamination=0.1,
)
iforest.fit(X_train_normal)

if_pred = iforest.predict(X_test_scaled)
if_score = iforest.decision_function(X_test_scaled)

# Save models and preprocessing.
joblib.dump(ada, MODELS_DIR / "adaboost_model.joblib")
joblib.dump(iforest, MODELS_DIR / "isolation_forest_model.joblib")
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

def report(name, y_true, predictions, scores):
    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.3f}")
    print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, predictions, zero_division=0):.3f}")
    print(f"F1 score:  {f1_score(y_true, predictions, zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, scores):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))

report("AdaBoost", y_test, ada_pred, ada_score)
report("Isolation Forest", y_test, if_pred, if_score)

# Save predictions for later dashboard use.
predictions_df = test_df[["segment", "channel", "anomaly"]].copy()
predictions_df = predictions_df.rename(columns={"anomaly": "actual_anomaly"})
predictions_df["adaboost_prediction"] = ada_pred
predictions_df["adaboost_score"] = ada_score
predictions_df["iforest_prediction"] = if_pred
predictions_df["iforest_score"] = if_score

predictions_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

print("\nSaved trained models to models/")
print("Saved test predictions to results/test_predictions.csv")
print("Training complete!")
