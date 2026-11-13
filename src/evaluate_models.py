
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
y_true = df["actual_anomaly"]

models = {
    "AdaBoost": (
        df["adaboost_prediction"],
        df["adaboost_score"],
    ),
    "Isolation Forest": (
        df["iforest_prediction"],
        df["iforest_score"],
    ),
}

metrics = []

for name, (pred, score) in models.items():
    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, score),
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

# Comparison chart
plot_df = metrics_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(11, 6))
sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model")
plt.ylim(0, 1)
plt.title("Satellite Anomaly Detection: Model Comparison")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (name, (pred, _)) in zip(axes, models.items()):
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, pred, labels=[0, 1]),
        display_labels=["Normal", "Anomaly"],
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
plt.close()

print("\nModel comparison:")
print(metrics_df.round(3).to_string(index=False))
print("\nSaved:")
print("results/model_metrics.csv")
print("results/model_comparison.png")
print("results/confusion_matrices.png")
