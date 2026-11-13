
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "dataset.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# 1. Distribution of anomaly labels
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="anomaly")
plt.title("OPSSAT-AD: Anomaly Label Distribution")
plt.xlabel("Anomaly label (0 = normal, 1 = anomaly)")
plt.ylabel("Number of segments")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "anomaly_distribution.png", dpi=150)
plt.close()

# 2. Distribution of the provided split indicator
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="train")
plt.title("OPSSAT-AD: Provided Split Indicator")
plt.xlabel("train value")
plt.ylabel("Number of segments")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "split_distribution.png", dpi=150)
plt.close()

# 3. Anomaly counts by channel
channel_counts = (
    df.groupby(["channel", "anomaly"])
    .size()
    .unstack(fill_value=0)
)

channel_counts.plot(kind="bar", figsize=(11, 6))
plt.title("Anomaly Labels by Telemetry Channel")
plt.xlabel("Channel")
plt.ylabel("Number of segments")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Anomaly label")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "anomalies_by_channel.png", dpi=150)
plt.close()

# 4. Compare selected feature distributions
features = ["mean", "std", "kurtosis", "n_peaks"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for feature, ax in zip(features, axes.flat):
    sns.boxplot(data=df, x="anomaly", y=feature, ax=ax)
    ax.set_title(f"{feature} by anomaly label")
    ax.set_xlabel("Anomaly label")

fig.suptitle("Telemetry Feature Distributions", fontsize=15)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_distributions.png", dpi=150)
plt.close(fig)

print("Visualization complete!")
print(f"Graphs saved in: {RESULTS_DIR}")

for path in sorted(RESULTS_DIR.glob("*.png")):
    print(f"- {path.name}")
