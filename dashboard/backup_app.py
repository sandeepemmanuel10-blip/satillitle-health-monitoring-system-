
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Satellite Health Monitoring System",
    page_icon="🛰️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    "duration",
    "len",
    "mean",
    "var",
    "std",
    "kurtosis",
    "skew",
    "n_peaks",
    "smooth10_n_peaks",
    "smooth20_n_peaks",
    "diff_peaks",
    "diff2_peaks",
    "diff_var",
    "diff2_var",
    "gaps_squared",
    "len_weighted",
    "var_div_duration",
    "var_div_len"
]


# =========================================================
# 2. LOAD DATA
# =========================================================

@st.cache_data
def load_predictions():

    path = RESULTS_DIR / "test_predictions.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_metrics():

    path = RESULTS_DIR / "model_metrics.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_dataset():

    path = DATA_DIR / "raw" / "dataset.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_resource
def load_models():

    scaler = joblib.load(
        MODELS_DIR / "scaler.joblib"
    )

    adaboost = joblib.load(
        MODELS_DIR / "adaboost_model.joblib"
    )

    iforest = joblib.load(
        MODELS_DIR / "isolation_forest_model.joblib"
    )

    return scaler, adaboost, iforest


predictions = load_predictions()
metrics = load_metrics()
dataset = load_dataset()


# =========================================================
# 3. HEADER
# =========================================================

st.title("🛰️ Satellite Health Monitoring System")

st.subheader(
    "Machine Learning-Based Satellite Telemetry Anomaly Detection"
)

st.write(
    "Monitor satellite telemetry, detect unusual behaviour, "
    "compare machine learning models and test new telemetry."
)

st.divider()


# =========================================================
# 4. SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Both Models", "AdaBoost", "Isolation Forest"]
)

anomaly_filter = st.sidebar.selectbox(
    "Filter Records",
    ["All Records", "Anomalies Only", "Normal Only"]
)


# =========================================================
# 5. CHECK EXISTING PREDICTIONS
# =========================================================

if predictions.empty:

    st.error(
        "Prediction data not found. "
        "Check results/test_predictions.csv"
    )

    st.stop()


required_columns = [
    "actual_anomaly",
    "adaboost_prediction",
    "iforest_prediction"
]

missing_columns = [
    col for col in required_columns
    if col not in predictions.columns
]

if missing_columns:

    st.error(
        f"Missing prediction columns: {missing_columns}"
    )

    st.stop()


# =========================================================
# 6. SATELLITE HEALTH OVERVIEW
# =========================================================

st.header("📊 Satellite Health Overview")

total_segments = len(predictions)

actual_anomalies = int(
    predictions["actual_anomaly"].sum()
)

actual_normal = total_segments - actual_anomalies

anomaly_rate = (
    actual_anomalies / total_segments * 100
    if total_segments else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Test Segments", f"{total_segments:,}")
c2.metric("Actual Anomalies", f"{actual_anomalies:,}")
c3.metric("Actual Normal Segments", f"{actual_normal:,}")
c4.metric("Actual Anomaly Rate", f"{anomaly_rate:.1f}%")

st.divider()


# =========================================================
# 7. MODEL DETECTION SUMMARY
# =========================================================

st.header("🤖 Model Detection Summary")

ada_detected = int(
    predictions["adaboost_prediction"].sum()
)

if_detected = int(
    predictions["iforest_prediction"].sum()
)

c1, c2 = st.columns(2)

with c1:

    st.subheader("AdaBoost")

    st.metric("Predicted Anomalies", ada_detected)

    st.metric(
        "Predicted Normal",
        total_segments - ada_detected
    )

with c2:

    st.subheader("Isolation Forest")

    st.metric("Predicted Anomalies", if_detected)

    st.metric(
        "Predicted Normal",
        total_segments - if_detected
    )

st.divider()


# =========================================================
# 8. MODEL PERFORMANCE
# =========================================================

st.header("📈 Model Performance Comparison")

if not metrics.empty:

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = [
        col for col in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
        if col in metrics.columns
    ]

    if numeric_columns:

        chart_data = metrics.set_index(
            metrics.columns[0]
        )[numeric_columns]

        fig, ax = plt.subplots(figsize=(10, 5))

        chart_data.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Machine Learning Model Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Model")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        plt.xticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

else:

    st.warning("Model metrics file not found.")

st.divider()


# =========================================================
# 9. ANOMALY DISTRIBUTION
# =========================================================

st.header("🔍 Anomaly Distribution")

left, right = st.columns(2)

with left:

    st.subheader("Actual Distribution")

    distribution = pd.DataFrame({
        "Count": [actual_anomalies, actual_normal]
    }, index=["Anomaly", "Normal"])

    st.bar_chart(distribution)

with right:

    st.subheader("Model Predictions")

    model_counts = pd.DataFrame({
        "Predicted Anomalies": [
            ada_detected,
            if_detected
        ]
    }, index=["AdaBoost", "Isolation Forest"])

    st.bar_chart(model_counts)

st.divider()


# =========================================================
# 10. CHANNEL ANALYSIS
# =========================================================

st.header("📡 Channel-Level Analysis")

if "channel" in predictions.columns:

    channel_column = st.selectbox(
        "Choose what to analyse",
        [
            "actual_anomaly",
            "adaboost_prediction",
            "iforest_prediction"
        ],
        format_func=lambda x: {
            "actual_anomaly": "Actual Anomalies",
            "adaboost_prediction": "AdaBoost Predictions",
            "iforest_prediction": "Isolation Forest Predictions"
        }[x]
    )

    channel_data = (
        predictions.groupby("channel")[channel_column]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(channel_data)

else:

    st.info("Channel information unavailable.")

st.divider()


# =========================================================
# 11. EXISTING VISUALISATIONS
# =========================================================

st.header("📉 Dataset Visualisations")

graph_files = [
    ("Anomalies by Channel", "anomalies_by_channel.png"),
    ("Anomaly Distribution", "anomaly_distribution.png"),
    ("Feature Distributions", "feature_distributions.png"),
    ("Train/Test Split Distribution", "split_distribution.png"),
    ("Model Comparison", "model_comparison.png"),
    ("Confusion Matrices", "confusion_matrices.png")
]

for title, filename in graph_files:

    image_path = RESULTS_DIR / filename

    if image_path.exists():

        with st.expander(title):

            st.image(
                str(image_path),
                caption=title,
                use_container_width=True
            )

st.divider()


# =========================================================
# 12. EXISTING PREDICTION RECORDS
# =========================================================

st.header("📋 Existing Anomaly Records")

filtered = predictions.copy()

if anomaly_filter == "Anomalies Only":

    filtered = filtered[
        filtered["actual_anomaly"] == 1
    ]

elif anomaly_filter == "Normal Only":

    filtered = filtered[
        filtered["actual_anomaly"] == 0
    ]


if model_choice == "AdaBoost":

    display_columns = [
        col for col in [
            "segment",
            "channel",
            "actual_anomaly",
            "adaboost_prediction",
            "adaboost_score"
        ]
        if col in filtered.columns
    ]

elif model_choice == "Isolation Forest":

    display_columns = [
        col for col in [
            "segment",
            "channel",
            "actual_anomaly",
            "iforest_prediction",
            "iforest_score"
        ]
        if col in filtered.columns
    ]

else:

    display_columns = filtered.columns.tolist()


st.write(f"Showing {len(filtered):,} records")

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Existing Predictions CSV",
    data=filtered.to_csv(index=False),
    file_name="satellite_predictions.csv",
    mime="text/csv"
)

st.divider()


# =========================================================
# 13. NEW TELEMETRY ANOMALY DETECTION
# =========================================================

st.header("🛰️ Test New Satellite Telemetry")

st.write(
    "Enter new telemetry feature values or upload a CSV "
    "to detect anomalies using the trained machine learning models."
)

st.info(
    "This feature uses the saved models. "
    "It does not retrain them."
)

try:

    scaler, adaboost, iforest = load_models()

    # Get the exact order of features used during scaler training.

    if hasattr(scaler, "feature_names_in_"):

        scaler_features = list(
            scaler.feature_names_in_
        )

    else:

        scaler_features = FEATURES.copy()

    if set(scaler_features) != set(FEATURES):

        st.error(
            "The saved scaler does not match the expected "
            "18 telemetry features. Please check model training."
        )

        st.stop()

    input_method = st.radio(
        "Choose your input method",
        ["Manual Entry", "Upload CSV"],
        horizontal=True
    )

    input_df = None

    # -----------------------------------------------------
    # MANUAL INPUT
    # -----------------------------------------------------

    if input_method == "Manual Entry":

        st.subheader("Enter Telemetry Feature Values")

        if not dataset.empty:

            defaults = dataset[
                FEATURES
            ].median(numeric_only=True)

        else:

            defaults = pd.Series(
                0.0,
                index=FEATURES
            )

        with st.form("telemetry_form"):

            values = {}

            columns = st.columns(3)

            for i, feature in enumerate(FEATURES):

                with columns[i % 3]:

                    values[feature] = st.number_input(
                        feature,
                        value=float(defaults[feature]),
                        format="%.8f"
                    )

            submitted = st.form_submit_button(
                "🔍 Detect Anomaly",
                use_container_width=True
            )

        if submitted:

            input_df = pd.DataFrame([values])

    # -----------------------------------------------------
    # CSV UPLOAD
    # -----------------------------------------------------

    else:

        st.subheader("Upload Telemetry CSV")

        uploaded_file = st.file_uploader(
            "Upload a CSV with all 18 feature columns",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                uploaded_df = pd.read_csv(uploaded_file)

                missing = [
                    col for col in FEATURES
                    if col not in uploaded_df.columns
                ]

                if missing:

                    st.error(
                        "Missing features: "
                        + ", ".join(missing)
                    )

                elif uploaded_df.empty:

                    st.warning("Uploaded CSV is empty.")

                else:

                    st.success(
                        f"Loaded {len(uploaded_df)} telemetry records."
                    )

                    st.dataframe(
                        uploaded_df.head(),
                        use_container_width=True
                    )

                    if st.button(
                        "🔍 Detect Anomalies",
                        use_container_width=True
                    ):

                        input_df = uploaded_df[
                            FEATURES
                        ].copy()

            except Exception as error:

                st.error(
                    f"CSV reading error: {error}"
                )

    # -----------------------------------------------------
    # RUN PREDICTIONS
    # -----------------------------------------------------

    if input_df is not None:

        if input_df.empty:

            st.warning("No telemetry records provided.")

        else:

            input_df = input_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            if input_df.isnull().any().any():

                st.error(
                    "Invalid or missing feature values. "
                    "Please check your inputs."
                )

            elif not input_df.map(
                lambda x: pd.notna(x) and abs(x) != float("inf")
            ).all().all():

                st.error(
                    "Infinite or invalid values detected."
                )

            else:

                # CRITICAL FIX:
                # Reorder columns to exactly match scaler training order.

                ordered_input = input_df[
                    scaler_features
                ].copy()

                # Apply saved scaler.

                scaled_input = scaler.transform(
                    ordered_input
                )

                # AdaBoost prediction.

                ada_predictions = adaboost.predict(
                    scaled_input
                )

                ada_probabilities = adaboost.predict_proba(
                    scaled_input
                )

                anomaly_class_index = list(
                    adaboost.classes_
                ).index(1)

                ada_scores = ada_probabilities[
                    :, anomaly_class_index
                ]

                # Isolation Forest prediction.

                if_predictions = iforest.predict(
                    scaled_input
                )

                if_scores = iforest.decision_function(
                    scaled_input
                )

                # Create result table.

                results = input_df.copy()

                results.insert(
                    0,
                    "Telemetry Segment",
                    range(1, len(input_df) + 1)
                )

                results["AdaBoost Prediction"] = [
                    "ANOMALY" if p == 1 else "NORMAL"
                    for p in ada_predictions
                ]

                results["AdaBoost Anomaly Probability"] = (
                    ada_scores
                )

                results["Isolation Forest Prediction"] = [
                    "ANOMALY" if p == 1 else "NORMAL"
                    for p in if_predictions
                ]

                results["Isolation Forest Decision Score"] = (
                    if_scores
                )

                # -------------------------------------------------
                # DISPLAY RESULTS
                # -------------------------------------------------

                st.divider()

                st.subheader("🎯 Machine Learning Results")

                total_input = len(results)

                ada_total = int(
                    (ada_predictions == 1).sum()
                )

                if_total = int(
                    (if_predictions == 1).sum()
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Segments Tested",
                    total_input
                )

                c2.metric(
                    "AdaBoost Anomalies",
                    ada_total
                )

                c3.metric(
                    "Isolation Forest Anomalies",
                    if_total
                )

                # Display individual results.

                for i in range(len(results)):

                    row = results.iloc[i]

                    with st.expander(
                        f"Telemetry Segment {i + 1}",
                        expanded=(total_input == 1)
                    ):

                        left, right = st.columns(2)

                        with left:

                            st.markdown("### AdaBoost")

                            if row["AdaBoost Prediction"] == "ANOMALY":

                                st.error("🚨 ANOMALY DETECTED")

                            else:

                                st.success("✅ NORMAL")

                            st.metric(
                                "Anomaly Probability",
                                f"{row['AdaBoost Anomaly Probability']:.2%}"
                            )

                        with right:

                            st.markdown("### Isolation Forest")

                            if row["Isolation Forest Prediction"] == "ANOMALY":

                                st.error("🚨 ANOMALY DETECTED")

                            else:

                                st.success("✅ NORMAL")

                            st.metric(
                                "Decision Score",
                                f"{row['Isolation Forest Decision Score']:.4f}"
                            )

                        if (
                            row["AdaBoost Prediction"] == "ANOMALY"
                            and row["Isolation Forest Prediction"] == "ANOMALY"
                        ):

                            st.error(
                                "Both models detected an anomaly."
                            )

                        elif (
                            row["AdaBoost Prediction"]
                            != row["Isolation Forest Prediction"]
                        ):

                            st.warning(
                                "The two models disagree. "
                                "Further investigation is recommended."
                            )

                        else:

                            st.success(
                                "Both models agree on this prediction."
                            )

                st.subheader("Detailed Prediction Table")

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download New Prediction Results",
                    data=results.to_csv(index=False),
                    file_name="new_telemetry_predictions.csv",
                    mime="text/csv"
                )

except FileNotFoundError as error:

    st.error(
        f"Required model file not found: {error}"
    )

except Exception as error:

    st.error(
        f"Prediction system error: {error}"
    )


# =========================================================
# 14. DATASET INFORMATION
# =========================================================

st.divider()

st.header("📁 Dataset Information")

if not dataset.empty:

    c1, c2 = st.columns(2)

    c1.metric(
        "Dataset Rows",
        f"{len(dataset):,}"
    )

    c2.metric(
        "Dataset Columns",
        len(dataset.columns)
    )

    with st.expander("View Original Dataset"):

        st.dataframe(
            dataset.head(100),
            use_container_width=True
        )

else:

    st.info("Original dataset not found.")


# =========================================================
# 15. FOOTER
# =========================================================

st.divider()

st.caption(
    "Satellite Health Monitoring System | "
    "OPSSAT-AD Dataset | Machine Learning Project"
)
