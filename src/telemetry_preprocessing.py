import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


FEATURES = [
    "mean",
    "var",
    "std",
    "kurtosis",
    "skew",
    "n_peaks",
    "diff_var",
    "duration",
]


def extract_features(values, timestamps):
    """
    Extract the eight features used by the trained models.

    Feature calculations follow the definitions validated against
    all 2,123 OPSSAT-AD dataset segments.
    """

    values = np.asarray(values, dtype=float)
    timestamps = pd.to_datetime(timestamps, utc=True)

    if len(values) < 8:
        raise ValueError(
            "At least 8 telemetry readings are required per segment."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Telemetry values contain missing or infinite numbers."
        )

    if len(timestamps) != len(values):
        raise ValueError(
            "The number of timestamps and values must be equal."
        )

    if timestamps.isna().any():
        raise ValueError("Invalid timestamps were found.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Timestamps must be sorted in ascending order."
        )

    signal_range = np.max(values) - np.min(values)
    prominence = 0.10 * signal_range

    peaks, _ = find_peaks(
        values,
        prominence=prominence
    )

    differences = np.diff(values)

    duration = (
        timestamps.max() - timestamps.min()
    ).total_seconds()

    return {
        "mean": float(np.mean(values)),
        "var": float(np.var(values, ddof=0)),
        "std": float(np.std(values, ddof=0)),
        "kurtosis": float(kurtosis(values)),
        "skew": float(skew(values)),
        "n_peaks": int(len(peaks)),
        "diff_var": float(np.var(differences, ddof=0)),
        "duration": float(duration),
    }


def preprocess_raw_telemetry(
    dataframe,
    window_size=None
):
    """
    Convert raw timestamped telemetry into model-ready features.

    Required columns:
        timestamp
        value

    Optional columns:
        channel
        segment

    If segment exists, it is used to preserve supplied grouping.

    If segment does not exist, window_size must be specified.
    Windows are formed separately for each channel.
    """

    df = dataframe.copy()

    required = {"timestamp", "value"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if "channel" not in df.columns:
        df["channel"] = "UNKNOWN"

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamps are missing or invalid."
        )

    if df["value"].isna().any():
        raise ValueError(
            "Some telemetry values are missing or non-numeric."
        )

    if not np.isfinite(df["value"].to_numpy(dtype=float)).all():
        raise ValueError(
            "Telemetry contains infinite values."
        )

    if df["channel"].isna().any():
        raise ValueError(
            "Channel names cannot be empty."
        )

    df = df.sort_values(
        ["channel", "timestamp"]
    ).reset_index(drop=True)

    if "segment" in df.columns:
        grouping = ["channel", "segment"]
        df["window_id"] = df["segment"].astype(str)

    else:
        if window_size is None:
            raise ValueError(
                "No segment column was provided. "
                "Specify a window_size to divide the telemetry."
            )

        if not isinstance(window_size, int) or window_size < 8:
            raise ValueError(
                "window_size must be an integer of at least 8."
            )

        df["window_number"] = (
            df.groupby("channel").cumcount() // window_size
        )

        df["window_id"] = (
            df["channel"].astype(str)
            + "_"
            + df["window_number"].astype(str)
        )

        grouping = ["channel", "window_id"]

    results = []

    for group_key, group in df.groupby(
        grouping,
        sort=False
    ):
        group = group.sort_values("timestamp")

        if len(group) < 8:
            continue

        features = extract_features(
            group["value"].to_numpy(),
            group["timestamp"]
        )

        if isinstance(group_key, tuple):
            identifier = str(group_key[-1])
        else:
            identifier = str(group_key)

        results.append({
            "channel": str(group["channel"].iloc[0]),
            "window_id": identifier,
            "start_time": group["timestamp"].min(),
            "end_time": group["timestamp"].max(),
            "readings": len(group),
            **features,
        })

    if not results:
        raise ValueError(
            "No valid telemetry windows were produced. "
            "Each window needs at least 8 readings."
        )

    result = pd.DataFrame(results)

    return result
