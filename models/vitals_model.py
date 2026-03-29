"""
Vitals Anomaly Detection
Uses rolling z-score and clinical reference ranges.
No heavy dependencies — pure NumPy/Pandas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ── Clinical Reference Ranges ──────────────────────────────────────────────────
VITAL_RANGES = {
    "Heart Rate (bpm)": {
        "low_critical": 40, "low_warning": 50,
        "high_warning": 100, "high_critical": 130,
        "unit": "bpm",
    },
    "Systolic BP (mmHg)": {
        "low_critical": 80, "low_warning": 90,
        "high_warning": 160, "high_critical": 190,
        "unit": "mmHg",
    },
    "SpO2 (%)": {
        "low_critical": 88, "low_warning": 92,
        "high_warning": 101, "high_critical": 102,  # max = 100
        "unit": "%",
    },
    "Temperature (°C)": {
        "low_critical": 35.0, "low_warning": 36.0,
        "high_warning": 38.0, "high_critical": 39.5,
        "unit": "°C",
    },
}

VITAL_DEFAULTS = {
    "Heart Rate (bpm)":    {"mean": 78, "std": 8},
    "Systolic BP (mmHg)":  {"mean": 122, "std": 10},
    "SpO2 (%)":            {"mean": 97, "std": 1.5},
    "Temperature (°C)":    {"mean": 37.0, "std": 0.3},
}


def classify_vital(value: float, ranges: dict) -> str:
    """Return severity label for a single vital reading."""
    if value <= ranges["low_critical"] or value >= ranges["high_critical"]:
        return "CRITICAL"
    if value <= ranges["low_warning"] or value >= ranges["high_warning"]:
        return "WARNING"
    return "NORMAL"


def rolling_zscore(series: pd.Series, window: int = 5) -> pd.Series:
    """Compute rolling z-score; NaN for the first <window> points."""
    roll_mean = series.rolling(window, min_periods=2).mean()
    roll_std  = series.rolling(window, min_periods=2).std().replace(0, np.nan)
    return (series - roll_mean) / roll_std


def detect_anomalies(df: pd.DataFrame, zscore_thresh: float = 2.0) -> Dict:
    """
    Detect anomalies across all vitals in the dataframe.

    Args:
        df: DataFrame with columns matching VITAL_RANGES keys + a 'Time' column.
        zscore_thresh: z-score threshold above which a reading is flagged.

    Returns:
        {
            vital_name: {
                "status":      overall "NORMAL" | "WARNING" | "CRITICAL",
                "anomaly_idx": list of integer indices,
                "zscore":      pd.Series of z-scores,
                "severity":    pd.Series of per-row severity labels,
                "stats":       {mean, std, min, max, anomaly_pct},
            }
        }
    """
    results = {}

    for vital, ranges in VITAL_RANGES.items():
        if vital not in df.columns:
            continue

        series   = df[vital].astype(float)
        zscores  = rolling_zscore(series)
        severity = series.apply(lambda v: classify_vital(v, ranges))

        # Flag anomalies: either z-score breach OR clinical threshold breach
        z_anomaly   = zscores.abs() > zscore_thresh
        clin_anomaly = severity.isin(["WARNING", "CRITICAL"])
        anomaly_mask = z_anomaly | clin_anomaly

        anomaly_idx = anomaly_mask[anomaly_mask].index.tolist()

        # Overall status = worst severity seen
        if "CRITICAL" in severity.values:
            overall = "CRITICAL"
        elif "WARNING" in severity.values:
            overall = "WARNING"
        else:
            overall = "NORMAL"

        results[vital] = {
            "status":      overall,
            "anomaly_idx": anomaly_idx,
            "zscore":      zscores,
            "severity":    severity,
            "stats": {
                "mean":        round(float(series.mean()), 2),
                "std":         round(float(series.std()), 2),
                "min":         round(float(series.min()), 2),
                "max":         round(float(series.max()), 2),
                "anomaly_pct": round(100 * anomaly_mask.sum() / len(series), 1),
            },
        }

    return results


def generate_sample_vitals(scenario: str = "stable", n_points: int = 24) -> pd.DataFrame:
    """Generate synthetic time-series vitals for demo."""
    rng = np.random.default_rng(99)
    times = pd.date_range("2024-01-01 00:00", periods=n_points, freq="1h")

    if scenario == "stable":
        hr   = rng.normal(76, 5, n_points).clip(60, 95)
        bp   = rng.normal(122, 6, n_points).clip(105, 140)
        spo2 = rng.normal(97.5, 0.8, n_points).clip(95, 100)
        temp = rng.normal(37.0, 0.2, n_points).clip(36.4, 37.8)

    elif scenario == "deteriorating":
        t = np.linspace(0, 1, n_points)
        hr   = 75 + 55 * t + rng.normal(0, 4, n_points)
        bp   = 125 - 45 * t + rng.normal(0, 5, n_points)
        spo2 = 98 - 12 * t + rng.normal(0, 0.8, n_points)
        temp = 37.0 + 2.8 * t + rng.normal(0, 0.2, n_points)

    elif scenario == "post_surgery":
        hr   = rng.normal(88, 10, n_points).clip(60, 130)
        bp   = rng.normal(110, 12, n_points).clip(80, 150)
        spo2 = rng.normal(95, 2, n_points).clip(88, 99)
        temp = 37.5 + rng.normal(0, 0.4, n_points)
        # Spike at midpoint
        mid  = n_points // 2
        hr[mid:mid+3]   += 30
        bp[mid:mid+3]   -= 25
        spo2[mid:mid+2] -= 7

    else:
        hr   = rng.normal(76, 5, n_points).clip(60, 95)
        bp   = rng.normal(122, 6, n_points).clip(105, 140)
        spo2 = rng.normal(97.5, 0.8, n_points).clip(95, 100)
        temp = rng.normal(37.0, 0.2, n_points).clip(36.4, 37.8)

    return pd.DataFrame({
        "Time":                 times,
        "Heart Rate (bpm)":    hr.clip(30, 180),
        "Systolic BP (mmHg)":  bp.clip(60, 220),
        "SpO2 (%)":            spo2.clip(70, 100),
        "Temperature (°C)":    temp.clip(34, 42),
    })


SAMPLE_SCENARIOS = {
    "Stable Patient": "stable",
    "Deteriorating Patient": "deteriorating",
    "Post-Surgery Recovery": "post_surgery",
}