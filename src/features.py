"""Feature engineering and the chronological train/test split."""
from __future__ import annotations

import pandas as pd

from . import config

# Raw numeric inputs a planner can know at the moment a vessel arrives.
RAW_NUMERIC = [
    "vessel_teu", "berths", "cranes",
    "queue_length", "berth_utilization", "arrivals_next_24h", "arrivals_last_24h",
    "avg_wait_prev_7d_h", "throughput_prev_7d",
    "wind_kn", "wave_m", "storm_flag",
    "hour", "weekday", "month",
]
DERIVED = ["demand_capacity_ratio"]
PORT_COLS = [f"port_{p}" for p in config.PORTS]
FEATURES = RAW_NUMERIC + DERIVED + PORT_COLS
TARGET = "congested"

# Features whose out-of-range values should lower confidence (continuous, not calendar/binary).
RANGE_CHECKED = [
    "vessel_teu", "queue_length", "berth_utilization", "arrivals_next_24h", "arrivals_last_24h",
    "avg_wait_prev_7d_h", "throughput_prev_7d", "wind_kn", "wave_m", "demand_capacity_ratio",
]

# Factors used to turn per-feature SHAP values into a readable explanation.
GROUPS = {
    "Berth pressure (demand vs capacity)": [
        "queue_length", "arrivals_next_24h", "arrivals_last_24h", "demand_capacity_ratio",
        "berth_utilization", "berths",
    ],
    "Weather": ["wind_kn", "wave_m", "storm_flag"],
    "Recent port performance": ["avg_wait_prev_7d_h", "throughput_prev_7d"],
    "Timing and seasonality": ["hour", "weekday", "month"],
    "Vessel and port characteristics": ["vessel_teu", "cranes"] + PORT_COLS,
}

LABELS = {
    "queue_length": "vessels already queued",
    "arrivals_next_24h": "vessels due in next 24h",
    "arrivals_last_24h": "vessels arrived in last 24h",
    "demand_capacity_ratio": "(queue + arrivals) per berth",
    "berth_utilization": "berth utilisation",
    "berths": "berths",
    "cranes": "cranes",
    "vessel_teu": "vessel size (TEU)",
    "wind_kn": "wind (kn)",
    "wave_m": "wave height (m)",
    "storm_flag": "storm in progress",
    "avg_wait_prev_7d_h": "avg wait, last 7 days (h)",
    "throughput_prev_7d": "throughput, last 7 days (vessels/day)",
    "hour": "hour of day",
    "weekday": "day of week",
    "month": "month",
}
for _p in config.PORTS:
    LABELS[f"port_{_p}"] = f"port is {_p}"


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature matrix. Missing values are left as NaN (imputed inside the model)."""
    X = df[RAW_NUMERIC].astype(float).copy()
    X["demand_capacity_ratio"] = (df["queue_length"] + df["arrivals_next_24h"]) / df["berths"]
    for p in config.PORTS:
        X[f"port_{p}"] = (df["port_id"] == p).astype(float)
    return X[FEATURES]


def time_split(df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    """Chronological split: train on earlier calls, test on later calls (no shuffling)."""
    df = df.sort_values("arrival_ts").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["arrival_ts"]).sort_values("arrival_ts").reset_index(drop=True)
