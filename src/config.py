"""Central configuration. Everything tunable lives here or in environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_PATH = Path(os.environ.get("PC_DATA_PATH", DATA_DIR / "port_calls.csv"))
MODEL_PATH = Path(os.environ.get("PC_MODEL_PATH", OUT_DIR / "model.joblib"))
METRICS_PATH = OUT_DIR / "metrics.json"

SEED = int(os.environ.get("PC_SEED", 42))
N_RECORDS = int(os.environ.get("PC_N_RECORDS", 450))

# --- Definition of "congestion" -------------------------------------------------
# A vessel call is congested if its (observed) wait for a berth exceeds this many hours.
CONGESTION_THRESHOLD_H = float(os.environ.get("PC_THRESHOLD_H", 12))

# --- Simulation -----------------------------------------------------------------
SIM_START = "2025-01-01"
SIM_DAYS = 365
WARMUP_DAYS = 14          # simulated but discarded so queues and 7-day history are warm
WAIT_NOISE_SD_H = 1.5     # timestamp / reporting noise on the observed wait
MISSING_RATES = {         # share of rows where the field is unavailable
    "weather": 0.06,      # sensor outage -> wind, wave, storm flag missing together
    "queue_length": 0.04, # AIS coverage gap
    "avg_wait_prev_7d_h": 0.05,
}

# Synthetic ports. berths/cranes are capacity; rate = mean arrivals per day;
# svc_h = mean berth time (hours) for a 5,000 TEU vessel with 2 cranes per berth.
PORTS = {
    "PORT_A": dict(berths=4, cranes=8, rate=1.9, svc_h=24, wind_base=11.0),
    "PORT_B": dict(berths=6, cranes=12, rate=2.9, svc_h=24, wind_base=12.0),
    "PORT_C": dict(berths=3, cranes=5, rate=1.25, svc_h=24, wind_base=14.0),
    "PORT_D": dict(berths=8, cranes=18, rate=4.1, svc_h=24, wind_base=10.0),
    "PORT_E": dict(berths=5, cranes=8, rate=2.0, svc_h=26, wind_base=13.0),
}

# --- Modelling ------------------------------------------------------------------
THRESHOLD_BETA = float(os.environ.get("PC_THRESHOLD_BETA", 1.0))  # F-beta used to pick the alert threshold
TEST_FRACTION = 0.30      # chronologically last 30% of calls is the test set
N_BOOTSTRAP = 30          # models in the confidence ensemble
STD_CAP = 0.30            # ensemble std at which model-agreement credit drops to zero
OOD_PENALTY = 0.80        # confidence multiplier when an input is outside the training range
CONF_HIGH = 0.70
CONF_MEDIUM = 0.50
