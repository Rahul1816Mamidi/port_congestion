"""Command-line predictor: probability + confidence + explanation + suggested actions.

Examples
  python -m src.predict --scenario storm-peak --port PORT_C
  python -m src.predict --port PORT_B --queue 6 --arrivals-24h 9 --wind 28 --wave 3 --storm
  python -m src.predict --call-id C0420          # a call from the dataset, shows the actual outcome too
  python -m src.predict --scenario busy --json
"""
from __future__ import annotations

import argparse
import json
import sys

import joblib
import pandas as pd

from . import config
from .explain import Explainer, format_report
from .features import RAW_NUMERIC, build_matrix, load_data

SCENARIOS = {
    "calm": dict(vessel_teu=4500, queue_length=0, berth_utilization=0.3, arrivals_next_24h=2,
                 arrivals_last_24h=2, avg_wait_prev_7d_h=2, throughput_prev_7d=3.0, wind_kn=9, wave_m=0.9,
                 storm_flag=0, hour=10, weekday=2, month=5),
    "busy": dict(vessel_teu=6000, queue_length=4, berth_utilization=1.0, arrivals_next_24h=6,
                 arrivals_last_24h=6, avg_wait_prev_7d_h=14, throughput_prev_7d=2.6, wind_kn=14, wave_m=1.4,
                 storm_flag=0, hour=8, weekday=0, month=10),
    "storm-peak": dict(vessel_teu=9000, queue_length=8, berth_utilization=1.0, arrivals_next_24h=10,
                       arrivals_last_24h=8, avg_wait_prev_7d_h=40, throughput_prev_7d=1.5, wind_kn=30, wave_m=3.5,
                       storm_flag=1, hour=3, weekday=0, month=12),
}

OVERRIDES = {  # CLI flag -> column
    "teu": "vessel_teu", "berths": "berths", "cranes": "cranes", "queue": "queue_length",
    "utilization": "berth_utilization", "arrivals_24h": "arrivals_next_24h",
    "arrivals_last_24h": "arrivals_last_24h", "avg_wait_7d": "avg_wait_prev_7d_h",
    "throughput_7d": "throughput_prev_7d", "wind": "wind_kn", "wave": "wave_m",
    "hour": "hour", "weekday": "weekday", "month": "month",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Predict port congestion for one vessel arrival.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--scenario", choices=SCENARIOS, default="busy", help="starting values (default: busy)")
    ap.add_argument("--port", choices=list(config.PORTS), default="PORT_C")
    ap.add_argument("--call-id", help="explain an existing call from the dataset instead")
    ap.add_argument("--arrival", help='arrival time, e.g. "2026-10-14 06:00" (sets hour/weekday/month)')
    ap.add_argument("--storm", action="store_true", help="storm in progress")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    for flag in OVERRIDES:
        ap.add_argument(f"--{flag.replace('_', '-')}", type=float)
    return ap


def row_from_args(a: argparse.Namespace) -> pd.DataFrame:
    spec = config.PORTS[a.port]
    r = dict(SCENARIOS[a.scenario])
    r.update(berths=spec["berths"], cranes=spec["cranes"])
    if a.arrival:
        ts = pd.Timestamp(a.arrival)
        r.update(hour=ts.hour, weekday=ts.dayofweek, month=ts.month)
    for flag, col in OVERRIDES.items():
        v = getattr(a, flag)
        if v is not None:
            r[col] = v
    if a.storm:
        r["storm_flag"] = 1
    r["port_id"] = a.port
    return pd.DataFrame([r])[RAW_NUMERIC + ["port_id"]]


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not config.MODEL_PATH.exists():
        sys.exit(f"No trained model at {config.MODEL_PATH}. Run: make train")
    bundle = joblib.load(config.MODEL_PATH)
    actual = None
    if a.call_id:
        df = load_data()
        hit = df[df["call_id"] == a.call_id]
        if hit.empty:
            sys.exit(f"call_id {a.call_id} not found in {config.DATA_PATH}")
        row, actual = hit.reset_index(drop=True), hit.iloc[0]
        title = f"{a.call_id} at {row.loc[0, 'port_id']}, arrival {row.loc[0, 'arrival_ts']:%Y-%m-%d %H:%M}"
    else:
        row = row_from_args(a)
        title = f"scenario '{a.scenario}' at {a.port}"
    X = build_matrix(row)
    res = Explainer(bundle).explain(X)[0]
    if a.json:
        print(json.dumps(res, indent=2, default=float))
    else:
        print(f"=== {title} ===")
        print(format_report(res))
        if actual is not None:
            print(f"\nActual outcome: wait {actual['wait_hours']:.1f} h -> "
                  f"{'CONGESTED' if actual['congested'] else 'not congested'} "
                  f"(threshold {config.CONGESTION_THRESHOLD_H:.0f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
