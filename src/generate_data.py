"""Synthetic port-call generator.

Instead of writing delay = f(features) + noise (which would make the model just
recover my formula), the generator simulates the port as a queueing system:

  * vessels arrive over time (seasonality, weekday/hour pattern, convoys, post-storm surges)
  * each port has a fixed number of berths; vessels are served first come, first served
  * service time depends on vessel size, crane intensity and weather at berthing
  * no berthing starts during a storm (pilot / safety hold)

The waiting time that comes out of this simulation is the ground truth. The features a
planner could know *at arrival time* are then computed from the simulated port state,
the label is derived from a noisy observation of the wait, and some inputs are blanked
to mimic real data gaps. Finally a sample of calls is drawn (default 450 rows).
"""
from __future__ import annotations

import argparse
import heapq

import numpy as np
import pandas as pd

from . import config


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """Unit-variance AR(1) series."""
    eps = rng.standard_normal(n)
    z = np.empty(n)
    z[0] = eps[0]
    s = np.sqrt(1 - phi**2)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + s * eps[t]
    return z


def _simulate_weather(rng, ts: pd.DatetimeIndex, spec: dict):
    n = len(ts)
    winter = 0.5 * (1 + np.cos(2 * np.pi * (ts.dayofyear.values - 15) / 365))
    wind = spec["wind_base"] + 4.0 * winter + 3.5 * _ar1(rng, n, 0.985)
    intensity = np.zeros(n)
    storm = np.zeros(n, dtype=bool)
    storm_ends = []
    starts = np.where(rng.random(n) < (0.03 + 0.05 * winter) / 24)[0]
    for s in starts:
        e = min(n, s + int(rng.uniform(10, 40)))
        if e - s < 4:
            continue
        ramp = np.sin(np.linspace(0, np.pi, e - s))
        intensity[s:e] = np.maximum(intensity[s:e], ramp)
        storm[s:e] = True
        storm_ends.append(e)
    wind = np.clip(wind + 14.0 * intensity, 0, None)
    wave = np.clip(0.4 + 0.07 * wind + 0.25 * _ar1(rng, n, 0.99) + 1.5 * intensity, 0.1, None)
    return wind, wave, storm, storm_ends


def _simulate_arrivals(rng, ts: pd.DatetimeIndex, spec: dict, storm_ends: list) -> np.ndarray:
    n = len(ts)
    season = 1 + 0.15 * np.cos(2 * np.pi * (ts.month.values - 10) / 12)
    dow = np.where(ts.dayofweek.values == 0, 1.15, np.where(ts.dayofweek.values >= 5, 0.85, 1.0))
    hod = 1 + 0.15 * np.cos(2 * np.pi * (ts.hour.values - 14) / 24)
    lam = spec["rate"] / 24 * season * dow * hod
    counts = rng.poisson(lam)
    times = [np.repeat(np.arange(n), counts) + rng.random(counts.sum())]
    for _ in range(rng.poisson(n / 24 / 9)):          # convoys: bunched vessels after upstream delays
        k = int(rng.integers(2, 5)) + spec["berths"] // 2
        times.append(rng.uniform(0, n - 24) + rng.uniform(0, 10, size=k))
    for e in storm_ends:                              # post-storm surge of delayed vessels
        if rng.random() < 0.7:
            times.append(e + rng.uniform(0, 20, size=rng.poisson(spec["rate"] * 0.8)))
    out = np.sort(np.concatenate(times))
    return out[out < n - 2]


def simulate_port(name: str, spec: dict, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp(config.SIM_START) - pd.Timedelta(days=config.WARMUP_DAYS)
    n = (config.SIM_DAYS + config.WARMUP_DAYS) * 24
    ts = pd.date_range(start, periods=n, freq="h")
    wind, wave, storm, storm_ends = _simulate_weather(rng, ts, spec)
    arr = _simulate_arrivals(rng, ts, spec, storm_ends)
    m = len(arr)

    # first hour >= h without a storm hold
    next_ok = np.empty(n + 1)
    next_ok[n] = n
    for h in range(n - 1, -1, -1):
        next_ok[h] = h if not storm[h] else next_ok[h + 1]

    teu = np.clip(np.exp(rng.normal(np.log(5000), 0.6, m)), 800, 20000)
    cpb = spec["cranes"] / spec["berths"]
    base_svc = spec["svc_h"] * (teu / 5000) ** 0.6 * (2 / cpb) ** 0.5

    berth_free = [0.0] * spec["berths"]
    heapq.heapify(berth_free)
    start_t = np.empty(m)
    finish_t = np.empty(m)
    for i in range(m):
        s = max(arr[i], heapq.heappop(berth_free))
        h = min(int(s), n - 1)
        if storm[h]:
            s = max(s, next_ok[h])
            h = min(int(s), n - 1)
        wx = 1 + 0.03 * max(0.0, wind[h] - 18) + 0.20 * max(0.0, wave[h] - 1.8)
        svc = base_svc[i] * wx * rng.lognormal(0, 0.3)
        start_t[i], finish_t[i] = s, s + svc
        heapq.heappush(berth_free, finish_t[i])

    wait = start_t - arr
    # ---- features known at arrival time (no future information about the vessel itself) ----
    queue = np.array([(start_t[:i] > arr[i]).sum() for i in range(m)])
    busy = np.array([((start_t[:i] <= arr[i]) & (finish_t[:i] > arr[i])).sum() for i in range(m)])
    nxt = np.array([((arr > arr[i]) & (arr <= arr[i] + 24)).sum() for i in range(m)])
    nxt = np.round(nxt * rng.uniform(0.8, 1.2, m)).astype(int)   # schedule / ETA uncertainty
    last = np.array([(arr[:i] > arr[i] - 24).sum() for i in range(m)])
    avg_wait7 = np.full(m, np.nan)
    thru7 = np.zeros(m)
    for i in range(m):
        w = (start_t[:i] <= arr[i]) & (start_t[:i] > arr[i] - 168)
        if w.any():
            avg_wait7[i] = wait[:i][w].mean()
        thru7[i] = ((finish_t[:i] <= arr[i]) & (finish_t[:i] > arr[i] - 168)).sum() / 7
    hh = np.minimum(arr.astype(int), n - 1)
    stamp = start + pd.to_timedelta(arr, unit="h")

    df = pd.DataFrame(
        {
            "port_id": name,
            "arrival_ts": stamp.floor("min"),
            "vessel_teu": teu.round(-1),
            "berths": spec["berths"],
            "cranes": spec["cranes"],
            "queue_length": queue,
            "berth_utilization": np.minimum(busy / spec["berths"], 1.0).round(3),
            "arrivals_next_24h": nxt,
            "arrivals_last_24h": last,
            "avg_wait_prev_7d_h": avg_wait7.round(2),
            "throughput_prev_7d": thru7.round(3),
            "wind_kn": (wind[hh] + rng.normal(0, 0.8, m)).round(1),
            "wave_m": (wave[hh] + rng.normal(0, 0.1, m)).round(2),
            "storm_flag": storm[hh].astype(int),
            "wait_true_h": wait,
        }
    )
    return df[df["arrival_ts"] >= pd.Timestamp(config.SIM_START)]


def generate(n_records: int = config.N_RECORDS, seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [simulate_port(name, spec, rng) for name, spec in config.PORTS.items()]
    full = pd.concat(frames, ignore_index=True)

    per_port = n_records // len(config.PORTS)
    parts = []
    for name, g in full.groupby("port_id"):
        parts.append(g.sample(n=min(per_port, len(g)), random_state=int(rng.integers(1e9))))
    df = pd.concat(parts).sort_values("arrival_ts").reset_index(drop=True)

    df["hour"] = df["arrival_ts"].dt.hour
    df["weekday"] = df["arrival_ts"].dt.dayofweek
    df["month"] = df["arrival_ts"].dt.month

    # Ground truth: noisy observed wait, thresholded.
    observed = np.maximum(0, df["wait_true_h"] + rng.normal(0, config.WAIT_NOISE_SD_H, len(df)))
    df["wait_hours"] = observed.round(2)
    df["congested"] = (df["wait_hours"] > config.CONGESTION_THRESHOLD_H).astype(int)
    df = df.drop(columns=["wait_true_h"])

    # Data gaps.
    wx = rng.random(len(df)) < config.MISSING_RATES["weather"]
    df.loc[wx, ["wind_kn", "wave_m", "storm_flag"]] = np.nan
    df.loc[rng.random(len(df)) < config.MISSING_RATES["queue_length"], "queue_length"] = np.nan
    df.loc[rng.random(len(df)) < config.MISSING_RATES["avg_wait_prev_7d_h"], "avg_wait_prev_7d_h"] = np.nan

    df.insert(0, "call_id", [f"C{i + 1:04d}" for i in range(len(df))])
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-records", type=int, default=config.N_RECORDS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=str(config.DATA_PATH))
    a = ap.parse_args()
    df = generate(a.n_records, a.seed)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}")
    print(f"congested share: {df['congested'].mean():.1%}")
    print(df.groupby("port_id")["congested"].agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
