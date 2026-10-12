"""Turn a prediction into an explanation a port planner can act on.

SHAP values (exact for tree models, additive on the log-odds scale) are summed into five
readable factors. Sentences are template-based on purpose: no LLM, no API, fully
reproducible, and the text can only mention what the model actually used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import FEATURES, GROUPS, LABELS
from .modeling import Bundle

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ACTIONS = {
    "Berth pressure (demand vs capacity)":
        "Stagger berth windows and prioritise quick-turn vessels; ask inbound carriers to slow-steam or "
        "re-time arrivals; add a berth/gang shift if labour allows.",
    "Weather":
        "Plan for weather holds: warn carriers and customers of probable delay now and schedule "
        "recovery shifts for when the front passes.",
    "Recent port performance":
        "A backlog is already building (high recent waits / low throughput): check crane and labour "
        "availability and clear yard congestion before accepting more calls.",
    "Timing and seasonality":
        "This is a known peak window: roster extra gang cover for this hour, weekday and season.",
    "Vessel and port characteristics":
        "Large call or port-specific constraint: assign the best-equipped berth and extra cranes.",
}


def _fmt(v, nd=0):
    return "n/a" if pd.isna(v) else f"{v:.{nd}f}"


def _describe(group: str, r: pd.Series, missing: set) -> str:
    port = next((c[5:] for c in r.index if c.startswith("port_") and r[c] == 1), "?")
    if group.startswith("Berth"):
        return (f"{_fmt(r['queue_length'])} vessels queued + {_fmt(r['arrivals_next_24h'])} due in 24h "
                f"vs {_fmt(r['berths'])} berths (load ratio {_fmt(r['demand_capacity_ratio'], 2)})")
    if group == "Weather":
        s = f"wind {_fmt(r['wind_kn'])} kn, waves {_fmt(r['wave_m'], 1)} m"
        return s + (", storm in progress" if r["storm_flag"] == 1 else "")
    if group.startswith("Recent"):
        return (f"avg wait last 7 days {_fmt(r['avg_wait_prev_7d_h'], 1)} h, "
                f"throughput {_fmt(r['throughput_prev_7d'], 1)} vessels/day")
    if group.startswith("Timing"):
        return f"{WEEKDAYS[int(r['weekday'])]} {int(r['hour']):02d}:00, {MONTHS[int(r['month']) - 1]}"
    return f"{_fmt(r['vessel_teu'])} TEU vessel at {port}, {_fmt(r['cranes'])} cranes"


class Explainer:
    def __init__(self, bundle: Bundle):
        self.b = bundle
        self.imp = bundle.main.pipeline.named_steps["imp"]
        self.clf = bundle.main.pipeline.named_steps["clf"]
        self.tree = shap.TreeExplainer(self.clf)

    def shap_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        Xi = self.imp.transform(X[self.b.features])
        return pd.DataFrame(np.asarray(self.tree.shap_values(Xi)), columns=self.b.features, index=X.index)

    def explain(self, X: pd.DataFrame) -> list[dict]:
        """X: feature matrix (one or more rows). Returns one result dict per row."""
        pred = self.b.predict_with_confidence(X)
        sv = self.shap_matrix(X)
        results = []
        for idx in X.index:
            row, s, pr = X.loc[idx], sv.loc[idx], pred.loc[idx]
            missing = {f for f in FEATURES if pd.isna(row[f])}
            factors = []
            for g, feats in GROUPS.items():
                contrib = float(s[feats].sum())
                top = sorted(feats, key=lambda f: -abs(s[f]))[:2]
                factors.append({
                    "factor": g, "contribution": contrib,
                    "detail": _describe(g, row, missing),
                    "top_features": [(LABELS[f], float(s[f])) for f in top],
                })
            total = sum(abs(f["contribution"]) for f in factors) or 1.0
            for f in factors:
                f["share"] = abs(f["contribution"]) / total
                f["direction"] = "raises risk" if f["contribution"] > 0 else "lowers risk"
            factors.sort(key=lambda f: -abs(f["contribution"]))

            p, conf = float(pr["probability"]), float(pr["confidence"])
            risk = "HIGH" if p >= 0.6 else "MODERATE" if p >= 0.3 else "LOW"
            up = [f for f in factors if f["contribution"] > 0 and f["share"] >= 0.15]
            actions = [ACTIONS[f["factor"]] for f in up[:3]]
            names = {f["factor"] for f in up}
            compound = (len([n for n in names if n.startswith(("Berth", "Weather"))]) == 2)
            if risk == "LOW":
                actions = ["No intervention needed; keep monitoring queue and weather."]
            if pr["confidence_label"] == "Low":
                actions.append("Confidence is low: verify the inputs (queue, ETAs, weather) before acting.")
            results.append({
                "probability": p, "risk": risk, "confidence": conf,
                "confidence_label": pr["confidence_label"],
                "predicted_congested": int(pr["predicted_congested"]),
                "ensemble_std": float(pr["ensemble_std"]),
                "out_of_range": list(pr["out_of_range"]),
                "missing_inputs": sorted(LABELS[f] for f in missing),
                "factors": factors, "actions": actions, "compound": compound,
            })
        return results


def format_report(r: dict) -> str:
    lines = [
        f"Congestion probability : {r['probability']:.0%}   ({r['risk']} risk)",
        (f"Confidence             : {r['confidence']:.2f}  ({r['confidence_label']})"
         f"   [model spread {r['ensemble_std']:.2f}]"),
        f"Alert (operating thr.) : {'CONGESTION EXPECTED' if r['predicted_congested'] else 'no alert'}",
        "",
        "Why (factor contributions, log-odds; + raises risk):",
    ]
    for f in r["factors"]:
        lines.append(f"  {f['contribution']:+.2f}  {f['factor']:<38} {f['detail']}")
    if r["compound"]:
        lines.append("  -> Heavy berth pressure AND adverse weather are compounding the risk.")
    lines += ["", "Suggested actions:"] + [f"  - {a}" for a in r["actions"]]
    notes = []
    if r["out_of_range"]:
        notes.append("outside training range: " + ", ".join(r["out_of_range"]))
    if r["missing_inputs"]:
        notes.append("missing inputs (imputed): " + ", ".join(r["missing_inputs"]))
    if notes:
        lines += ["", "Data notes: " + "; ".join(notes)]
    return "\n".join(lines)
