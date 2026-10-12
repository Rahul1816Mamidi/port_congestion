"""Evaluate on the chronologically later 30% of calls and write metrics + figures."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config
from .explain import Explainer
from .features import GROUPS, LABELS, build_matrix, load_data, time_split
from .modeling import fit_calibrated, make_gb

RNG = np.random.default_rng(config.SEED)


def _cls_metrics(y, pred) -> dict:
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alert_rate": float(np.mean(pred)),
    }


def _auc_ci(y, p, n_boot=1000):
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _ece(y, p, bins=5):
    q = pd.qcut(p, q=bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return float(sum(len(d) * abs(d.y.mean() - d.p.mean()) for _, d in g) / len(y))


def main() -> None:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(config.MODEL_PATH)
    df = load_data()
    train_df, test_df = time_split(df)
    Xte, yte = build_matrix(test_df), test_df["congested"].to_numpy()
    ytr = train_df["congested"].to_numpy()

    out: dict = {
        "n_total": len(df), "n_train": len(train_df), "n_test": len(test_df),
        "congestion_threshold_h": config.CONGESTION_THRESHOLD_H,
        "train_period": [str(train_df.arrival_ts.min().date()), str(train_df.arrival_ts.max().date())],
        "test_period": [str(test_df.arrival_ts.min().date()), str(test_df.arrival_ts.max().date())],
        "positive_rate_train": float(ytr.mean()), "positive_rate_test": float(yte.mean()),
        "missing_share": {k: float(v) for k, v in df.isna().mean().items() if v > 0},
    }

    # ---- model comparison --------------------------------------------------------
    p_main = bundle.main.predict_proba(Xte)
    p_lr = bundle.logreg.predict_proba(Xte)
    s_rule = bundle.rule_score(Xte)
    base_rate = float(ytr.mean())
    models = {
        "rule (load ratio)": dict(score=s_rule, pred=(s_rule >= bundle.rule_threshold).astype(int), prob=None),
        "logistic regression": dict(score=p_lr, pred=(p_lr >= bundle.logreg.threshold).astype(int), prob=p_lr),
        "gradient boosting (main)": dict(score=p_main, pred=(p_main >= bundle.main.threshold).astype(int), prob=p_main),
    }
    comp = {}
    for name, m in models.items():
        d = {"roc_auc": float(roc_auc_score(yte, m["score"])),
             "pr_auc": float(average_precision_score(yte, m["score"]))}
        if m["prob"] is not None:
            d["brier"] = float(brier_score_loss(yte, m["prob"]))
        d.update(_cls_metrics(yte, m["pred"]))
        comp[name] = d
    out["models"] = comp
    out["brier_constant_baseline"] = float(brier_score_loss(yte, np.full(len(yte), base_rate)))
    out["main_auc_ci95"] = _auc_ci(yte, p_main)
    out["thresholds"] = {"main": bundle.main.threshold, "logreg": bundle.logreg.threshold,
                         "rule_load_ratio": bundle.rule_threshold}

    # ---- threshold trade-off (operational choice) -----------------------------------
    out["threshold_tradeoff"] = {
        f"{t:.2f}": _cls_metrics(yte, (p_main >= t).astype(int)) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    }

    # ---- ablation: what if the live queue state is unavailable (earlier warning)? -------
    live = ["queue_length", "berth_utilization", "avg_wait_prev_7d_h", "arrivals_last_24h",
            "throughput_prev_7d", "demand_capacity_ratio"]
    keep = [c for c in Xte.columns if c not in live] + ["arrivals_next_24h"]
    keep = list(dict.fromkeys(keep))
    Xtr_full = build_matrix(train_df)
    abl = fit_calibrated(make_gb, Xtr_full[keep], train_df["congested"], config.SEED)
    p_abl = abl.predict_proba(Xte[keep])
    out["ablation_no_live_state"] = {
        "features_dropped": live,
        "roc_auc": float(roc_auc_score(yte, p_abl)), "pr_auc": float(average_precision_score(yte, p_abl)),
        "brier": float(brier_score_loss(yte, p_abl)),
    }

    # ---- calibration ------------------------------------------------------------------
    out["calibration"] = {"ece_main": _ece(yte, p_main), "ece_logreg": _ece(yte, p_lr)}

    # ---- confidence score check: does higher confidence mean fewer mistakes? -----------
    pc = bundle.predict_with_confidence(Xte)
    pc["y"] = yte
    pc["correct"] = ((pc["probability"] >= 0.5).astype(int) == pc["y"]).astype(int)
    pc["sq_err"] = (pc["probability"] - pc["y"]) ** 2
    buckets = {}
    for lab in ("High", "Medium", "Low"):
        g = pc[pc.confidence_label == lab]
        buckets[lab] = {"n": len(g),
                        "accuracy": float(g.correct.mean()) if len(g) else None,
                        "brier": float(g.sq_err.mean()) if len(g) else None}
    out["confidence_buckets"] = buckets
    out["out_of_range_share_test"] = float(np.mean([len(o) > 0 for o in pc.out_of_range]))

    # ---- per-port ------------------------------------------------------------------------
    per_port = {}
    for port, g in test_df.groupby("port_id"):
        idx = [test_df.index.get_loc(i) for i in g.index]
        y_g, p_g = yte[idx], p_main[idx]
        per_port[port] = {
            "n": len(g), "positive_rate": float(y_g.mean()),
            "roc_auc": float(roc_auc_score(y_g, p_g)) if len(np.unique(y_g)) == 2 else None,
            "recall_at_alert": float(recall_score(y_g, (p_g >= bundle.main.threshold).astype(int), zero_division=0)),
        }
    out["per_port"] = per_port

    # ---- robustness to extra missing data ---------------------------------------------------
    key_cols = ["queue_length", "arrivals_next_24h", "avg_wait_prev_7d_h", "wind_kn", "wave_m", "throughput_prev_7d"]
    aucs = []
    for _ in range(30):
        Xn = Xte.copy()
        for c in key_cols:
            Xn.loc[RNG.random(len(Xn)) < 0.20, c] = np.nan
        Xn["demand_capacity_ratio"] = (Xn["queue_length"] + Xn["arrivals_next_24h"]) / Xn["berths"]
        aucs.append(roc_auc_score(yte, bundle.main.predict_proba(Xn)))
    out["robustness_20pct_extra_missing"] = {"roc_auc_mean": float(np.mean(aucs)), "roc_auc_min": float(np.min(aucs))}

    # ---- global explanation ------------------------------------------------------------------------
    ex = Explainer(bundle)
    sv = ex.shap_matrix(Xte)
    feat_imp = sv.abs().mean().sort_values(ascending=False)
    grp_imp = {g: float(sv[f].sum(axis=1).abs().mean()) for g, f in GROUPS.items()}
    out["global_importance_features"] = {LABELS[k]: float(v) for k, v in feat_imp.head(8).items()}
    out["global_importance_groups"] = dict(sorted(grp_imp.items(), key=lambda kv: -kv[1]))

    # ---- figures ------------------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(yte, m["score"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {comp[name]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on held-out later period")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "roc.png", dpi=170); plt.close(fig)

    q = pd.qcut(p_main, q=5, duplicates="drop")
    cal = pd.DataFrame({"y": yte, "p": p_main, "q": q}).groupby("q", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect")
    ax.plot(cal.p, cal.y, "o-", label="main model")
    ax.set(xlabel="Predicted probability (bin mean)", ylabel="Observed congestion rate",
           title="Calibration on held-out period", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "calibration.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    gi = pd.Series(grp_imp).sort_values()
    ax.barh([g.split(" (")[0] for g in gi.index], gi.values, color="#3b6ea8")
    ax.set(xlabel="Mean |SHAP| (log-odds)", title="What drives predictions (test set)")
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "global_factors.png", dpi=170); plt.close(fig)

    ranked = pc.sort_values("probability", ascending=False)
    pick = ranked[(ranked.confidence_label == "High")].index
    pick = pick[0] if len(pick) else ranked.index[0]
    res = ex.explain(Xte.loc[[pick]])[0]
    fs = sorted(res["factors"], key=lambda f: f["contribution"])
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.barh([f["factor"].split(" (")[0] for f in fs], [f["contribution"] for f in fs],
            color=["#c0392b" if f["contribution"] > 0 else "#2e86c1" for f in fs])
    ax.axvline(0, color="k", lw=0.6)
    ax.set(xlabel="Contribution to risk (log-odds)",
           title=f"Example call at {test_df.loc[pick, 'port_id']}: p={res['probability']:.0%}, {res['confidence_label']} confidence", )
    ax.title.set_fontsize(8.5)
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "example_explanation.png", dpi=170); plt.close(fig)
    out["example_row"] = {"call_id": test_df.loc[pick, "call_id"], "port": test_df.loc[pick, "port_id"]}

    config.METRICS_PATH.write_text(json.dumps(out, indent=2))

    # ---- console summary ---------------------------------------------------------------------------------------
    print(f"Test: {len(test_df)} calls {out['test_period'][0]} to {out['test_period'][1]} | "
          f"congested share {yte.mean():.1%} (train {ytr.mean():.1%})")
    print(pd.DataFrame(comp).T.round(3).to_string())
    print(f"main AUC 95% CI (bootstrap): {out['main_auc_ci95'][0]:.2f}-{out['main_auc_ci95'][1]:.2f} | "
          f"Brier const baseline {out['brier_constant_baseline']:.3f} | ECE main {out['calibration']['ece_main']:.3f}")
    print("Threshold trade-off (main):"); print(pd.DataFrame(out["threshold_tradeoff"]).T.round(2).to_string())
    print("Confidence buckets:", json.dumps(buckets))
    print("Per port:", json.dumps({k: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()} for k, v in per_port.items()}))
    print("Robustness (20% extra missing):", out["robustness_20pct_extra_missing"])
    print("Global factors:", {k: round(v, 2) for k, v in out["global_importance_groups"].items()})


if __name__ == "__main__":
    main()
