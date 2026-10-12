"""How much do the results depend on one lucky random draw?

Regenerates the synthetic data and retrains for several seeds, then reports the spread of the
held-out (chronologically later) scores. A single 135-row test set is noisy; this shows by how much.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config
from .features import build_matrix, time_split
from .generate_data import generate
from .modeling import train_bundle

SEEDS = [1, 2, 3, 4, 5]


def main() -> None:
    rows = []
    for seed in SEEDS:
        df = generate(config.N_RECORDS, seed)
        tr, te = time_split(df)
        b = train_bundle(tr, build_matrix, seed)
        X, y = build_matrix(te), te["congested"].to_numpy()
        pc = b.predict_with_confidence(X)
        hi = (pc.confidence_label == "High").to_numpy()
        acc_hi = float(((pc.probability[hi] >= 0.5).astype(int) == y[hi]).mean()) if hi.any() else float("nan")
        rows.append({
            "seed": seed, "positive_rate_test": float(y.mean()),
            "auc_rule": float(roc_auc_score(y, b.rule_score(X))),
            "auc_logreg": float(roc_auc_score(y, b.logreg.predict_proba(X))),
            "auc_main": float(roc_auc_score(y, b.main.predict_proba(X))),
            "brier_main": float(brier_score_loss(y, b.main.predict_proba(X))),
            "share_high_conf": float(hi.mean()), "acc_high_conf": acc_hi,
        })
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()})
    summary = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "sd": float(np.nanstd([r[k] for r in rows]))}
               for k in rows[0] if k != "seed"}
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "seed_stability.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))
    print("\nmean +/- sd over seeds", SEEDS)
    for k, v in summary.items():
        print(f"  {k:<20} {v['mean']:.3f} +/- {v['sd']:.3f}")


if __name__ == "__main__":
    main()
