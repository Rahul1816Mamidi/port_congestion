"""Models, probability calibration and the confidence score.

Three predictors are trained so the complex one has to earn its place:
  1. rule      - one threshold on the demand/capacity ratio
  2. logreg    - logistic regression (interpretable linear baseline)
  3. main      - gradient boosting (captures interactions such as weather x traffic)

Probabilities from (2) and (3) are calibrated with Platt scaling fitted on out-of-fold
predictions from an expanding-window (time-ordered) split, so calibration never looks
at the future.

Confidence = calibrated probability of the predicted outcome, discounted by
  * disagreement between bootstrap-trained models (ensemble spread), and
  * inputs that fall outside the range seen in training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURES, RANGE_CHECKED

EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_gb(seed: int, n_estimators: int = 150) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=5, random_state=seed)),
    ])


def make_lr(seed: int) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
    ])


def best_threshold(y, p, beta: float = config.THRESHOLD_BETA) -> float:
    """Alert threshold maximising F-beta on out-of-fold predictions (beta > 1 favours recall)."""
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


@dataclass
class CalibratedModel:
    pipeline: Pipeline
    platt: LogisticRegression
    threshold: float

    def raw_logit(self, X) -> np.ndarray:
        return logit(self.pipeline.predict_proba(X)[:, 1])

    def predict_proba(self, X) -> np.ndarray:
        return self.platt.predict_proba(self.raw_logit(X).reshape(-1, 1))[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


def fit_calibrated(make_fn, X: pd.DataFrame, y: pd.Series, seed: int, n_splits: int = 4) -> CalibratedModel:
    """X must be in chronological order."""
    scores, labels = [], []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        if y.iloc[tr].nunique() < 2:
            continue
        m = make_fn(seed).fit(X.iloc[tr], y.iloc[tr])
        scores.append(logit(m.predict_proba(X.iloc[va])[:, 1]))
        labels.append(y.iloc[va].to_numpy())
    s, l = np.concatenate(scores), np.concatenate(labels)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), l)
    oof_p = platt.predict_proba(s.reshape(-1, 1))[:, 1]
    threshold = best_threshold(l, oof_p)
    final = make_fn(seed).fit(X, y)
    return CalibratedModel(final, platt, threshold)


def fit_bootstrap(X: pd.DataFrame, y: pd.Series, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    models = []
    for b in range(n):
        idx = rng.integers(0, len(X), len(X))
        if y.iloc[idx].nunique() < 2:
            continue
        models.append(make_gb(seed + b + 1, n_estimators=100).fit(X.iloc[idx], y.iloc[idx]))
    return models


@dataclass
class Bundle:
    features: list
    main: CalibratedModel
    logreg: CalibratedModel
    rule_threshold: float
    rule_fill: float
    bootstrap: list
    ranges: dict          # feature -> (min, max) seen in training
    train_end: pd.Timestamp
    n_train: int

    # ---- confidence machinery -------------------------------------------------
    def ensemble_std(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.vstack([
            self.main.platt.predict_proba(logit(m.predict_proba(X)[:, 1]).reshape(-1, 1))[:, 1]
            for m in self.bootstrap
        ])
        return probs.std(axis=0)

    def out_of_range(self, X: pd.DataFrame) -> list:
        out = []
        for _, row in X.iterrows():
            bad = [f for f in RANGE_CHECKED
                   if pd.notna(row[f]) and not (self.ranges[f][0] <= row[f] <= self.ranges[f][1])]
            out.append(bad)
        return out

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.main.predict_proba(X)
        std = self.ensemble_std(X)
        ood = self.out_of_range(X)
        conf = np.maximum(p, 1 - p) * (1 - np.minimum(1, std / config.STD_CAP))
        conf = np.where([len(o) > 0 for o in ood], conf * config.OOD_PENALTY, conf)
        label = np.where(conf >= config.CONF_HIGH, "High",
                         np.where(conf >= config.CONF_MEDIUM, "Medium", "Low"))
        return pd.DataFrame({
            "probability": p, "ensemble_std": std, "out_of_range": ood,
            "confidence": conf, "confidence_label": label,
            "predicted_congested": (p >= self.main.threshold).astype(int),
        }, index=X.index)

    def rule_score(self, X: pd.DataFrame) -> np.ndarray:
        return X["demand_capacity_ratio"].fillna(self.rule_fill).to_numpy()


def train_bundle(train_df: pd.DataFrame, build_matrix, seed: int = config.SEED) -> Bundle:
    X, y = build_matrix(train_df), train_df["congested"]
    main = fit_calibrated(make_gb, X, y, seed)
    logreg = fit_calibrated(make_lr, X, y, seed)
    fill = float(X["demand_capacity_ratio"].median())
    score = X["demand_capacity_ratio"].fillna(fill).to_numpy()
    grid = np.unique(np.round(np.quantile(score, np.linspace(0.05, 0.95, 91)), 3))
    f2 = [fbeta_score(y, (score >= t).astype(int), beta=config.THRESHOLD_BETA, zero_division=0) for t in grid]
    rule_t = float(grid[int(np.argmax(f2))])
    boot = fit_bootstrap(X, y, config.N_BOOTSTRAP, seed)
    ranges = {f: (float(X[f].min()), float(X[f].max())) for f in RANGE_CHECKED}
    return Bundle(list(FEATURES), main, logreg, rule_t, fill, boot, ranges,
                  train_df["arrival_ts"].max(), len(train_df))
