"""Train the models on the chronologically earlier part of the data and save a bundle."""
from __future__ import annotations

import joblib

from . import config
from .features import build_matrix, load_data, time_split
from .generate_data import generate
from .modeling import train_bundle


def main() -> None:
    if not config.DATA_PATH.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate().to_csv(config.DATA_PATH, index=False)
        print(f"data missing -> generated {config.DATA_PATH}")
    df = load_data()
    train_df, test_df = time_split(df)
    print(f"train: {len(train_df)} calls up to {train_df['arrival_ts'].max():%Y-%m-%d} | "
          f"test: {len(test_df)} calls from {test_df['arrival_ts'].min():%Y-%m-%d}")
    bundle = train_bundle(train_df, build_matrix)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    print(f"main-model alert threshold (F-beta on out-of-fold): {bundle.main.threshold:.2f}")
    print(f"rule baseline: congested if (queue + arrivals_24h)/berths >= {bundle.rule_threshold:.2f}")
    print(f"saved -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
