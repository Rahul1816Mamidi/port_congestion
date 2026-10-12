import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src import predict as predict_cli
from src.explain import Explainer
from src.features import FEATURES, build_matrix, time_split
from src.generate_data import generate
from src.modeling import train_bundle


@pytest.fixture(scope="session")
def data():
    return generate(450, 3)


@pytest.fixture(scope="session")
def bundle(data):
    old = config.N_BOOTSTRAP
    config.N_BOOTSTRAP = 6  # keep the test fast
    try:
        train_df, _ = time_split(data)
        return train_bundle(train_df, build_matrix, seed=3)
    finally:
        config.N_BOOTSTRAP = old


# ---------------------------------------------------------------- data
def test_generator_shape_balance_and_gaps(data):
    assert len(data) == 450
    assert {"call_id", "port_id", "arrival_ts", "congested", "wait_hours"} <= set(data.columns)
    assert 0.2 < data["congested"].mean() < 0.6          # usable class balance
    assert data["queue_length"].isna().any() and data["wind_kn"].isna().any()  # injected gaps
    assert data["queue_length"].dropna().ge(0).all()
    assert data["berth_utilization"].between(0, 1).all()
    assert data["port_id"].nunique() == len(config.PORTS)


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate(100, 11), generate(100, 11))


def test_label_matches_definition(data):
    assert ((data["wait_hours"] > config.CONGESTION_THRESHOLD_H).astype(int) == data["congested"]).all()


# ---------------------------------------------------------------- features / split
def test_time_split_is_chronological(data):
    tr, te = time_split(data)
    assert tr["arrival_ts"].max() <= te["arrival_ts"].min()
    assert len(te) == pytest.approx(len(data) * config.TEST_FRACTION, abs=1)


def test_demand_capacity_ratio(data):
    X = build_matrix(data)
    r = data.dropna(subset=["queue_length"]).iloc[0]
    expected = (r["queue_length"] + r["arrivals_next_24h"]) / r["berths"]
    assert X.loc[r.name, "demand_capacity_ratio"] == pytest.approx(expected)
    assert list(X.columns) == FEATURES


# ---------------------------------------------------------------- model and explanation
def test_shap_is_additive(bundle, data):
    _, te = time_split(data)
    X = build_matrix(te)
    ex = Explainer(bundle)
    sv = ex.shap_matrix(X)
    raw = ex.clf.decision_function(ex.imp.transform(X[bundle.features]))
    base = float(np.ravel(ex.tree.expected_value)[0])
    assert np.abs(base + sv.sum(axis=1).to_numpy() - raw).max() < 1e-6


def test_factor_contributions_sum_to_total_shap(bundle, data):
    _, te = time_split(data)
    X = build_matrix(te).iloc[:5]
    ex = Explainer(bundle)
    sv = ex.shap_matrix(X)
    for res, (_, s) in zip(ex.explain(X), sv.iterrows()):
        assert sum(f["contribution"] for f in res["factors"]) == pytest.approx(s.sum(), abs=1e-6)


def test_scenarios_are_ordered_and_outputs_valid(bundle):
    ex = Explainer(bundle)
    res = {}
    for name in ("calm", "busy", "storm-peak"):
        a = predict_cli.build_parser().parse_args(["--scenario", name, "--port", "PORT_C"])
        res[name] = ex.explain(build_matrix(predict_cli.row_from_args(a)))[0]
    assert res["calm"]["probability"] < 0.3
    assert res["storm-peak"]["probability"] > 0.7
    for r in res.values():
        assert 0 <= r["probability"] <= 1 and 0 <= r["confidence"] <= 1
        assert r["confidence_label"] in {"High", "Medium", "Low"}
        assert r["actions"]


def test_missing_inputs_do_not_crash_and_are_reported(bundle):
    a = predict_cli.build_parser().parse_args(["--scenario", "busy"])
    row = predict_cli.row_from_args(a)
    row.loc[0, ["queue_length", "wind_kn"]] = np.nan
    r = Explainer(bundle).explain(build_matrix(row))[0]
    assert "vessels already queued" in r["missing_inputs"] and 0 <= r["probability"] <= 1


def test_out_of_range_input_lowers_confidence(bundle):
    ex = Explainer(bundle)
    normal = predict_cli.build_parser().parse_args(["--scenario", "busy"])
    extreme = predict_cli.build_parser().parse_args(["--scenario", "busy", "--queue", "500", "--arrivals-24h", "400"])
    r_norm = ex.explain(build_matrix(predict_cli.row_from_args(normal)))[0]
    r_ext = ex.explain(build_matrix(predict_cli.row_from_args(extreme)))[0]
    assert r_ext["out_of_range"] and not r_norm["out_of_range"]


# ---------------------------------------------------------------- CLI
def test_cli_json(bundle, tmp_path, monkeypatch, capsys):
    path = tmp_path / "m.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setattr(config, "MODEL_PATH", path)
    assert predict_cli.main(["--scenario", "storm-peak", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {"probability", "confidence", "factors", "actions"} <= set(out)
