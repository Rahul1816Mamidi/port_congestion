"""Interactive tester for the port congestion model.

Run from the repository root:   streamlit run app.py

Enter the state of a port and an arriving vessel; the app returns the congestion probability,
a confidence score, the factors behind the prediction and suggested actions. It uses the same
trained model and explanation code as the command line (src/predict.py).
"""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.explain import MONTHS, WEEKDAYS, Explainer  # noqa: E402
from src.features import RAW_NUMERIC, build_matrix, load_data, time_split  # noqa: E402
from src.generate_data import generate  # noqa: E402
from src.modeling import train_bundle  # noqa: E402
from src.predict import SCENARIOS  # noqa: E402

st.set_page_config(page_title="Port Congestion Risk Estimator", layout="wide")

INT_FIELDS = {"vessel_teu", "queue_length", "arrivals_next_24h", "arrivals_last_24h", "hour", "weekday", "month"}
FLOAT_FIELDS = {"berth_utilization", "avg_wait_prev_7d_h", "throughput_prev_7d", "wind_kn", "wave_m"}


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    """Use the saved model; if it is missing or unreadable (e.g. different library versions), retrain it."""
    if not config.DATA_PATH.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate().to_csv(config.DATA_PATH, index=False)
    df = load_data()
    train_df, test_df = time_split(df)
    bundle, retrained = None, False
    if config.MODEL_PATH.exists():
        try:
            bundle = joblib.load(config.MODEL_PATH)
        except Exception:
            bundle = None
    if bundle is None:
        bundle, retrained = train_bundle(train_df, build_matrix), True
    return bundle, Explainer(bundle), test_df, retrained


bundle, explainer, test_df, retrained = load_model()
S = st.session_state


def set_values(vals: dict) -> None:
    for k, v in vals.items():
        if k in INT_FIELDS:
            S[k] = int(v)
        elif k in FLOAT_FIELDS:
            S[k] = float(v)
        elif k == "storm_flag":
            S[k] = bool(v)
        else:
            S[k] = v


def on_preset() -> None:
    if S.preset != "Custom":
        set_values(SCENARIOS[S.preset])
        S.unk_queue = S.unk_hist = S.unk_weather = False
        S.loaded_call = None


def on_load_call() -> None:
    r = test_df[test_df["call_id"] == S.call_pick].iloc[0]
    base = dict(SCENARIOS["busy"])
    vals = {k: r[k] for k in RAW_NUMERIC if k not in ("berths", "cranes") and pd.notna(r[k])}
    set_values({**base, **vals})
    S.port = r["port_id"]
    S.unk_queue = bool(pd.isna(r["queue_length"]))
    S.unk_hist = bool(pd.isna(r["avg_wait_prev_7d_h"]))
    S.unk_weather = bool(pd.isna(r["wind_kn"]))
    S.preset = "Custom"
    S.loaded_call = dict(call_id=r["call_id"], wait=float(r["wait_hours"]), congested=int(r["congested"]))


if "init" not in S:
    set_values(SCENARIOS["busy"])
    S.update(init=True, preset="busy", port="PORT_C", unk_queue=False, unk_hist=False, unk_weather=False, loaded_call=None)

st.title("Port Congestion Risk Estimator")
st.caption(
    f"Estimates the probability that an arriving vessel waits more than {config.CONGESTION_THRESHOLD_H:.0f} hours for a berth. "
    "Trained on synthetic data (queueing simulation), so treat it as a demonstration of the method, not an operational tool."
)
if retrained:
    st.info("No saved model was found, so one was trained just now from the included dataset.")

left, right = st.columns([5, 6], gap="large")

with left:
    st.selectbox("Start from a preset", ["Custom", "calm", "busy", "storm-peak"], key="preset", on_change=on_preset)
    with st.expander("Or load a real call from the held-out test period (to compare with the actual outcome)"):
        st.selectbox("Call", test_df["call_id"].tolist(), key="call_pick")
        st.button("Load this call", key="load_call", on_click=on_load_call)

    st.subheader("Port")
    port = st.selectbox("Port", list(config.PORTS), key="port")
    spec = config.PORTS[port]
    c1, c2 = st.columns(2)
    berths = c1.number_input("Berths", 1, 20, spec["berths"], key=f"berths_{port}")
    cranes = c2.number_input("Cranes", 1, 40, spec["cranes"], key=f"cranes_{port}")

    st.subheader("Traffic and port state")
    c1, c2 = st.columns(2)
    c1.number_input("Vessels already queued", 0, 60, step=1, key="queue_length", disabled=S.unk_queue)
    c2.number_input("Vessels due in next 24 h", 0, 60, step=1, key="arrivals_next_24h")
    c1.number_input("Vessels arrived in last 24 h", 0, 60, step=1, key="arrivals_last_24h")
    c2.number_input("Vessel size (TEU)", 800, 20000, step=500, key="vessel_teu")
    st.slider("Berth utilisation (share of berths in use)", 0.0, 1.0, step=0.05, key="berth_utilization")
    c1, c2 = st.columns(2)
    c1.number_input("Average wait, last 7 days (h)", 0.0, 200.0, step=1.0, key="avg_wait_prev_7d_h", disabled=S.unk_hist)
    c2.number_input("Throughput, last 7 days (vessels/day)", 0.0, 15.0, step=0.1, key="throughput_prev_7d", disabled=S.unk_hist)

    st.subheader("Weather")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Wind (kn)", 0.0, 70.0, step=1.0, key="wind_kn", disabled=S.unk_weather)
    c2.number_input("Wave height (m)", 0.0, 10.0, step=0.1, key="wave_m", disabled=S.unk_weather)
    c3.checkbox("Storm in progress", key="storm_flag", disabled=S.unk_weather)

    st.subheader("Arrival time")
    c1, c2, c3 = st.columns(3)
    c1.slider("Hour of day", 0, 23, key="hour")
    c2.selectbox("Weekday", list(range(7)), format_func=lambda i: WEEKDAYS[i], key="weekday")
    c3.selectbox("Month", list(range(1, 13)), format_func=lambda i: MONTHS[i - 1], key="month")

    st.subheader("Missing data")
    st.caption("Tick if a value is not available; the model handles gaps and says so.")
    c1, c2, c3 = st.columns(3)
    c1.checkbox("Queue unknown", key="unk_queue")
    c2.checkbox("7-day history unknown", key="unk_hist")
    c3.checkbox("Weather unknown", key="unk_weather")

# ---- build one input row ----------------------------------------------------------------------
vals = {k: S[k] for k in RAW_NUMERIC if k not in ("berths", "cranes", "storm_flag")}
vals["storm_flag"] = 1.0 if S.storm_flag else 0.0
if S.unk_queue:
    vals["queue_length"] = np.nan
if S.unk_hist:
    vals["avg_wait_prev_7d_h"] = vals["throughput_prev_7d"] = np.nan
if S.unk_weather:
    vals["wind_kn"] = vals["wave_m"] = vals["storm_flag"] = np.nan
vals.update(berths=berths, cranes=cranes, port_id=port)
X = build_matrix(pd.DataFrame([vals])[RAW_NUMERIC + ["port_id"]])
res = explainer.explain(X)[0]

# ---- results -------------------------------------------------------------------------------------
with right:
    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Congestion probability", f"{res['probability']:.0%}")
    m2.metric("Risk band", res["risk"])
    m3.metric("Confidence", f"{res['confidence']:.2f}", res["confidence_label"], delta_color="off")
    st.progress(float(res["probability"]))
    if res["predicted_congested"]:
        st.error("Alert: congestion expected for this call.")
    else:
        st.success("No alert: congestion not expected for this call.")
    lc = S.get("loaded_call")
    if lc:
        st.caption(
            f"Loaded call {lc['call_id']} (inputs may have been edited since). Actual outcome in the data: "
            f"wait {lc['wait']:.1f} h, {'congested' if lc['congested'] else 'not congested'}."
        )
    if res["out_of_range"]:
        st.warning("Outside the range seen in training: " + ", ".join(res["out_of_range"]) + ". Confidence is reduced.")
    if res["missing_inputs"]:
        st.info("Missing inputs (filled with training medians): " + ", ".join(res["missing_inputs"]) + ".")

    st.subheader("Why")
    fs = sorted(res["factors"], key=lambda f: f["contribution"])
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.barh([f["factor"].split(" (")[0] for f in fs], [f["contribution"] for f in fs],
            color=["#c0392b" if f["contribution"] > 0 else "#2e86c1" for f in fs])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Contribution to risk (log-odds): red raises, blue lowers")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if res["compound"]:
        st.warning("Heavy berth pressure and adverse weather are compounding the risk.")
    st.dataframe(
        pd.DataFrame([{"Factor": f["factor"], "Effect": f["direction"], "Contribution": round(f["contribution"], 2),
                       "Share of push": f"{f['share']:.0%}", "This call": f["detail"]} for f in res["factors"]]),
        hide_index=True, width="stretch",
    )

    st.subheader("Suggested actions")
    for a in res["actions"]:
        st.markdown(f"- {a}")

    with st.expander("How to read this"):
        st.markdown(
            f"- **Congested** means the vessel waits more than {config.CONGESTION_THRESHOLD_H:.0f} h for a berth.\n"
            "- **Probability** is calibrated: of calls rated about 70%, roughly 70% were congested in testing.\n"
            "- **Confidence** = probability of the predicted outcome, reduced when the 30 bootstrap models disagree "
            "or an input is outside the training range. High >= 0.70, Medium >= 0.50, otherwise Low.\n"
            "- **Contributions** are exact SHAP values on the log-odds scale, summed into five factors.\n"
            "- **Risk band:** HIGH at 60% or more, MODERATE at 30% or more, otherwise LOW. The **alert** uses a separate "
            f"threshold ({bundle.main.threshold:.2f}) tuned for F1 on training data.\n"
            "- Almost all predictive skill comes from the live queue state, so this is a short-horizon nowcast."
        )
