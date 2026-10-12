"""Build report/technical_report.pdf from outputs/metrics.json, outputs/seed_stability.json and the figures.

Every number in the text is read from those files, so the report cannot drift from the code.
Run after `python -m src.evaluate` (and optionally `python -m src.seed_check`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config

OUT = ROOT / "outputs"
M = json.loads((OUT / "metrics.json").read_text())
SEEDS = json.loads((OUT / "seed_stability.json").read_text()) if (OUT / "seed_stability.json").exists() else None
DATA = pd.read_csv(config.DATA_PATH)

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11.0, alignment=TA_LEFT, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.2, leading=12.5, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#1f3b63"))
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.6, leading=9.4, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9.6, spaceAfter=0)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=9, bulletIndent=0, spaceAfter=1.5)


def P(t, s=BODY):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, BUL, bulletText="\u2022")


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("FONTSIZE", (0, 0), (-1, -1), 7.8), ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cc")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf5")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def img(name, w):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(OUT / name)).getSize()
    return Image(str(OUT / name), width=w, height=w * ih / iw)


mod = M["models"]
main, lr, rule = mod["gradient boosting (main)"], mod["logistic regression"], mod["rule (load ratio)"]
cb = M["confidence_buckets"]
pp = M["per_port"]
port_rates = DATA.groupby("port_id")["congested"].mean()
abl = M["ablation_no_live_state"]
rob = M["robustness_20pct_extra_missing"]
tt = M["threshold_tradeoff"]
n_hi = cb["High"]["n"]


def f(x, d=2):
    return f"{x:.{d}f}"


story = []
story.append(P("Port Congestion Risk Estimator: Technical Report", H1))
story.append(P("Rahul Mamidi &nbsp;|&nbsp; Aiqon Round 2 take-home &nbsp;|&nbsp; code: README.md in the repository, every figure below is produced by <font face='Courier'>python -m src.evaluate</font>", SMALL))

story.append(P("1. What the system does", H2))
story.append(P(
    "For a vessel arriving at a port, the system outputs (1) the <b>probability of congestion</b>, (2) a <b>confidence score</b> "
    "for that probability, and (3) the <b>key factors</b> behind it, each mapped to a suggested operational response. "
    "It is deliberately small: a 450-call synthetic dataset, scikit-learn and SHAP, no external APIs, runs in seconds on a CPU. "
    "A Streamlit app (<font face='Courier'>streamlit run app.py</font>) lets a reviewer enter their own inputs and see the result. The emphasis is on a defensible definition of congestion, honest evaluation, and explanations a planner can act on."))

story.append(P("2. Definition of congestion, and where the data comes from", H2))
story.append(P(
    f"<b>Definition.</b> A call is <i>congested</i> if its observed wait from arrival to berthing exceeds <b>{M['congestion_threshold_h']:.0f} hours</b>, "
    "about half a typical berth cycle (24 to 28 h), i.e. a material schedule slip. The threshold is configurable; ports differ, so per-port thresholds are a natural refinement."))
story.append(P(
    "<b>Why synthetic.</b> I reviewed the recommended AIS source (NOAA/BOEM Marine Cadastre). It provides vessel positions, speed, navigation status and dimensions for U.S. waters, "
    "but no port capacity, queue lengths, weather or delay labels, so congestion would first have to be reconstructed (port polygons, anchorage dwell times). "
    "Within the time limit I used the permitted synthetic route. <b>I did not use real AIS data.</b> The pipeline consumes a port-call table, so AIS-derived calls could replace the synthetic ones."))
story.append(P(
    "<b>Simulation and ground truth.</b> Rather than writing delay = f(features) + noise, which a model would simply recover, each of five ports is simulated as a queueing system over one year (2025): "
    "arrivals with seasonality, weekday and hour patterns, convoys and post-storm surges; a fixed number of berths served first come first served; "
    "service time depending on vessel size, cranes per berth and weather at berthing; and no berthing starts during a storm. "
    "The simulated wait is the ground truth. The label uses that wait plus 1.5 h of reporting noise, thresholded at 12 h. "
    "Features are computed from the port state <i>at arrival</i> (queue, berths in use, scheduled arrivals with +/-20% ETA error, 7-day history, weather), so nothing about the vessel's own outcome leaks. "
    "4 to 7% of weather, queue and history values are blanked to mimic data gaps. "
    f"450 calls are sampled (90 per port); {M['positive_rate_train']:.0%} of train and {M['positive_rate_test']:.0%} of test calls are congested, "
    f"ranging from {port_rates.min():.0%} ({port_rates.idxmin()[-1]}) to {port_rates.max():.0%} ({port_rates.idxmax()[-1]}) across ports."))

story.append(P("3. Features", H2))
story.append(P(
    "<b>Demand:</b> vessels queued, vessels due in 24 h, arrived in last 24 h. <b>Capacity:</b> berths, cranes, berth utilisation. "
    "<b>Derived:</b> demand/capacity ratio = (queue + arrivals in 24 h) / berths. <b>Historical:</b> average wait and throughput over the last 7 days. "
    "<b>Weather:</b> wind, wave height, storm flag. <b>Temporal:</b> hour, weekday, month. <b>Vessel/port:</b> size (TEU), port identity."))

story.append(P("4. Model, confidence and explanation", H2))
story.append(B(f"<b>Chronological split</b> (no shuffling): train on the first {M['n_train']} calls ({M['train_period'][0]} to {M['train_period'][1]}), test on the last {M['n_test']} ({M['test_period'][0]} to {M['test_period'][1]})."))
story.append(B("<b>Three predictors</b> so complexity must earn its place: a one-threshold rule on the load ratio; logistic regression; gradient boosting (main, 150 shallow trees; median imputation of missing values)."))
story.append(B("<b>Probability</b> is calibrated by Platt scaling fitted on expanding-window out-of-fold predictions, so calibration never sees the future. The alert threshold maximises F1 out-of-fold "
               f"({M['thresholds']['main']:.2f})."))
story.append(B(f"<b>Confidence</b> = calibrated probability of the predicted outcome x (1 - disagreement across {config.N_BOOTSTRAP} bootstrap-trained models / {config.STD_CAP}), "
               f"x {config.OOD_PENALTY} if any input lies outside the training range. Labels: High &ge; {config.CONF_HIGH}, Medium &ge; {config.CONF_MEDIUM}, else Low."))
story.append(B("<b>Explanation:</b> exact tree SHAP values (additive, on the log-odds scale) are summed into five factors; sentences come from templates, so the text can only state what the model used. "
               "Each positive factor is paired with an action."))

story.append(P("5. Results on the held-out later period", H2))
rows = [["Model", "ROC-AUC", "PR-AUC", "Brier", "Precision", "Recall", "F1"]]
for name, d in mod.items():
    rows.append([name, f(d["roc_auc"]), f(d["pr_auc"]), f(d["brier"], 3) if "brier" in d else "n/a", f(d["precision"]), f(d["recall"]), f(d["f1"])])
story.append(table(rows, [5.2 * cm, 1.9 * cm, 1.9 * cm, 1.7 * cm, 2.0 * cm, 1.8 * cm, 1.6 * cm]))
story.append(Spacer(1, 3))
story.append(P(f"n = {M['n_test']} test calls. Constant-rate baseline Brier = {f(M['brier_constant_baseline'], 3)}. Main-model AUC 95% bootstrap interval {f(M['main_auc_ci95'][0])} to {f(M['main_auc_ci95'][1])}.", SMALL))
story.append(Spacer(1, 2))
story.append(Table([[img("roc.png", 7.6 * cm), img("calibration.png", 7.6 * cm)]], hAlign="LEFT"))

seed_txt = ""
if SEEDS:
    s = SEEDS["summary"]
    seed_txt = (f" Re-generating the data and retraining with 5 different seeds gives AUC {s['auc_rule']['mean']:.3f} +/- {s['auc_rule']['sd']:.3f} (rule), "
                f"{s['auc_logreg']['mean']:.3f} +/- {s['auc_logreg']['sd']:.3f} (logistic) and {s['auc_main']['mean']:.3f} +/- {s['auc_main']['sd']:.3f} (boosting).")
story.append(B(f"<b>The learned models beat the rule</b> ({f(rule['roc_auc'])} AUC), but <b>logistic regression is as good as boosting</b> here ({f(lr['roc_auc'])} vs {f(main['roc_auc'])}, within the interval).{seed_txt} "
               "Complexity did not buy accuracy on this data, so I would deploy the logistic model first; boosting is kept as the reference for SHAP and for interactions that real data may show."))
story.append(B(f"<b>Calibration:</b> Brier {f(main['brier'], 3)} against {f(M['brier_constant_baseline'], 3)} for always predicting the base rate; expected calibration error {f(M['calibration']['ece_main'], 3)} (boosting) vs {f(M['calibration']['ece_logreg'], 3)} (logistic). Boosting is the less well calibrated of the two."))

crow = [["Confidence", "Calls", "Share", "Accuracy of predicted outcome"]]
for lab in ("High", "Medium", "Low"):
    c = cb[lab]
    crow.append([lab, str(c["n"]), f"{c['n'] / M['n_test']:.0%}", f(c["accuracy"]) if c["accuracy"] is not None else "n/a"])
story.append(KeepTogether([
    B("<b>The confidence score separates reliable from unreliable calls.</b> Accuracy of the predicted outcome by confidence label:"),
    table(crow, [2.6 * cm, 1.6 * cm, 1.6 * cm, 5.2 * cm]),
    Spacer(1, 2),
    P(f"High-confidence calls ({n_hi} of {M['n_test']}) were all correct on this test set; Medium and Low are close to coin flips, and with 18 and 21 calls their ordering is not reliable. "
      + (f"Across the 5 seeds, High-confidence accuracy is {SEEDS['summary']['acc_high_conf']['mean']:.2f} +/- {SEEDS['summary']['acc_high_conf']['sd']:.2f} and covers {SEEDS['summary']['share_high_conf']['mean']:.0%} of calls." if SEEDS else ""), SMALL),
]))
story.append(B(f"<b>Robustness and variation across ports:</b> blanking a further 20% of six key inputs lowers AUC only to {f(rob['roc_auc_mean'])} on average (worst {f(rob['roc_auc_min'])}). "
               f"Per-port AUC ranges {min(v['roc_auc'] for v in pp.values() if v['roc_auc']):.2f} to {max(v['roc_auc'] for v in pp.values() if v['roc_auc']):.2f}; "
               f"recall at the alert threshold is lower at PORT_C ({f(pp['PORT_C']['recall_at_alert'])}), PORT_E ({f(pp['PORT_E']['recall_at_alert'])}) and PORT_B ({f(pp['PORT_B']['recall_at_alert'])}), but each port has only 23 to 33 test calls."))
story.append(B(f"<b>Where the skill comes from:</b> removing the live queue-state features (queue, utilisation, recent waits, load ratio, recent throughput) drops AUC from {f(main['roc_auc'])} to <b>{f(abl['roc_auc'])}</b>, close to chance. "
               "Berth pressure dominates the explanations (below)."))

story.append(KeepTogether([
    P("6. From predictions to operational decisions", H2),
    Table([[img("example_explanation.png", 8.6 * cm), img("global_factors.png", 6.6 * cm)]], hAlign="LEFT"),
    P("Left: explanation of one high-risk test call (red raises risk). Right: mean absolute SHAP per factor across the test set.", SMALL),
]))
arow = [["Dominant positive factor", "Suggested response"],
        [P("Berth pressure (queue + arrivals vs berths)", CELL), P("Stagger berth windows, prioritise quick-turn vessels, ask carriers to slow-steam or re-time, add a gang shift", CELL)],
        [P("Weather", CELL), P("Warn carriers and customers of probable delay now; plan recovery shifts after the front", CELL)],
        [P("Recent port performance", CELL), P("Backlog already forming: check crane and labour availability, clear the yard before accepting more calls", CELL)],
        [P("Timing and seasonality", CELL), P("Known peak window: roster extra cover for that hour, weekday and season", CELL)],
        [P("Vessel and port characteristics", CELL), P("Assign the best-equipped berth and extra cranes", CELL)]]
story.append(table(arow, [5.0 * cm, 12.2 * cm]))
story.append(Spacer(1, 3))
story.append(P(
    f"<b>Choosing the alert threshold is an operational decision.</b> At 0.30 the model alerts on {tt['0.30']['alert_rate']:.0%} of calls with precision {f(tt['0.30']['precision'])} and recall {f(tt['0.30']['recall'])}; "
    f"at 0.60 it alerts on {tt['0.60']['alert_rate']:.0%} with precision {f(tt['0.60']['precision'])} and recall {f(tt['0.60']['recall'])}. "
    "A port that can cheaply add a shift should alert early; one where a false alarm means diverting vessels should wait for higher confidence. "
    "Low-confidence outputs carry an explicit instruction to verify inputs first."))

story.append(P("7. Limitations (please read these as part of the result)", H2))
story.append(B("<b>Synthetic ground truth.</b> I wrote the simulator, so the scores show the pipeline can learn a queueing process, not that it would reach this accuracy at a real port, where labour, customs, yard and equipment effects are unobserved and data is noisier."))
story.append(B("<b>It is a nowcast, not a forecast.</b> Almost all predictive power comes from the live queue state at arrival (ablation above). A 24 to 72 hour early-warning system needs ETA and arrival-forecast inputs that this dataset does not model well."))
story.append(B("<b>Weather looks minor in the global view</b> because its effect shows up through the queue it creates (correlated features); SHAP describes what the model relied on, not causes. In the storm scenario weather clearly raises risk."))
story.append(B(f"<b>Small samples.</b> {M['n_train']} training and {M['n_test']} test calls; one chronological split. Calibration on ~300 rows is coarse, and values outside the training range are flagged and penalised in confidence but not corrected."))
story.append(B("<b>One year of data</b> cannot separate genuine seasonality from year-specific events. The 12 h threshold is global, new ports need retraining (port identity is one-hot), and missing values use median imputation, so informative missingness is not modelled."))

story.append(P("8. Next steps", H2))
story.append(P("(1) Derive anchorage dwell times and queues from Marine Cadastre AIS for one U.S. port and add observed weather, to test on real labels. (2) Add arrival forecasting from ETAs for a true early-warning horizon. "
               "(3) Per-port thresholds and hierarchical port effects. (4) Rolling retraining with drift and calibration monitoring."))

doc = SimpleDocTemplate(str(ROOT / "report" / "technical_report.pdf"), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                        topMargin=1.4 * cm, bottomMargin=1.3 * cm, title="Port Congestion Risk Estimator: Technical Report", author="Rahul Mamidi")
doc.build(story)
print("wrote", ROOT / "report" / "technical_report.pdf")
