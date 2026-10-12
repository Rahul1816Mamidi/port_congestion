# Port Congestion Predictor: Run Log and Results

Structured summary of a full pipeline run (`C:\Users\bhavy\Downloads\prt`, Windows PowerShell, `.venv`). All numbers are copied from the terminal output; section 9 lists observations drawn from them.

## 1. Commands run

| # | Command | Outcome |
|---|---|---|
| 1 | `python -m src.generate_data` | 450 rows written to `data\port_calls.csv` |
| 2 | `python -m src.train` | model saved to `outputs\model.joblib` |
| 3 | `python -m src.evaluate` | metrics, trade-off, per-port, robustness, global factors |
| 4 | `python -m src.seed_check` | stability across seeds 1-5 |
| 5 | `python -m pytest -q` | **13 passed** in 29.28 s |
| 6 | `python -m src.predict ...` | scenarios, custom inputs, dataset calls, `--json` |
| 7 | `python report/build_report.py` | `report\technical_report.pdf` |
| 8 | `streamlit run app.py` | app at `http://localhost:8501` |

## 2. Data

* **450 calls, 5 ports × 90 calls each**; overall congested share **40.7%**.
* Congested threshold used for outcomes: wait > **12 h**.

| Port | Calls | Congested share |
|---|---|---|
| PORT_A | 90 | 0.456 |
| PORT_B | 90 | 0.389 |
| PORT_C | 90 | 0.578 |
| PORT_D | 90 | 0.156 |
| PORT_E | 90 | 0.456 |

## 3. Training

| Item | Value |
|---|---|
| Split (chronological) | train **315** calls up to 2025-09-29 · test **135** calls from 2025-09-29 to 2025-12-31 |
| Congested share | train 40.6% · test 40.7% |
| Main-model alert threshold | **0.46** (F-beta on out-of-fold predictions) |
| Rule baseline | congested if `(queue + arrivals_24h) / berths >= 1.07` |

## 4. Test-set results (135 calls)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Alert rate | Brier |
|---|---|---|---|---|---|---|---|
| Rule (load ratio) | 0.828 | 0.811 | 0.761 | 0.636 | 0.693 | 0.341 | n/a |
| Logistic regression | **0.972** | **0.960** | **0.865** | **0.818** | **0.841** | 0.385 | **0.069** |
| Gradient boosting (main) | 0.961 | 0.949 | 0.830 | 0.800 | 0.815 | 0.393 | 0.086 |

* Main model AUC 95% bootstrap CI: **0.93-0.98**
* Brier of a constant baseline: **0.241** · ECE of main model: **0.085**

### Threshold trade-off (main model)

| Threshold | Precision | Recall | F1 | Alert rate |
|---|---|---|---|---|
| 0.20 | 0.79 | 0.91 | 0.85 | 0.47 |
| 0.30 | 0.80 | 0.87 | 0.83 | 0.44 |
| 0.40 | 0.82 | 0.84 | 0.83 | 0.41 |
| 0.50 | 0.84 | 0.78 | 0.81 | 0.38 |
| 0.60 | 0.91 | 0.76 | 0.83 | 0.34 |
| 0.70 | 0.93 | 0.75 | 0.83 | 0.33 |

### Confidence buckets

| Bucket | Calls | Accuracy | Brier |
|---|---|---|---|
| High | 96 | 1.000 | 0.0017 |
| Medium | 18 | 0.444 | 0.387 |
| Low | 21 | 0.524 | 0.215 |

### Per port

| Port | Calls | Positive rate | ROC-AUC | Recall at alert |
|---|---|---|---|---|
| PORT_A | 29 | 0.62 | 0.99 | 1.00 |
| PORT_B | 23 | 0.35 | 0.94 | 0.75 |
| PORT_C | 24 | 0.42 | 0.92 | 0.60 |
| PORT_D | 26 | 0.12 | 1.00 | 1.00 |
| PORT_E | 33 | 0.48 | 0.96 | 0.69 |

### Robustness and global factors

* **20% extra missing inputs:** mean AUC **0.945**, minimum **0.924**.
* **Global factor importance** (as printed):

| Factor | Score |
|---|---|
| Berth pressure (demand vs capacity) | 3.50 |
| Timing and seasonality | 0.42 |
| Recent port performance | 0.38 |
| Weather | 0.26 |
| Vessel and port characteristics | 0.21 |

## 5. Stability across seeds (1-5)

| Seed | Test positive rate | AUC rule | AUC logreg | AUC main | Brier main | Share high-conf | Accuracy high-conf |
|---|---|---|---|---|---|---|---|
| 1 | 0.437 | 0.862 | 0.973 | 0.969 | 0.089 | 0.541 | 0.973 |
| 2 | 0.526 | 0.857 | 0.964 | 0.939 | 0.115 | 0.496 | 0.970 |
| 3 | 0.356 | 0.903 | 0.977 | 0.972 | 0.066 | 0.719 | 0.990 |
| 4 | 0.363 | 0.909 | 0.970 | 0.975 | 0.057 | 0.674 | 0.989 |
| 5 | 0.474 | 0.845 | 0.961 | 0.937 | 0.113 | 0.644 | 0.977 |
| **Mean ± sd** | 0.431 ± 0.065 | 0.875 ± 0.026 | **0.969 ± 0.006** | 0.958 ± 0.017 | 0.088 ± 0.024 | 0.615 ± 0.083 | 0.980 ± 0.008 |

## 6. Predictor demonstrations

| Case | Command | Probability | Risk | Confidence | Alert | Top factor (log-odds) |
|---|---|---|---|---|---|---|
| Calm, PORT_C | `--scenario calm --port PORT_C` | 2% | LOW | 0.94 High | none | Berth pressure −3.69 (0 queued + 2 due vs 3 berths, ratio 0.67) |
| Busy, PORT_C | `--scenario busy --port PORT_C` | 97% | HIGH | 0.94 High | congestion expected | Berth pressure +5.29 (4 queued + 6 due, ratio 3.33) |
| Storm-peak, PORT_C | `--scenario storm-peak --port PORT_C` | 99% | HIGH | 0.75 High | congestion expected | Berth pressure +4.27 (ratio 6.00); Weather +1.69; flagged as compounding |
| Custom, PORT_B | `--port PORT_B --queue 6 --arrivals-24h 9 --wind 28 --wave 3 --storm` | 98% | HIGH | 0.93 High | congestion expected | Berth pressure +4.16 (ratio 2.50); Weather +1.87; compounding |
| Stress test | `--scenario busy --queue 500 --arrivals-24h 400` | 96% | HIGH | 0.73 High | congestion expected | Berth pressure +4.82 (ratio 300.00) |

Out-of-range inputs were reported under "Data notes": `demand_capacity_ratio` (storm-peak); `queue_length`, `arrivals_next_24h`, `demand_capacity_ratio` (stress test).

### Calls from the dataset

| Call | Port / arrival | Probability | Confidence | Alert | Actual outcome |
|---|---|---|---|---|---|
| C0400 | PORT_E, 2025-11-22 00:26 | 2% (LOW) | 0.77 High | none | wait 0.4 h, not congested ✔ |
| C0430 | PORT_E, 2025-12-16 17:25 | 11% (LOW) | 0.66 Medium | none | wait 13.0 h, **congested** ✘ |
| C0440 | PORT_E, 2025-12-21 21:46 | 19% (LOW) | 0.65 Medium | none | wait 35.4 h, **congested** ✘ |

## 7. JSON output (`--json`)

Fields returned for the `busy` scenario: `probability` (0.969), `risk` (HIGH), `confidence` (0.936), `confidence_label`, `predicted_congested`, `ensemble_std` (0.0103), `out_of_range`, `missing_inputs`, `factors[]`, `actions[]`, `compound`.

Each entry in `factors[]` has: `factor`, `contribution` (log-odds), `detail`, `top_features` (feature, value pairs), `share`, `direction` ("raises risk" / "lowers risk").

| Factor (busy scenario) | Contribution | Share |
|---|---|---|
| Berth pressure (demand vs capacity) | +5.293 | 0.823 |
| Recent port performance | +0.380 | 0.059 |
| Vessel and port characteristics | +0.368 | 0.057 |
| Timing and seasonality | +0.337 | 0.052 |
| Weather | −0.052 | 0.008 |

## 8. CLI reference (`python -m src.predict --help`)

| Option | Purpose |
|---|---|
| `--scenario {calm,busy,storm-peak}` | starting values (default `busy`) |
| `--port {PORT_A..PORT_E}` | choose the port |
| `--call-id ID` | explain an existing dataset call and show its actual outcome |
| `--arrival "YYYY-MM-DD HH:MM"` | sets hour, weekday and month |
| `--storm` | storm in progress |
| `--json` | machine-readable output |
| `--teu --berths --cranes --queue --utilization` | vessel and port overrides |
| `--arrivals-24h --arrivals-last-24h --avg-wait-7d --throughput-7d` | traffic and history overrides |
| `--wind --wave --hour --weekday --month` | weather and timing overrides |

## 9. Observations from these results

**Strengths**
* Every model clearly beats the rule baseline (AUC 0.83-0.91 across seeds vs 0.94-0.98).
* High-confidence predictions were right 98% of the time across seeds (100% in the main run), covering about 62% of calls.
* AUC held at 0.945 mean with 20% extra missing inputs.
* The explanations are readable and consistent: berth pressure dominates, and weather becomes a visible second factor in the storm cases, with the compounding message.

**Points to state or check**
* **Logistic regression matches or beats the main model** (test AUC 0.972 vs 0.961; seed means 0.969 ± 0.006 vs 0.958 ± 0.017; Brier 0.069 vs 0.086). The report should say plainly why the gradient-boosting model is the main one.
* **The test set is small** (135 calls), hence the wide CI of 0.93-0.98 and port-level cells of 23-33 calls.
* **Confidence labels are not monotonic:** Medium accuracy (0.444, n = 18) is below Low (0.524, n = 21). The sample is small, but it weakens the claim that the label ranks reliability beyond the High bucket.
* **Out-of-range inputs did not lower the label:** the storm-peak (0.75) and 500-vessel stress case (0.73) are both still labelled High, with the out-of-range note shown separately.
* **Missed congestions:** C0430 (11%, actual 13.0 h, just over the 12 h threshold) and C0440 (19%, actual 35.4 h), both PORT_E in December. Recall at the alert threshold is lowest for PORT_C (0.60) and PORT_E (0.69).
* **Calibration:** ECE 0.085 for the main model, so probabilities such as "97%" should be described as scores unless a calibration plot supports them.
* **Presentation glitches:** the text output of the action reads "gang shiftif labour allows" (the JSON has "shift if"), and a custom run with `--port PORT_B` prints the header "scenario 'busy'".
* **Global weather importance is the lowest but one** (0.26) even though weather is the clear second factor in storm cases. This is worth one sentence in the report, since it matches an interaction pattern (weather matters mostly when berth pressure is already high).
