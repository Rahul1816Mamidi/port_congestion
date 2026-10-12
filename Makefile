# Convenience wrapper. On Windows without make, run the python commands from README.md directly.
PY ?= python

.PHONY: all setup data train evaluate seeds report test demo app clean

all: data train evaluate report

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.generate_data

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

seeds:
	$(PY) -m src.seed_check

report: evaluate
	$(PY) report/build_report.py

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m src.predict --scenario calm --port PORT_C
	$(PY) -m src.predict --scenario busy --port PORT_C
	$(PY) -m src.predict --scenario storm-peak --port PORT_C
	$(PY) -m src.predict --call-id C0420

app:
	streamlit run app.py

clean:
	rm -f outputs/*.png outputs/*.json outputs/model.joblib
