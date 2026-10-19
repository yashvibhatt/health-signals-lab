# Uses the active Python (e.g. `conda activate hsl`). Override with: make PY=/path/to/python ...
PY ?= python

# Load optional keys from .env (never committed).
-include .env
export

.PHONY: install data reports forecast backtest train predict classify test clean

install:
	$(PY) -m pip install -e ".[dev]"

data:          ## download ILI data (Delphi)
	$(PY) -m lab.data

reports:       ## Module 2: lead/lag HTML reports -> outputs/reports/index.html
	$(PY) -m lab.reports

forecast: backtest train predict   ## Module 3: backtest, train, forecast next 4 weeks

backtest:
	$(PY) -m lab.forecast backtest

train:
	$(PY) -m lab.forecast train

predict:
	$(PY) -m lab.forecast predict

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs/reports outputs/models
