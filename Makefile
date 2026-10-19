# Uses the active Python (e.g. `conda activate hsl`). Override with: make PY=/path/to/python ...
PY ?= python

# Load optional keys from .env (never committed).
-include .env
export

.PHONY: install data reports forecast classify test clean

install:
	$(PY) -m pip install -e ".[dev]"

data:          ## download ILI data (Delphi)
	$(PY) -m lab.data

reports:       ## Module 2: lead/lag HTML reports -> outputs/reports/index.html
	$(PY) -m lab.reports

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs/reports outputs/models
