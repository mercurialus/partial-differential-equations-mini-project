# make test | results | figures | report | slides | all | clean
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: all test results figures macros report slides clean

all: test figures report slides

test:
	$(PY) -m pytest -q

results/results.json:
	$(PY) scripts/run_experiments.py

results:
	$(PY) scripts/run_experiments.py

figures: results/results.json
	$(PY) scripts/make_figures.py

report/results.tex: results/results.json scripts/export_macros.py
	$(PY) scripts/export_macros.py

macros: report/results.tex

report: figures report/results.tex
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

slides: figures report/results.tex
	cd slides && latexmk -pdf -interaction=nonstopmode -halt-on-error slides.tex

clean:
	cd report && latexmk -C report.tex || true
	cd slides && latexmk -C slides.tex || true
	rm -f figures/*.pdf report/results.tex
	rm -rf .pytest_cache src/bsheat/__pycache__ tests/__pycache__ scripts/__pycache__
