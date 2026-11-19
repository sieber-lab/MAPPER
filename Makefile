PYTHON ?= python3

.PHONY: figures figure-inputs test figure-tests notebook-tests reports operator-intervals
figures:
	$(PYTHON) scripts/build_figures.py all

figure-inputs:
	PYTHONPATH=src $(PYTHON) scripts/check_inputs.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

figure-tests:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/figures -v

notebook-tests:
	$(PYTHON) scripts/check_notebooks.py

reports:
	$(PYTHON) scripts/recompute_results.py
	$(PYTHON) scripts/recompute_uncertainty.py
	$(PYTHON) scripts/recompute_significance_selection.py
	$(PYTHON) scripts/recompute_selection.py
	$(PYTHON) scripts/recompute_modalities.py
	$(PYTHON) scripts/recompute_preprocessing.py

operator-intervals:
	PYTHONPATH=src $(PYTHON) -m mapper.figures.operator_bootstrap --output results/operator_intervals
