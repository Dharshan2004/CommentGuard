PYTHON ?= python
.PHONY: test check public-results

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

check:
	$(PYTHON) scripts/check_repository.py

public-results:
	$(PYTHON) scripts/export_public_results.py
