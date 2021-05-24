# Makefile
#
# Reference:
# https://github.com/GokuMohandas/MLOps/blob/main/Makefile

.PHONY: help
help:
	@echo "Commands:"
	@echo "install          : installs requirements."
	@echo "install-dev      : installs development requirements."
	@echo "install-test     : installs test requirements."
	@echo "install-docs     : installs docs requirements."
	@echo "venv             : sets up the virtual environment for development."
	@echo "test             : runs all tests."
	@echo "test             : runs non-training tests."
	@echo "lint             : runs linting."
	@echo "clean            : cleans all unecessary files."

# Installation
.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

.PHONY: install-test
install-test:
	python -m pip install -e ".[test]" --no-cache-dir

.PHONY: install-docs
install-docs:
	python -m pip install -e ".[docs]" --no-cache-dir

# Set up virtual environment
# Usage: make venv name=venv env=dev
venv:
	python3 -m venv ${name}
	source ${name}/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	make install-$(env)

# Linting
.PHONY: lint
lint:
	isort .
	black .
	flake8 .
	mypy .

# Tests
.PHONY: test
test:
	pytest --cov image_to_latex --cov app --cov-report html

.PHONY: test-non-training
test-non-training:
	pytest -m "not training"

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . -type f -name ".coverage*" -ls -delete
	rm -rf htmlcov
	rm -rf .mypy_cache
