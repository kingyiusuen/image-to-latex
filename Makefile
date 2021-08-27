.PHONY: help
help:
	@echo "Commands:"
	@echo "install            : installs requirements."
	@echo "install-dev        : installs development requirements."
	@echo "venv               : sets up virtual environment for development."
	@echo "api                : launches FastAPI app."
	@echo "docker             : builds and runs a docker image."
	@echo "streamlit          : runs streamlit app."
	@echo "lint               : runs linting."
	@echo "clean              : cleans all unnecessary files."

# Installation
.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

# Set up virtual environment
.PHONY: venv
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	make install-dev

# Linting
.PHONY: lint
lint:
	isort .
	black .
	flake8 .
	mypy .

# API
.PHONY: api
api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir image-to-latex --reload-dir api

# Docker
.PHONY: docker
docker:
	docker build -t image-to-latex:latest -f api/Dockerfile .
	docker run -p 8000:8000 --name image-to-latex image-to-latex:latest

# Streamlit
.PHONY: streamlit
streamlit:
	streamlit run streamlit/app.py

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