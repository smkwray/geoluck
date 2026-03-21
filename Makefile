SHELL := /bin/bash

.PHONY: sync test lint fmt web-install web-build web-preview-local check

VENV_PATH ?= /Users/shanewray/venvs/geoluck
VENV_PYTHON := $(VENV_PATH)/bin/python
VENV_PIP := $(VENV_PATH)/bin/pip
LOAD_ENV = if [ -f .env ]; then set -a; . ./.env; set +a; fi;

sync:
	$(LOAD_ENV) if command -v uv >/dev/null 2>&1; then uv sync; else "$(VENV_PIP)" install -e ".[dev]"; fi

test:
	$(LOAD_ENV) if command -v uv >/dev/null 2>&1; then uv run python -B -m pytest; else "$(VENV_PYTHON)" -B -m pytest; fi

lint:
	$(LOAD_ENV) if command -v uv >/dev/null 2>&1; then uv run ruff check .; else "$(VENV_PYTHON)" -m ruff check .; fi

fmt:
	$(LOAD_ENV) if command -v uv >/dev/null 2>&1; then uv run ruff format .; else "$(VENV_PYTHON)" -m ruff format .; fi

web-install:
	$(LOAD_ENV) cd web && npm install

web-build:
	$(LOAD_ENV) cd web && npm run build

web-preview-local:
	$(LOAD_ENV) python3 scripts/web_preview_local.py --port $${PORT:-4173}

check: test web-build
