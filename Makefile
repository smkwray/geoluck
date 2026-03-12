SHELL := /bin/bash
.ONESHELL:

.PHONY: sync test lint fmt web-install web-build check

define load_env
if [ -f .env ]; then
  set -a
  source ./.env
  set +a
fi
endef

sync:
	$(load_env)
	uv sync

test:
	$(load_env)
	uv run python -B -m pytest

lint:
	$(load_env)
	uv run ruff check .

fmt:
	$(load_env)
	uv run ruff format .

web-install:
	$(load_env)
	cd web
	npm install

web-build:
	$(load_env)
	cd web
	npm run build

check: test web-build

