.PHONY: venv install install-dev frontend-install bootstrap models build verify-deps lint test test-e2e release-check api web demo demo-ready seed seed-reset eval

PYTHON ?= $(CURDIR)/.venv/bin/python
NPM ?= npm --prefix frontend

venv:
	python3.11 -m venv .venv

install: venv
	$(PYTHON) -m pip install -r requirements.txt

install-dev: venv
	$(PYTHON) -m pip install -r requirements-dev.txt

frontend-install:
	$(NPM) ci

bootstrap: install-dev frontend-install

models:
	$(PYTHON) scripts/prepare_models.py

build:
	$(NPM) run build

verify-deps:
	$(PYTHON) -m pip check
	$(NPM) audit --omit=dev --audit-level=high

lint:
	$(PYTHON) -m ruff check backend scripts tests
	$(PYTHON) -m ruff format --check backend scripts tests
	$(NPM) run lint

test:
	TRACEFOLD_RUNTIME=fake $(PYTHON) -m pytest --cov=backend --cov-report=term-missing --cov-fail-under=75
	$(NPM) run test
	$(NPM) run build

test-e2e:
	$(NPM) run test:e2e

release-check: verify-deps lint test test-e2e

api:
	cd backend && TRACEFOLD_RUNTIME=foundry $(PYTHON) -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

web:
	$(NPM) run dev -- --host 127.0.0.1

seed:
	TRACEFOLD_RUNTIME=foundry $(PYTHON) scripts/seed_demo.py

seed-reset:
	TRACEFOLD_RUNTIME=foundry $(PYTHON) scripts/seed_demo.py --reset

demo-ready:
	TRACEFOLD_RUNTIME=foundry $(PYTHON) scripts/seed_demo.py --reset --evaluate --report-json docs/evaluation-report.json

demo: build
	cd backend && TRACEFOLD_RUNTIME=foundry $(PYTHON) -m uvicorn main:app --host 127.0.0.1 --port 8000

eval:
	TRACEFOLD_RUNTIME=foundry TRACEFOLD_DATA_DIR=$(CURDIR)/.tracefold-eval $(PYTHON) scripts/seed_demo.py --reset --evaluate --report-json docs/evaluation-report.json
