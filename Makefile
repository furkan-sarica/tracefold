.PHONY: venv install install-dev frontend-install frontend-browser bootstrap lock models build verify-deps lint test test-e2e release-check release-checksums release-artifacts release-ready api web demo demo-ready seed seed-reset eval

PYTHON ?= $(CURDIR)/.venv/bin/python
NPM ?= npm --prefix frontend
UV ?= uv
RUNTIME_LOCK ?= requirements-lock.txt
DEV_LOCK ?= requirements-dev-lock.txt

venv:
	python3.11 -m venv .venv

install: venv
	$(PYTHON) -m pip install --require-hashes -r $(RUNTIME_LOCK)

install-dev: venv
	$(PYTHON) -m pip install --require-hashes -r $(DEV_LOCK)

frontend-install:
	$(NPM) ci

frontend-browser:
	$(NPM) exec -- playwright install chromium

bootstrap: install-dev frontend-install frontend-browser

lock:
	$(UV) pip compile requirements.txt --universal --python-version 3.11 --generate-hashes --custom-compile-command 'make lock' --output-file $(RUNTIME_LOCK)
	$(UV) pip compile requirements-dev.txt --universal --python-version 3.11 --generate-hashes --custom-compile-command 'make lock' --output-file $(DEV_LOCK)

models:
	$(PYTHON) scripts/prepare_models.py

build:
	$(NPM) run build

verify-deps:
	$(PYTHON) -m pip check
	$(PYTHON) -m pip_audit --require-hashes --disable-pip --strict --progress-spinner off -r $(DEV_LOCK)
	$(NPM) audit --audit-level=high

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

release-checksums:
	$(PYTHON) scripts/check_release_artifacts.py --write-checksums

release-artifacts:
	$(PYTHON) scripts/check_release_artifacts.py

release-ready: release-check release-artifacts

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
