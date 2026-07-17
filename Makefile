.PHONY: venv install install-dev frontend-install bootstrap models build test api web demo demo-ready seed seed-reset eval

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

test:
	TRACEFOLD_RUNTIME=fake $(PYTHON) -m pytest -q
	$(NPM) run build

api:
	cd backend && TRACEFOLD_RUNTIME=foundry $(PYTHON) -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

web:
	$(NPM) run dev -- --host 127.0.0.1

seed:
	TRACEFOLD_RUNTIME=foundry $(PYTHON) scripts/seed_demo.py

seed-reset:
	TRACEFOLD_RUNTIME=foundry $(PYTHON) scripts/seed_demo.py --reset

demo-ready:
	TRACEFOLD_RUNTIME=foundry $(PYTHON) scripts/seed_demo.py --reset --evaluate

demo: build
	cd backend && TRACEFOLD_RUNTIME=foundry $(PYTHON) -m uvicorn main:app --host 127.0.0.1 --port 8000

eval:
	TRACEFOLD_RUNTIME=foundry TRACEFOLD_DATA_DIR=$(CURDIR)/.tracefold-eval $(PYTHON) scripts/seed_demo.py --reset --evaluate
