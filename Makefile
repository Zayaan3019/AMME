# AMME - Adaptive Market Making Engine
#
# `make help` lists everything. The two that matter most:
#   make up               bring up the full live stack
#   make regression-test  the gate that must pass before any change is done

SHELL := /bin/sh
PY := python
VENV := .venv
VENV_PY := $(VENV)/Scripts/python.exe
ifeq ($(OS),)
  VENV_PY := $(VENV)/bin/python
endif
COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env
SERVICES := ingestion-svc feature-svc deeplob-inference-svc quoting-engine-svc \
            risk-svc execution-sim-svc backtest-svc api-gateway-svc

export PYTHONIOENCODING := utf-8

.DEFAULT_GOAL := help
.PHONY: help venv install env lint format typecheck test test-unit test-property \
        determinism latency regression-test calibrate ablation coverage \
        build up down logs ps health restart clean train record smoke ci

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup --
$(VENV_PY):
	$(PY) -m venv $(VENV)

venv: $(VENV_PY) ## Create the virtualenv

install: venv ## Install shared package + dev tooling (editable)
	$(VENV_PY) -m pip install -q --upgrade pip setuptools wheel
	$(VENV_PY) -m pip install -q -e shared
	$(VENV_PY) -m pip install -q pytest pytest-cov pytest-asyncio hypothesis ruff mypy \
		websockets aiohttp httpx
	@echo "installed. run 'make test' to verify."

env: ## Create .env from the template if absent
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

# ----------------------------------------------------------- code health --
lint: ## ruff check
	$(VENV_PY) -m ruff check shared services tests train

format: ## ruff format + autofix
	$(VENV_PY) -m ruff format shared services tests train
	$(VENV_PY) -m ruff check --fix shared services tests train

typecheck: ## mypy --strict over the shared core
	$(VENV_PY) -m mypy shared/amme_shared shared/amme_core

# ---------------------------------------------------------------- tests --
test: ## Full test suite
	$(VENV_PY) -m pytest tests -q

test-unit: ## Unit tests only (hand-verified quant formulas)
	$(VENV_PY) -m pytest tests/unit -q

test-property: ## Property-based tests (hypothesis)
	$(VENV_PY) -m pytest tests/unit/test_properties.py -q

determinism: ## Byte-for-byte replay reproducibility (spec s.6.1)
	$(VENV_PY) -m pytest tests/regression/test_determinism.py -q \
		-k "Determinism or NoHidden"

latency: ## Per-stage latency budgets (spec s.6.4)
	$(VENV_PY) -m pytest tests/regression -q -m latency

coverage: ## Coverage over the quant/ML core
	$(VENV_PY) -m pytest tests -q --cov=shared/amme_core --cov=shared/amme_shared \
		--cov-report=term-missing --cov-report=html

## The gate. Everything spec s.6 requires, in one command.
regression-test: lint typecheck ## Lint + types + all tests + ablation report
	$(VENV_PY) -m pytest tests -q
	$(MAKE) ablation
	@echo ""
	@echo "=== regression suite PASSED ==="
	@echo "ablation report: artifacts/ablation_multiseed.md"

# -------------------------------------------------------------- research --
ablation: ## Multi-seed ablation -> artifacts/ + docs/RESULTS.md
	PYTHONPATH=services/backtest-svc $(VENV_PY) -m backtest_svc.cli ablation

calibrate: ## Grid-search gamma/kappa -> artifacts/calibration.md
	PYTHONPATH=services/backtest-svc $(VENV_PY) -m backtest_svc.cli calibrate

record: ## Record live Binance data to TimescaleDB for training
	PYTHONPATH=services/ingestion-svc $(VENV_PY) -m amme_train.record --minutes 60

train: ## Train DeepLOB and export TorchScript -> models/
	PYTHONPATH=train $(VENV_PY) -m amme_train.train

# ---------------------------------------------------------------- stack --
build: env ## Build all service images
	$(COMPOSE) build

up: env ## Start the full stack (live Binance -> paper quotes -> dashboard)
	$(COMPOSE) up -d
	@echo ""
	@echo "  dashboard   http://localhost:8000"
	@echo "  grafana     http://localhost:3000  (admin/admin)"
	@echo "  prometheus  http://localhost:9090"
	@echo ""
	@echo "  run 'make health' once containers have settled (~40s)"

down: ## Stop the stack
	$(COMPOSE) down

clean-volumes: ## Stop the stack AND delete recorded data
	$(COMPOSE) down -v

logs: ## Tail all service logs
	$(COMPOSE) logs -f --tail=100

ps: ## Container status
	$(COMPOSE) ps

restart: ## Restart application services (leaves redis/timescale up)
	$(COMPOSE) restart $(SERVICES)

health: ## Probe every service's /health endpoint
	@for p in 8000 8001 8002 8003 8004 8005 8006 8007; do \
	  printf "  :%s  " $$p; \
	  curl -s -o /dev/null -w "%{http_code}" http://localhost:$$p/health || echo -n "unreachable"; \
	  echo ""; \
	done

smoke: ## End-to-end smoke test against a running stack
	$(VENV_PY) -m pytest tests/integration -q

# ------------------------------------------------------------------- CI --
ci: lint typecheck test ## What CI runs on every PR

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} + 2>/dev/null || true
