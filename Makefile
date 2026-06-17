# VoltEdge — Makefile

.PHONY: test test-fast test-causal test-streaming test-ai test-carbon api dashboard install lint

install:
	pip install -r requirements.txt

# Full suite (~60 seconds, 895 tests)
test:
	python3 -m pytest tests/ \
	  --ignore=tests/test_models/test_lstm \
	  --ignore=tests/test_models/test_lstm_forecaster.py \
	  -v --tb=short

# Fast smoke test (~15 seconds)
test-fast:
	python3 -m pytest tests/ \
	  --ignore=tests/test_models/test_lstm \
	  --ignore=tests/test_models/test_lstm_forecaster.py \
	  --ignore=tests/test_ai \
	  --ignore=tests/test_streaming \
	  -q

# Module suites
test-causal:
	python3 -m pytest tests/test_causal/ -v

test-streaming:
	python3 -m pytest tests/test_streaming/ -v

test-ai:
	python3 -m pytest tests/test_ai/ -v

test-carbon:
	python3 -m pytest tests/test_carbon/ tests/test_demand_response/ tests/test_scope3_marketplace/ -v

# Services
api:
	uvicorn voltedge.api.app:app --reload --host 0.0.0.0 --port 8000

dashboard:
	python3 -m voltedge.dashboard.app

# Quality
lint:
	ruff check voltedge/ tests/
