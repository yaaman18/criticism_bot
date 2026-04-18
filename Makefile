VENV_BIN := $(abspath ./.venv/bin)
export PATH := $(VENV_BIN):$(PATH)
PYTHON ?= $(VENV_BIN)/python

.PHONY: bootstrap doctor test test-cov run ui harness-smoke dataset-smoke production-campaign production-finalize

bootstrap:
	./scripts/bootstrap_env.sh

doctor:
	$(PYTHON) -m trm_pipeline.experiment_harness doctor

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=trm_pipeline --cov-report=term-missing

run:
	$(PYTHON) anthropic_art_critic_chat.py

ui:
	./run_ui.sh

harness-smoke:
	./scripts/run_harness_smoke.sh

dataset-smoke:
	./scripts/run_dataset_smoke.sh

production-campaign:
	./scripts/run_production_campaign.sh

production-finalize:
	./scripts/finalize_production_campaign.sh
