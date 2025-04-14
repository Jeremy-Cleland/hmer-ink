# Makefile for HMER-Ink project

.PHONY: clean-pyc clean-outputs clean-all lint lint-fix format typecheck check-all train train-fast evaluate test visualize-training watch-training dashboard

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-outputs:
	@echo "Removing all model runs from outputs directory..."
	@echo "Removing all experiment directories and registry files..."
	rm -rf outputs/*
	@echo "Recreating directory structure..."
	@mkdir -p outputs/checkpoints
	@mkdir -p outputs/logs
	@mkdir -p outputs/registry
	@echo "{}" > outputs/registry/experiment_registry.json
	@echo "Outputs directory completely cleaned"

clean-all: clean-pyc clean-outputs
	@echo "All cleaning operations completed"

lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	ruff format .

typecheck:
	mypy .

check-all: lint format typecheck

# Train target with experiment naming support
# Usage: make train [EXPERIMENT=name] [CONFIG=configs/custom.yaml]
# If no experiment name is provided, "hmer-ink" will be used
# If the experiment name already exists, a version suffix will be added (_v1, _v2, etc.)
train:
	@if [ -z "$(EXPERIMENT)" ]; then \
		BASE_NAME="hmer-ink"; \
	else \
		BASE_NAME="$(EXPERIMENT)"; \
	fi; \
	if [[ "$$BASE_NAME" == *_v* ]]; then \
		echo "Error: Please provide a base experiment name without version suffix (_v1, _v2, etc.)"; \
		exit 1; \
	fi; \
	VERSION=1; \
	while [ -d "outputs/registry/$$BASE_NAME_v$$VERSION" ]; do \
		VERSION=$$((VERSION + 1)); \
	done; \
	EXPERIMENT_NAME="$$BASE_NAME_v$$VERSION"; \
	echo "Starting training for experiment: $$EXPERIMENT_NAME"; \
	if [ -z "$(CONFIG)" ]; then \
		CONFIG_PATH="configs/default.yaml"; \
	else \
		CONFIG_PATH="$(CONFIG)"; \
	fi; \
	python cli.py train --config $$CONFIG_PATH --output-dir outputs/$$EXPERIMENT_NAME

# Fast train using optimized configuration for Apple Silicon
# Usage: make train-fast [EXPERIMENT=name]
train-fast:
	@if [ -z "$(EXPERIMENT)" ]; then \
		BASE_NAME="hmer-ink-fast"; \
	else \
		BASE_NAME="$(EXPERIMENT)-fast"; \
	fi; \
	if [[ "$$BASE_NAME" == *_v* ]]; then \
		echo "Error: Please provide a base experiment name without version suffix (_v1, _v2, etc.)"; \
		exit 1; \
	fi; \
	VERSION=1; \
	while [ -d "outputs/registry/$$BASE_NAME_v$$VERSION" ]; do \
		VERSION=$$((VERSION + 1)); \
	done; \
	EXPERIMENT_NAME="$$BASE_NAME_v$$VERSION"; \
	echo "Starting fast training with optimized settings for experiment: $$EXPERIMENT_NAME"; \
	python cli.py train --config configs/fasttraining.yaml --output-dir outputs/$$EXPERIMENT_NAME

# Evaluate model
# Usage: make evaluate MODEL=outputs/experiment_name/checkpoints/best_model.pt [CONFIG=configs/custom.yaml] [SPLIT=test]
evaluate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter is required"; \
		echo "Usage: make evaluate MODEL=outputs/experiment_name/checkpoints/best_model.pt [CONFIG=configs/custom.yaml] [SPLIT=test]"; \
		exit 1; \
	fi; \
	if [ -z "$(CONFIG)" ]; then \
		CONFIG_PATH="configs/default.yaml"; \
	else \
		CONFIG_PATH="$(CONFIG)"; \
	fi; \
	if [ -z "$(SPLIT)" ]; then \
		SPLIT="test"; \
	else \
		SPLIT="$(SPLIT)"; \
	fi; \
	echo "Evaluating model $(MODEL) on $$SPLIT split using $$CONFIG_PATH"; \
	python cli.py evaluate --model $(MODEL) --config $$CONFIG_PATH --split $$SPLIT --output outputs/evaluation_$$SPLIT.json

# Run prediction on a single file
# Usage: make predict MODEL=path/to/model.pt INPUT=path/to/file.inkml [VISUALIZE=true]
predict:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter is required"; \
		echo "Usage: make predict MODEL=path/to/model.pt INPUT=path/to/file.inkml [VISUALIZE=true]"; \
		exit 1; \
	fi; \
	if [ -z "$(INPUT)" ]; then \
		echo "Error: INPUT parameter is required"; \
		echo "Usage: make predict MODEL=path/to/model.pt INPUT=path/to/file.inkml [VISUALIZE=true]"; \
		exit 1; \
	fi; \
	VIZ_FLAG=""; \
	if [ "$(VISUALIZE)" = "true" ]; then \
		VIZ_FLAG="--visualize"; \
	fi; \
	python cli.py predict --model $(MODEL) --input $(INPUT) $$VIZ_FLAG

# Visualize an InkML file
# Usage: make visualize INPUT=path/to/file.inkml [OUTPUT=path/to/output.pdf]
visualize:
	@if [ -z "$(INPUT)" ]; then \
		echo "Error: INPUT parameter is required"; \
		echo "Usage: make visualize INPUT=path/to/file.inkml [OUTPUT=path/to/output.pdf]"; \
		exit 1; \
	fi; \
	OUTPUT_FLAG=""; \
	if [ -n "$(OUTPUT)" ]; then \
		OUTPUT_FLAG="--output $(OUTPUT)"; \
	fi; \
	python cli.py visualize --input $(INPUT) $$OUTPUT_FLAG

# Run hyperparameter optimization with Weights & Biases
# Usage: make hpo [EXPERIMENT=name] [RUNS=10]
hpo:
	@if [ -z "$(EXPERIMENT)" ]; then \
		EXPERIMENT_NAME="hpo-hmer-ink"; \
	else \
		EXPERIMENT_NAME="hpo-$(EXPERIMENT)"; \
	fi; \
	if [ -z "$(RUNS)" ]; then \
		NUM_RUNS=10; \
	else \
		NUM_RUNS=$(RUNS); \
	fi; \
	echo "Starting hyperparameter optimization with $$NUM_RUNS runs under project $$EXPERIMENT_NAME"; \
	python scripts/hyperparameter_optimization.py --experiment $$EXPERIMENT_NAME --num-runs $$NUM_RUNS

# Generate a report of model metrics and visualizations
# Usage: make report MODEL=outputs/experiment_name/checkpoints/best_model.pt [OUTPUT=report.html]
report:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter is required"; \
		echo "Usage: make report MODEL=path/to/model.pt [OUTPUT=report.html]"; \
		exit 1; \
	fi; \
	OUTPUT_PATH="report.html"; \
	if [ -n "$(OUTPUT)" ]; then \
		OUTPUT_PATH=$(OUTPUT); \
	fi; \
	python scripts/generate_report.py --model $(MODEL) --output $$OUTPUT_PATH

# Extract training metrics from wandb and generate visualizations
# Usage: make visualize-training [WANDB_DIR=wandb] [OUTPUT_DIR=outputs/training_metrics]
visualize-training:
	@WANDB_DIR="wandb"; \
	if [ -n "$(WANDB_DIR)" ]; then \
		WANDB_DIR=$(WANDB_DIR); \
	fi; \
	OUTPUT_DIR="outputs/training_metrics"; \
	if [ -n "$(OUTPUT_DIR)" ]; then \
		OUTPUT_DIR=$(OUTPUT_DIR); \
	fi; \
	echo "Extracting training metrics from $$WANDB_DIR to $$OUTPUT_DIR"; \
	python cli.py monitor extract --wandb-dir $$WANDB_DIR --output-dir $$OUTPUT_DIR

# Watch training metrics and update visualizations
# Usage: make watch-training [METRICS_FILE=outputs/training_metrics/training_metrics.json] [REFRESH=300]
watch-training:
	@METRICS_FILE="outputs/training_metrics/training_metrics.json"; \
	if [ -n "$(METRICS_FILE)" ]; then \
		METRICS_FILE=$(METRICS_FILE); \
	fi; \
	REFRESH=300; \
	if [ -n "$(REFRESH)" ]; then \
		REFRESH=$(REFRESH); \
	fi; \
	echo "Watching training metrics in $$METRICS_FILE (refreshing every $$REFRESH seconds)"; \
	python cli.py monitor watch --metrics-file $$METRICS_FILE --refresh-rate $$REFRESH

# Launch interactive training dashboard
# Usage: make dashboard [METRICS_DIR=outputs/training_metrics] [PORT=8501]
dashboard:
	@METRICS_DIR="outputs/training_metrics"; \
	if [ -n "$(METRICS_DIR)" ]; then \
		METRICS_DIR=$(METRICS_DIR); \
	fi; \
	PORT=8501; \
	if [ -n "$(PORT)" ]; then \
		PORT=$(PORT); \
	fi; \
	echo "Launching training dashboard for $$METRICS_DIR on port $$PORT"; \
	python cli.py monitor dashboard --metrics-dir $$METRICS_DIR --port $$PORT