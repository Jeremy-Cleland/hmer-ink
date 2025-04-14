# Makefile for HMER-Ink project

.PHONY: clean-pyc clean-outputs clean-all fast-clean lint lint-fix format typecheck check-all train train-fast evaluate test visualize-training watch-training dashboard expand-model train-expanded

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
# If the experiment name already exists, a version suffix will be added (-v1, -v2, etc.)
train:
	@if [ -z "$(EXPERIMENT)" ]; then \
		BASE_NAME="hmer-ink-m4-max"; \
	else \
		BASE_NAME="$(EXPERIMENT)"; \
	fi; \
	if [[ "$$BASE_NAME" == *-v* ]]; then \
		echo "Error: Please provide a base experiment name without version suffix (-v1, -v2, etc.)"; \
		exit 1; \
	fi; \
	VERSION=1; \
	while [ -d "outputs/$$BASE_NAME-v$$VERSION" ]; do \
		VERSION=$$((VERSION + 1)); \
	done; \
	EXPERIMENT_NAME="$$BASE_NAME-v$$VERSION"; \
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
	if [[ "$$BASE_NAME" == *-v* ]]; then \
		echo "Error: Please provide a base experiment name without version suffix (-v1, -v2, etc.)"; \
		exit 1; \
	fi; \
	VERSION=1; \
	while [ -d "outputs/$$BASE_NAME-v$$VERSION" ]; do \
		VERSION=$$((VERSION + 1)); \
	done; \
	EXPERIMENT_NAME="$$BASE_NAME-v$$VERSION"; \
	echo "Starting fast training with optimized settings for experiment: $$EXPERIMENT_NAME"; \
	python cli.py train --config configs/fasttraining.yaml --output-dir outputs/$$EXPERIMENT_NAME

# Evaluate model
# Usage: make evaluate MODEL=path/to/model/checkpoints/best_model.pt [CONFIG=configs/custom.yaml] [SPLIT=test]
evaluate:
	@if [ -z "$(MODEL)" ]; then \
		# Try to find the most recent model's best checkpoint \
		LATEST_MODEL=$$(find outputs/models -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2); \
		if [ -n "$$LATEST_MODEL" ] && [ -f "$$LATEST_MODEL/checkpoints/best_model.pt" ]; then \
			MODEL="$$LATEST_MODEL/checkpoints/best_model.pt"; \
			echo "Auto-detected latest model at $$MODEL"; \
		else \
			echo "Error: MODEL parameter is required"; \
			echo "Usage: make evaluate MODEL=path/to/model/checkpoints/best_model.pt [CONFIG=configs/custom.yaml] [SPLIT=test]"; \
			exit 1; \
		fi; \
	fi; \
	if [ -z "$(CONFIG)" ]; then \
		# Try to use the model's saved config if available \
		MODEL_DIR=$$(dirname "$(MODEL)"); \
		MODEL_DIR=$$(dirname "$$MODEL_DIR"); \
		if [ -f "$$MODEL_DIR/config.yaml" ]; then \
			CONFIG_PATH="$$MODEL_DIR/config.yaml"; \
			echo "Using model's saved config at $$CONFIG_PATH"; \
		else \
			CONFIG_PATH="configs/default.yaml"; \
		fi; \
	else \
		CONFIG_PATH="$(CONFIG)"; \
	fi; \
	if [ -z "$(SPLIT)" ]; then \
		SPLIT="test"; \
	else \
		SPLIT="$(SPLIT)"; \
	fi; \
	# Determine output path \
	MODEL_DIR=$$(dirname "$(MODEL)"); \
	MODEL_DIR=$$(dirname "$$MODEL_DIR"); \
	OUTPUT_PATH="$$MODEL_DIR/evaluation_$$SPLIT.json"; \
	echo "Evaluating model $(MODEL) on $$SPLIT split using $$CONFIG_PATH"; \
	echo "Results will be saved to $$OUTPUT_PATH"; \
	python cli.py evaluate --model $(MODEL) --config $$CONFIG_PATH --split $$SPLIT --output $$OUTPUT_PATH

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
# Usage: make watch-training [METRICS_FILE=path/to/model/metrics/training_metrics.json] [REFRESH=300]
watch-training:
	@METRICS_FILE="outputs/models/latest/metrics/training_metrics.json"; \
	if [ ! -f "$$METRICS_FILE" ]; then \
		# Look for the most recent model directory if the default file doesn't exist \
		LATEST_MODEL=$$(find outputs/models -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2); \
		if [ -n "$$LATEST_MODEL" ]; then \
			METRICS_FILE="$$LATEST_MODEL/metrics/training_metrics.json"; \
			echo "Auto-detected latest model metrics at $$METRICS_FILE"; \
		fi; \
	fi; \
	if [ -n "$(METRICS_FILE)" ]; then \
		METRICS_FILE=$(METRICS_FILE); \
	fi; \
	REFRESH=300; \
	if [ -n "$(REFRESH)" ]; then \
		REFRESH=$(REFRESH); \
	fi; \
	if [ ! -f "$$METRICS_FILE" ]; then \
		echo "Error: Metrics file not found at $$METRICS_FILE"; \
		echo "Please specify a valid metrics file with METRICS_FILE=path/to/metrics.json"; \
		exit 1; \
	fi; \
	echo "Watching training metrics in $$METRICS_FILE (refreshing every $$REFRESH seconds)"; \
	python cli.py monitor watch --metrics-file $$METRICS_FILE --refresh $$REFRESH

# Launch interactive training dashboard
# Usage: make dashboard [METRICS_DIR=path/to/model/metrics] [PORT=8501]
dashboard:
	@METRICS_DIR="outputs/models/latest/metrics"; \
	if [ ! -d "$$METRICS_DIR" ]; then \
		# Look for the most recent model directory if the default doesn't exist \
		LATEST_MODEL=$$(find outputs/models -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2); \
		if [ -n "$$LATEST_MODEL" ]; then \
			METRICS_DIR="$$LATEST_MODEL/metrics"; \
			echo "Auto-detected latest model metrics at $$METRICS_DIR"; \
		fi; \
	fi; \
	if [ -n "$(METRICS_DIR)" ]; then \
		METRICS_DIR=$(METRICS_DIR); \
	fi; \
	PORT=8501; \
	if [ -n "$(PORT)" ]; then \
		PORT=$(PORT); \
	fi; \
	if [ ! -d "$$METRICS_DIR" ]; then \
		echo "Error: Metrics directory not found at $$METRICS_DIR"; \
		echo "Please specify a valid metrics directory with METRICS_DIR=path/to/metrics"; \
		exit 1; \
	fi; \
	echo "Launching training dashboard for $$METRICS_DIR on port $$PORT"; \
	python cli.py monitor dashboard --metrics-dir $$METRICS_DIR --port $$PORT

# Expand a model from a smaller architecture to a larger one
# Usage: make expand-model SRC=outputs/experiment_name/checkpoints/best_model.pt [DST=outputs/checkpoints/expanded_model.pt] [CONFIG=configs/fast_expanded.yaml] [DRY_RUN=true]
expand-model:
	@if [ -z "$(SRC)" ]; then \
		echo "Error: SRC parameter is required"; \
		echo "Usage: make expand-model SRC=path/to/source_model.pt [DST=path/to/expanded_model.pt] [CONFIG=configs/fast_expanded.yaml] [DRY_RUN=true]"; \
		exit 1; \
	fi; \
	if [ -z "$(DST)" ]; then \
		DST="outputs/checkpoints/expanded_model.pt"; \
	fi; \
	if [ -z "$(CONFIG)" ]; then \
		CONFIG="configs/fast_expanded.yaml"; \
	fi; \
	DRY_RUN_FLAG=""; \
	if [ "$(DRY_RUN)" = "true" ]; then \
		DRY_RUN_FLAG="--dry-run"; \
		echo "Previewing model expansion (dry run)..."; \
	else \
		echo "Expanding model from $(SRC) to $(DST) using $$CONFIG"; \
	fi; \
	python scripts/expand_model.py "$(SRC)" "$(DST)" --config "$$CONFIG" $$DRY_RUN_FLAG

# Train with the expanded model
# Usage: make train-expanded [CHECKPOINT=outputs/checkpoints/expanded_model.pt] [CONFIG=configs/fast_expanded.yaml] [EXPERIMENT=expanded_model_finetuning]
train-expanded:
	@if [ -z "$(CHECKPOINT)" ]; then \
		CHECKPOINT="outputs/checkpoints/expanded_model.pt"; \
	fi; \
	if [ -z "$(CONFIG)" ]; then \
		CONFIG="configs/fast_expanded.yaml"; \
	fi; \
	if [ -z "$(EXPERIMENT)" ]; then \
		EXPERIMENT="expanded_model_finetuning"; \
	fi; \
	echo "Training expanded model from $(CHECKPOINT) using $$CONFIG"; \
	python cli.py train --config "$$CONFIG" --checkpoint "$(CHECKPOINT)" --experiment "$$EXPERIMENT"