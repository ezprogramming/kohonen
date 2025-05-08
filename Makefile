.PHONY: build up down test train api clean inspect logs logs-mlflow logs-train logs-api help use-best use-specific fix-percent stop-all examples jupyter

# Build the Docker images
build:
	docker compose build

# Start all services
up:
	docker compose up -d

# Start specific service
mlflow:
	docker compose up -d mlflow

# Run only the training service
train:
	@if [ -z "$(TRAIN_ARGS)" ]; then \
		docker compose run --rm train python -m kohonen.scripts.train_script; \
	else \
		docker compose run --rm train python -m kohonen.scripts.train_script $(TRAIN_ARGS); \
	fi
	@echo "Adding newline to run_id.txt to prevent % character display issue..."
	@cat mlflow_data/run_id.txt | tr -d '\n' > mlflow_data/run_id.txt.tmp && echo "" >> mlflow_data/run_id.txt.tmp && mv mlflow_data/run_id.txt.tmp mlflow_data/run_id.txt

# Run only the API service (ensure MLflow is running first)
api:
	@echo "Ensuring MLflow is running..."
	@docker compose up -d mlflow
	@echo "Waiting for MLflow to be healthy..."
	@sleep 3
	@echo "Starting API service..."
	@docker compose up -d api
	@echo "API service started and accessible at: http://localhost:${PORT:-8000}"
	@echo "To see API logs: make logs-api"

# Configure API to use the best model (force best model selection)
use-best:
	@echo "Setting SOM_FORCE_BEST=true in environment..."
	@export SOM_FORCE_BEST=true
	@docker compose down api
	@SOM_FORCE_BEST=true docker compose up -d api
	@echo "API service now configured to use best model based on metrics"
	@echo "Metric: $${SOM_METRIC_KEY:-quantization_error}, Ascending: $${SOM_METRIC_ASCENDING:-true}"

# Configure API to use a specific model run ID
use-specific:
	@if [ -z "$(run_id)" ]; then \
		echo "Error: run_id parameter is required. Usage: make use-specific run_id=YOUR_RUN_ID"; \
		exit 1; \
	fi
	@echo "Setting API to use specific run ID: $(run_id)"
	@export SOM_FORCE_BEST=false
	@docker compose down api
	@echo "$(run_id)" > mlflow_data/run_id.txt
	@echo "" >> mlflow_data/run_id.txt
	@SOM_FORCE_BEST=false docker compose up -d api
	@echo "API service now configured to use run ID: $(run_id)"

# Fix the % display issue by adding a newline to run_id.txt
fix-percent:
	@echo "Fixing % display issue in run_id.txt..."
	@cat mlflow_data/run_id.txt | tr -d '\n' > mlflow_data/run_id.txt.tmp && echo "" >> mlflow_data/run_id.txt.tmp && mv mlflow_data/run_id.txt.tmp mlflow_data/run_id.txt
	@echo "Done!"

# Stop all Docker containers related to the project (including those outside docker-compose)
stop-all:
	@echo "Stopping all Docker Compose services..."
	@docker compose down
	@echo "Checking for any remaining Kohonen containers..."
	@CONTAINERS=$$(docker ps -a --filter "name=kohonen" --format "{{.ID}}"); \
	if [ -n "$$CONTAINERS" ]; then \
		echo "Found additional containers. Stopping and removing them..."; \
		docker stop $$CONTAINERS || true; \
		docker rm $$CONTAINERS || true; \
		echo "All remaining containers removed."; \
	else \
		echo "No additional containers found."; \
	fi

# Run tests
test:
	docker compose run --rm train python -m kohonen.scripts.test_script

# Stop all services
down:
	docker compose down

# Clean MLflow data
clean:
	rm -rf mlflow_data/*

# Inspect MLflow data 
# Usage: make inspect [run_id=<run_id>] to view details of a specific run
inspect:
	@if [ -z "$(run_id)" ]; then \
		docker compose run --rm train python -m kohonen.scripts.inspect_mlflow; \
	else \
		docker compose run --rm train python -m kohonen.scripts.inspect_mlflow --run-id=$(run_id); \
	fi

# View logs
logs:
	docker compose logs

# View logs for a specific service
logs-mlflow:
	docker compose logs mlflow

logs-train:
	docker compose logs train

logs-api:
	docker compose logs api

# Run example notebooks in Jupyter
jupyter:
	jupyter notebook examples/

# Open example notebooks in browser
examples: jupyter

# Display help information about Makefile commands
help:
	@echo "Available commands:"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start all services"
	@echo "  make mlflow      - Start only MLflow server"
	@echo "  make train       - Run training (add TRAIN_ARGS=\"--width 100\" for custom parameters)"
	@echo "  make api         - Run API service (automatically starts MLflow if not running)"
	@echo "  make use-best    - Configure API to use the best model (ignore run_id.txt)"
	@echo "  make use-specific run_id=YOUR_RUN_ID - Configure API to use a specific run ID"
	@echo "  make fix-percent - Fix the % display issue by adding a newline to run_id.txt"
	@echo "  make stop-all    - Stop and remove all Docker containers related to this project"
	@echo "  make test        - Run tests"
	@echo "  make down        - Stop all services"
	@echo "  make clean       - Clean MLflow data"
	@echo "  make inspect     - Inspect MLflow data (add run_id=<id> for specific run details)"
	@echo "  make logs        - View all logs"
	@echo "  make logs-mlflow - View MLflow logs"
	@echo "  make logs-train  - View training logs"
	@echo "  make logs-api    - View API logs"
	@echo "  make jupyter     - Start Jupyter Notebook server for example notebooks"
	@echo "  make examples    - Alias for jupyter command" 