.PHONY: build up down test train api clean inspect logs

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

# Run only the API service
api:
	docker compose run --rm api

# Run tests
test:
	docker compose run --rm train python -m kohonen.scripts.test_script

# Stop all services
down:
	docker compose down

# Clean MLflow data
clean:
	rm -rf mlflow_data/*

# Inspect MLflow data (optional run_id parameter)
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

help:
	@echo "Available commands:"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start all services"
	@echo "  make mlflow      - Start only MLflow server"
	@echo "  make train       - Run training"
	@echo "  make api         - Run API service"
	@echo "  make test        - Run tests"
	@echo "  make down        - Stop all services"
	@echo "  make clean       - Clean MLflow data"
	@echo "  make inspect     - Inspect MLflow data"
	@echo "  make logs        - View all logs"
	@echo "  make logs-mlflow - View MLflow logs"
	@echo "  make logs-train  - View training logs"
	@echo "  make logs-api    - View API logs" 