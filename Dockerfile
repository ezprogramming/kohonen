FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy files needed for building
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -e ".[mlflow,api,dev,demos]"

# Start with a fresh image for the final stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheels from builder stage
COPY --from=builder /app/wheels /app/wheels

# Copy source code
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Install the wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /app/wheels/*.whl && \
    pip install -e ".[dev,demos]"

# Create directory for comparison results
RUN mkdir -p /app/examples/comparison

# Create and set permissions for mlflow data directory
RUN mkdir -p /app/mlflow_data/artifacts && \
    chmod -R 777 /app/mlflow_data

# Create non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlflow_data
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Default command - runs API server
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn kohonen.api:app --host 0.0.0.0 --port $PORT"] 