# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Dependencies stage - install heavy ML dependencies first
FROM base AS dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch sentence-transformers

# Install project dependencies after stable deps
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# Pre-download and cache the sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Application stage
FROM base AS application
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin
COPY --from=dependencies /root/.cache/huggingface /root/.cache/huggingface

# Create data directory for Sara database
RUN mkdir -p /app/data

# Copy application code
COPY . .

# Sara package is already installed in dependencies stage

# Create non-root user for security
RUN groupadd -r sara && useradd -r -g sara sara && \
    chown -R sara:sara /app
USER sara

# Expose port
EXPOSE 8765

# Health check - reduced start period since model is pre-cached
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8765/health || exit 1

# Default command
CMD ["sara", "serve", "--host", "0.0.0.0", "--port", "8765"]