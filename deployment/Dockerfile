# Multi-stage Dockerfile for AI Principles Gym

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Install production dependencies for PostgreSQL
RUN pip install --no-cache-dir --user \
    asyncpg==0.29.0 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    aioredis==2.0.1

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 gymuser && \
    useradd -r -u 1000 -g gymuser -m -s /bin/bash gymuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/gymuser/.local

# Copy application code
COPY --chown=gymuser:gymuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chown -R gymuser:gymuser /app

# Switch to non-root user
USER gymuser

# Add .local/bin to PATH
ENV PATH=/home/gymuser/.local/bin:$PATH

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
