# ============================================
# AI Research & Knowledge Assistant
# Production Dockerfile
# ============================================

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/chromadb /app/models

# Copy application code
COPY backend/ /app/backend/

# Expose Render's default web service port. Render also injects PORT at runtime.
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '10000')}/api/analytics/health\")" || exit 1

# Run the application
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}
