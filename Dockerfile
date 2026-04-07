# ── Inventory Restocking Decision System ─────────────────────────────────────
# HuggingFace Spaces Docker build — port 7860 required
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py                                          /app/models.py
COPY server/__init__.py                                 /app/server/__init__.py
COPY server/app.py                                      /app/server/app.py
COPY server/inventory_restock_env_environment.py        /app/server/inventory_restock_env_environment.py

RUN touch /app/__init__.py

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
