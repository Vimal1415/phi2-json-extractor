# ── Stage 1: Base image ──────────────────────────────────────
# We start from an official Python image.
# python:3.11-slim is Python 3.11 on minimal Linux (Debian).
# "slim" means no unnecessary packages — smaller image size.
FROM python:3.11-slim

# ── Stage 2: System dependencies ─────────────────────────────
# Some Python packages need system-level C libraries to compile.
# We install them here before installing Python packages.
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
# rm -rf cleans up apt cache — keeps image size small

# ── Stage 3: Working directory ───────────────────────────────
# All subsequent commands run from /app inside the container.
# This is the standard convention for containerized apps.
WORKDIR /app

# ── Stage 4: Install Python dependencies ─────────────────────
# We copy requirements.txt FIRST (before the rest of the code).
# Why: Docker caches each step. If requirements.txt hasn't changed,
# Docker reuses the cached layer and skips reinstalling packages.
# This makes rebuilds much faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir keeps image size small

# ── Stage 5: Copy application code ───────────────────────────
# Now copy the rest of the project.
# This comes AFTER pip install so code changes don't
# invalidate the expensive package installation cache.
COPY . .

# ── Stage 6: Environment variables ───────────────────────────
# Tell HuggingFace where to cache models inside the container.
# Without this, the model downloads to a random temp location.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# ── Stage 7: Expose port ──────────────────────────────────────
# Tell Docker this container listens on port 8000.
# This doesn't publish the port — it documents it.
# The actual port mapping happens when you run the container.
EXPOSE 8000

# ── Stage 8: Start command ────────────────────────────────────
# This runs when the container starts.
# host 0.0.0.0 means accept connections from outside the container.
# Without 0.0.0.0, the server only accepts internal connections
# and you can never reach it from your browser or from Render.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]