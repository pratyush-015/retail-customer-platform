# =============================================================================
# Backend image - multi-stage.
#
# Stage 1 builds wheels for every dependency; stage 2 installs those prebuilt
# wheels into a clean runtime. This keeps the compilers needed by pandas,
# numpy, scipy and psycopg2 out of the shipped image, which is both smaller and
# a smaller attack surface than a single-stage build.
#
# Build context is the repo root (see docker-compose.yml), not backend/, so the
# image can also carry the scripts/ directory the seed pipeline runs from.
#   docker build -f backend/Dockerfile -t retail-insights-api .
# =============================================================================

# ------------------------------------------------------------------ builder --
FROM python:3.12-slim AS builder

# gcc/g++ and the postgres client headers are needed to compile psycopg2 and
# any dependency without a manylinux wheel for this platform. They exist only
# in this stage.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copied alone so this layer is cached on the requirements hash: application
# code changes do not trigger a full dependency rebuild.
COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ------------------------------------------------------------------ runtime --
FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE : no .pyc litter in a layer that is never reused
# PYTHONUNBUFFERED        : logs reach `docker logs` immediately, not on flush
# PYTHONPATH              : makes `app` importable and lets scripts/ resolve it
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    PIP_NO_CACHE_DIR=1

# libpq5 is the runtime half of libpq-dev - psycopg2 links against it, so the
# slim image needs it even though the compiler does not ship.
# curl backs the compose healthcheck.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY backend/requirements.txt .
# --no-index: install strictly from the wheels built above, so the runtime
# stage never reaches the network and cannot silently resolve a different
# version than the one that was built.
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Non-root. Ownership covers the data and artifact directories because the ETL
# writes the SQLite file, the quality report and the model artifact at runtime.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/raw /app/backend/app/ml/artifacts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Hits the real health endpoint, which reports DB connectivity - so an API that
# is up but cannot reach Postgres is correctly reported unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

# Exec form: uvicorn becomes PID 1 and receives SIGTERM directly, so
# `docker stop` shuts down gracefully instead of waiting out the 10s timeout.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
