# API: FastAPI + Postgres + Redis. No torch, no weights.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements-api.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

COPY backend/ /app/backend/

# A non-root user, because the data volume is writable and this container is
# the one exposed to the network.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data/cases /data/masks \
 && chown -R app:app /app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3).status==200 else 1)"

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
