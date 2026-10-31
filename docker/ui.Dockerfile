# UI: Streamlit client. Talks to the API over HTTP and nothing else.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY frontend/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

COPY frontend/      /app/frontend/

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3).status==200 else 1)"

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
