FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501 || exit 1

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
