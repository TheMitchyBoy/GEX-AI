FROM python:3.12-slim

WORKDIR /app

# psycopg binary wheels + runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x scripts/start_web.sh scripts/start_dashboard.sh scripts/start_agent.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Railway sets PORT at runtime
EXPOSE 8000

CMD ["./scripts/start_web.sh"]
