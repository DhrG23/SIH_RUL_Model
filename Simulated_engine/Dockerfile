FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY Simulated_engine/requirements.txt ./Simulated_engine/requirements.txt
RUN pip install --no-cache-dir -r Simulated_engine/requirements.txt

COPY Simulated_engine/ ./Simulated_engine/
COPY src/ ./src/
COPY models/ ./models/

WORKDIR /app/Simulated_engine
CMD ["sh", "-c", "exec uvicorn engine_simulator.ws_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
