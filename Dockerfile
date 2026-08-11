FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY flows ./flows
RUN pip install --no-cache-dir .

ENV SQMS_AI_HOST=0.0.0.0
ENV SQMS_AI_PORT=8200
ENV SQMS_AI_PROJECT_ROOT=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8200
CMD ["uvicorn", "sqms_ai_orchestrator.main:app", "--host", "0.0.0.0", "--port", "8200"]
