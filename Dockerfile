
FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt ./


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KEDRO_PIPELINE="__default__"


EXPOSE 8080


CMD [
  "bash",
  "-c",
  "if command -v dvc >/dev/null 2>&1 && [ -f dvc.yaml ]; then dvc pull; fi && kedro run --pipeline ${KEDRO_PIPELINE}"
]