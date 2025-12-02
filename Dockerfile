
FROM python:3.11-slim

# 1.- Dependencias del sistema y limpieza de caché de apt :D!
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# 2.- Instalación de dependencias de Python
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KEDRO_PIPELINE="__default__"

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# 3.- Copiar el resto del proyecto
COPY . .

EXPOSE 8080

# 4.- Sincronización opcional de datos con DVC y ejecución del pipeline
CMD ["bash", "-c", "if command -v dvc >/dev/null 2>&1 && [ -f dvc.yaml ]; then dvc pull; fi && kedro run --pipeline=${KEDRO_PIPELINE}"]
