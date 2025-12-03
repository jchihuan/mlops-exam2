FROM python:3.11-slim AS base

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalamos solo dependencias necesarias para CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias SIN paquetes de desarrollo
RUN pip install --no-cache-dir --no-compile -r requirements.txt

# Copiar solo el código necesario
COPY src/ src/
COPY models/2025-12-01_17-29-26 models/

EXPOSE 8000

CMD ["python", "src/api.py"]
