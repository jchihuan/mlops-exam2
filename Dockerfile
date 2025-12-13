FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile -r requirements.txt

COPY src/ src/
COPY models/best models/

# Copiar script serve para SageMaker
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve
RUN sed -i 's/\r$//' /usr/local/bin/serve

ENV PORT=8080
EXPOSE 8080
