# syntax=docker/dockerfile:1.6      # включает поддержку RUN --mount=type=cache
FROM python:3.10

#───────────────────────────────────── 1. APT ─────────────────────────────────────
# a) включаем встроенные повторы apt
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries

# b) устанавливаем пакеты с кеш-маунтами BuildKit
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt \
    apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

#────────────────────────────────── 2. Python deps ───────────────────────────────
WORKDIR /app
COPY requirements.txt .

# pip: 10 повторов, 60 с таймаут, общий кеш
ENV PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# библиотека Hub + быстрый клиент на Rust
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install "huggingface_hub[hf_transfer]>=0.23"

#─────────────────────── 3. Скачиваем модели при build ───────────────────────────


RUN python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("d0rj/e5-large-en-ru")
snapshot_download("BAAI/bge-reranker-large")
snapshot_download("Qdrant/bm25")
EOF


#──────────────────────────── 4. Код приложения ─────────────────────────────────
COPY ./app ./app
COPY faq_canonical_expanded.csv .
ENV PYTHONPATH=/app

# отключаем сеть у transformers на рантайме
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

#──────────────────────────── 5. Запуск ──────────────────────────────────────────
EXPOSE 9000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "9000"]
