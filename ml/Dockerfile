FROM python:3.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
EXPOSE 80
