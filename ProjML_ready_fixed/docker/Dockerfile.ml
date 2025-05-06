FROM python:3.9-slim
WORKDIR /app
COPY ProjMLKodiim/sematic_search/semantic_search_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ProjMLKodiim/sematic_search/semantic_search_service .
EXPOSE 50052
ENTRYPOINT ["python", "app.py"]
