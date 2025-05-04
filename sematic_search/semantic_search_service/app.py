import json
import pathlib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Semantic QA Search")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    results: list[str]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE = pathlib.Path(__file__).with_name("qa_data.json")

model = SentenceTransformer(MODEL_NAME)
with DATA_FILE.open(encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item["question"] for item in qa_data]
answers   = [item["answer"]   for item in qa_data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Empty query")

    query_vec = model.encode([req.query], convert_to_tensor=True)
    sims = cosine_similarity(query_vec, question_embeddings)[0]
    top_idx = np.argsort(-sims)[: req.top_k]
    results = [answers[i] for i in top_idx]
    return SearchResponse(results=results)
