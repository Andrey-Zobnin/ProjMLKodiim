from typing import List
from fastapi import FastAPI, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from .faqbot import FAQBot

app = FastAPI(title="FAQ Black-Box")

bot: FAQBot | None = None        # lazy-init

class AskRequest(BaseModel):
    question: str

class Hit(BaseModel):
    answer: str

class AskResponse(BaseModel):
    hits: List[Hit]

async def get_bot() -> FAQBot:
    global bot
    if bot is None:
        bot = FAQBot()
    return bot

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, fbot: FAQBot = Depends(get_bot)):
    ans = await run_in_threadpool(fbot.ask, req.question)  # non-blocking :contentReference[oaicite:2]{index=2}
    return AskResponse(hits=[Hit(answer=ans)])

@app.get("/health")
async def health():
    return {"status": "ok"}
