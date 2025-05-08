from __future__ import annotations
"""REST‑API wrapper for the hybrid FAQ‑bot.

Prerequisites
-------------
1. Keep the original FAQ‑bot implementation in the *same* directory and name it
   ``faq_bot.py`` (the file you posted above, unchanged).
2. Install the extra runtime dependencies::

       pip install fastapi uvicorn[standard] pydantic

Running
-------
» Development mode (autoreload off)::

    python faq_api.py

» Or via *uvicorn* (preferred for production)::

    uvicorn faq_api:app --host 0.0.0.0 --port 9000 --workers 2

The service listens on **http://<host>:9000/ask** and expects JSON::

    {"text": "Ваш вопрос"}

It responds with::

    {"answer": "..."}
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from chatbot import FAQBot  # original bot implementation
except ModuleNotFoundError as exc:
    raise SystemExit(
        "❌ Не найден файл 'faq_bot.py' с классом FAQBot.\n"
        "   Положите исходный код бота рядом с faq_api.py либо скорректируйте импорт."
    ) from exc

log = logging.getLogger("FAQ‑API")

# ───────────────────────── API models ────────────────────────── #
class Query(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str

# ───────────────────── FastAPI initialisation ─────────────────── #

def create_app() -> FastAPI:
    bot = FAQBot()  # one shared instance for all requests

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # startup
        log.info("⚡️ FAQ‑API started, ready on port 9000")
        yield
        # shutdown
        bot.close()
        log.info("👋 FAQ‑API stopped – bot resources released")

    app = FastAPI(title="FAQ‑Bot API", version="1.0", lifespan=lifespan)

    @app.post("/ask", response_model=Answer)
    async def ask_endpoint(payload: Query):
        """Return the FAQ‑bot answer for the given text question."""
        question = payload.text.strip()
        if not question:
            raise HTTPException(status_code=422, detail="'text' must be a non‑empty string")
        answer_text = bot.ask(question)
        return {"answer": answer_text}

    return app

app = create_app()

# ─────────────────────── local dev entry‑point ────────────────── #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("faq_api:app", host="0.0.0.0", port=9000, reload=False)
