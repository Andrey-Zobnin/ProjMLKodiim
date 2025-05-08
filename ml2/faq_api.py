from __future__ import annotations
"""REST‑API wrapper for the hybrid FAQ‑bot (FastAPI).

Fixes
-----
* **Однократная инициализация** моделей – `FAQBot` создаётся в событии *lifespan*,
  поэтому даже при импорте модуля несколькими воркерами дублирования нет.
* В режиме `python faq_api.py` передаём **готовый объект** в `uvicorn.run()` –
  это исключает повторный импорт.
* Импорт бота оставлен как `from chatbot import FAQBot` – переименуйте при
  необходимости.

Запуск
------
```bash
# production (без autoreload)
uvicorn faq_api:app --host 0.0.0.0 --port 9000 --workers 2

# development
python faq_api.py
```

HTTP‑интерфейс
--------------
`POST /ask`  – принимает JSON `{ "text": "…" }`, возвращает `{ "answer": "…" }`.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

try:
    # исходный класс бота
    from chatbot import FAQBot
except ModuleNotFoundError as exc:
    raise SystemExit(
        "❌ Не найден файл 'chatbot.py' с классом FAQBot.\n"
        "   Положите исходный код бота рядом с faq_api.py либо скорректируйте импорт."
    ) from exc

log = logging.getLogger("FAQ‑API")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

# ──────────────────────────── Pydantic models ──────────────────────────── #
class Query(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str

# ─────────────────────────────── FastAPI app ────────────────────────────── #

def create_app() -> FastAPI:
    """Factory that builds a FastAPI app with one shared FAQBot instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("🚀 Initialising FAQBot (this may take some time)…")
        app.state.bot = FAQBot()  # single shared instance
        log.info("⚡️ FAQ‑API started → ready on /ask")
        try:
            yield
        finally:
            app.state.bot.close()
            log.info("👋 FAQ‑API stopped – resources released")

    app = FastAPI(title="FAQ‑Bot API", version="1.1", lifespan=lifespan)

    # ───────────────────────────── endpoints ────────────────────────────── #

    @app.post("/ask", response_model=Answer)
    async def ask_endpoint(payload: Query):
        """Return the FAQ‑bot answer for the given text question."""
        question = payload.text.strip()
        if not question:
            raise HTTPException(422, "'text' must be a non‑empty string")
        return {"answer": app.state.bot.ask(question)}

    # optional fallback: allow raw text or malformed JSON
    @app.post("/ask_plain", response_model=Answer, include_in_schema=False)
    async def ask_plain(request: Request):
        raw = (await request.body()).decode().strip()
        if not raw:
            raise HTTPException(422, "Request body is empty")
        try:
            data = json.loads(raw)
            question = str(data.get("text", "")).strip()
        except json.JSONDecodeError:
            question = raw  # plain text
        if not question:
            raise HTTPException(422, "No question supplied")
        return {"answer": app.state.bot.ask(question)}

    return app

app = create_app()

# ───────────────────────────── local dev run ────────────────────────────── #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000, reload=False)
