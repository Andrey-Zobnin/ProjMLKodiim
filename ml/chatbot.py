#!/usr/bin/env python
# coding: utf‑8
"""
FAQ‑бот «Пятёрочка» (v2, 2025‑05‑05)
Изменения:
• мягче пороги (FIRST/SECOND, CE_ACCEPT/CE_FLOOR)
• адаптивный порог от 35‑го перцентиля
• очистка запроса от геотегов (Россия/Казахстан/…)
• удаление русских стоп‑слов при лемматизации
• расширенное TOP_K/CE_TOP_K
• do_sample=True, stop‑токен для TinyLlama
"""

import inspect
import os
import sys
import logging
import warnings
import re
from collections import namedtuple
from pathlib import Path

if not hasattr(inspect, "getargspec"):
    # патч для Py3.11+
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    def getargspec(func):
        f = inspect.getfullargspec(func)
        return ArgSpec(f.args, f.varargs, f.varkw, f.defaults)
    inspect.getargspec = getargspec

# ────────────────────────── базовые импорты ─────────────────────────
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer
from qdrant_client import QdrantClient, models
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words          # ← стоп‑слова

# ───────────────────────────── конфиг ───────────────────────────────
CSV      = "faq_canonical_expanded.csv"
COLL     = "x5_faq_prod_customer_v2"

EMB_NAME   = "d0rj/e5-large-en-ru"
Q_PREF, P_PREF = "query: ", "passage: "

CE_NAME    = "DiTy/cross-encoder-russian-msmarco"
CE_DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
CE_TOP_K   = 20                               # было 10

LLM_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

FIRST_THRESHOLD, SECOND_THRESHOLD = 0.72, 0.60  # было 0.78 / 0.68
FALLBACK_TOPK, TOP_K             = 25, 25        # шире окно
CE_ACCEPT, CE_FLOOR              = 0.45, 0.28    # было 0.48 / 0.32
LEMMA_MIN_OVERLAP                = 0.35
MAX_DIRECT_WORDS                 = 250           # длиннее лимит
LOG_LVL                          = logging.INFO

REFUSAL = (
    "К сожалению, я не нашёл точного ответа на ваш вопрос. "
    "Пожалуйста, обратитесь в службу поддержки «Пятёрочки»."
)

# ─────────────────────────── логирование ────────────────────────────
logging.basicConfig(
    level=LOG_LVL,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    stream=sys.stdout,
)
for noisy in (
    "httpx", "httpcore",
    "sentence_transformers",
    "transformers.generation_utils"
):
    logging.getLogger(noisy).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
log = logging.getLogger("FAQ-Bot")

# ─────────────────────────── утилиты ────────────────────────────────
morph = MorphAnalyzer()
_word_re = re.compile(r"[а-яёa-z]+", re.I)

STOP_RU = set(get_stop_words("russian"))

def lemmatize(text: str) -> set[str]:
    """Лемматизация + удаление стоп‑слов."""
    return {
        morph.parse(w)[0].normal_form
        for w in _word_re.findall(text.lower())
        if w not in STOP_RU
    }

COUNTRIES = r"(?:росси[ия]|рф|казахстан|кз|беларус[ьи]|рб|узбекистан|kg|кыргызстан)"
SPURIOUS_LOC_RE = re.compile(rf"\b(?:в|во)\s+{COUNTRIES}\b", re.I)

# ───────────────────────────── бот ───────────────────────────────────
class FAQBot:
    def __init__(self):
        log.info("🔄 инициализация бота…")

        # эмбеддер
        self.emb = SentenceTransformer(EMB_NAME).to(CE_DEVICE)
        self.dim = self.emb.get_sentence_embedding_dimension()
        log.info("📐 embedder %s, dim=%d", EMB_NAME, self.dim)

        # reranker
        self.ce = CrossEncoder(CE_NAME, device=CE_DEVICE)
        log.info("🔗 cross-encoder %s загружен на %s", CE_NAME, CE_DEVICE)

        # векторная БД
        self.qdr = QdrantClient("localhost", port=6333, timeout=90)
        self._ensure_collection()

        # LLM
        self.tok = AutoTokenizer.from_pretrained(LLM_NAME, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = pipeline(
            "text-generation",
            model=LLM_NAME,
            tokenizer=self.tok,
            device_map="auto",
            do_sample=True,              # включаем семплирование
            temperature=0.15,
            top_p=0.90,
            repetition_penalty=1.2,
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            ),
        )
        log.info("✅ бот готов")

    # ──────────────── работа с коллекцией ────────────────
    def _collection_exists(self) -> bool:
        return COLL in [c.name for c in self.qdr.get_collections().collections]

    def _ensure_collection(self):
        if not self._collection_exists():
            self.qdr.create_collection(
                COLL,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE
                )
            )
            log.info("🆕 создана коллекция %s", COLL)

        df = (
            pd.read_csv(CSV, dtype=str)
              .dropna(subset=["id", "question", "content"])
              .loc[:, ["id", "question", "content"]]
        )

        points, _ = self.qdr.scroll(COLL, with_payload=False, limit=1_000_000)
        existing = {p.id for p in points}

        new = df.loc[~df["id"].astype(int).isin(existing)].reset_index(drop=True)
        if new.empty:
            log.info("📦 коллекция актуальна, новых записей нет")
            return

        log.info("↗️ upsert %d новых Q‑A", len(new))
        texts = [
            f"{P_PREF}{row.question} {row.content[:512]}"
            for row in new.itertuples()
        ]
        vecs = self.emb.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        points = [
            models.PointStruct(
                id=int(row.id),
                vector=vecs[i].tolist(),
                payload={"question": row.question, "answer": row.content}
            )
            for i, row in enumerate(new.itertuples())
        ]
        for i in range(0, len(points), 128):
            self.qdr.upsert(COLL, points[i:i+128], wait=True)

        total = self.qdr.count(COLL, exact=True).count
        log.info("📦 коллекция обновлена, всего векторов: %d", total)

    # ──────────────── вспомогательные методы ────────────────
    def _clean_query(self, q: str) -> str:
        """Удаляем указание страны/региона, лишние пробелы."""
        q = SPURIOUS_LOC_RE.sub("", q)
        q = re.sub(r"\s{2,}", " ", q)
        return q.strip()

    def _adaptive_threshold(self, vec: list[float]) -> float:
        """Считаем динамический порог = 35‑перцентиль из топ‑20."""
        base = self.qdr.search(COLL, vec, limit=20)
        if not base:
            return FIRST_THRESHOLD
        scores = sorted(h.score for h in base)
        idx = int(len(scores) * 0.35)
        return max(SECOND_THRESHOLD, scores[idx])

    # ──────────────── основной публичный метод ────────────────
    def ask(self, raw_query: str) -> str:
        raw_query = raw_query.strip()
        if not raw_query:
            return REFUSAL

        log.info("💬 вопрос: %s", raw_query)

        query = self._clean_query(raw_query)
        q_vec = self.emb.encode(Q_PREF + query, convert_to_tensor=False).tolist()

        dyn_thr = self._adaptive_threshold(q_vec)
        hits = (
            self.qdr.search(COLL, q_vec, limit=TOP_K, score_threshold=dyn_thr)
            or self.qdr.search(COLL, q_vec, limit=TOP_K, score_threshold=SECOND_THRESHOLD)
            or self.qdr.search(COLL, q_vec, limit=FALLBACK_TOPK)
        )
        if not hits:
            return REFUSAL

        # rerank
        ce_pairs = [
            (query, f"{h.payload['question']} {h.payload['answer']}")  # noqa: E501
            for h in hits[:CE_TOP_K]
        ]
        ce_scores = self.ce.predict(ce_pairs, convert_to_numpy=True, batch_size=16)
        best_s, best_hit = max(zip(ce_scores, hits[:CE_TOP_K]), key=lambda x: x[0])
        log.debug("🏷 id=%s cos=%.3f ce=%.3f", best_hit.id, best_hit.score, best_s)

        # фильтры качества
        if best_s < CE_ACCEPT:
            ql = lemmatize(query)
            hl = lemmatize(
                best_hit.payload["question"] + " " + best_hit.payload["answer"]
            )
            overlap = len(ql & hl) / max(1, len(ql))
            if best_s < CE_FLOOR or overlap < LEMMA_MIN_OVERLAP:
                return REFUSAL

        answer = best_hit.payload["answer"].strip()
        if len(answer.split()) <= MAX_DIRECT_WORDS:
            return answer

        # длинный ответ — резюмируем LLM на первых 1 024 символах
        context = answer[:1024]
        msgs = [
            {"role": "system", "content": "Ты — виртуальный помощник сети «Пятёрочка»."},
            {"role": "system", "content": f"КОНТЕКСТ:\n{context}"},
            {"role": "user",   "content": raw_query}
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = self.llm(
            prompt,
            max_new_tokens=120,
            stop=[self.tok.eos_token],
            pad_token_id=self.tok.pad_token_id,
            no_repeat_ngram_size=3,
            return_full_text=False
        )[0]["generated_text"].strip()

        return out or REFUSAL

    def close(self):
        self.qdr.close()


# ──────────────────────────── тесты ────────────────────────────
if __name__ == "__main__":
    bot = FAQBot()
    tests = [
        "Какие документы нужны для оформления командировки в Казахстан?",
        "мне нужно оформить командировку, как это сделать?",
        "Подскажите, в какие сроки перечислят выплату за мой ежегодный отпуск?"
    ]
    for q in tests:
        print("—" * 100)
        print("Вопрос:", q)
        print("Ответ :", bot.ask(q))
    bot.close()
