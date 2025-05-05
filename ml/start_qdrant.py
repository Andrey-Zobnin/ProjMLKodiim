#!/usr/bin/env python
# coding: utf-8
"""
FAQ‑бот «Пятёрочка» (v3.3, 2025‑05‑05)
────────────────────────────────────────
Изменения к v3.2 (Hybrid‑ready)
• Коллекция создаётся с двумя ИМЕНОВАННЫМИ векторами: dense и sparse.
• Добавлена интеграция с fastembed (BM25 по умолчанию) для генерации sparse‑векторов.
• При upsert сохраняем dense + sparse вектора одной точкой.
• Гибридный поиск через Query API (prefetch + RRF).
• Автоматический fallback на чистый dense‑поиск, если fastembed недоступен.
• Обновлены проверки совместимости с qdrant‑client ≥ 1.7.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import sys
import warnings
from collections import Counter, namedtuple
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, pipeline
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words

# ——— попытка подгрузить fastembed для sparse‑векторов
try:
    from fastembed import SparseTextEmbedding, SparseEmbedding

    _HAS_FASTEMBED = True
except ImportError:  # пакет не установлен
    _HAS_FASTEMBED = False

# ——— Monkey‑patch для Py3.11+ (sentence‑transformers опирается на deprecated API)
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):  # type: ignore[misc]
        f = inspect.getfullargspec(func)
        return ArgSpec(f.args, f.varargs, f.varkw, f.defaults)

    inspect.getargspec = getargspec  # type: ignore[assignment]

# ——— конфиг
CSV = "faq_canonical_expanded.csv"
COLL = "x5_faq_prod_customer_v3"

EMB_NAME = "d0rj/e5-large-en-ru"
Q_PREF, P_PREF = "query: ", "passage: "

CE_NAME = "BAAI/bge-reranker-large"
CE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CE_TOP_K = 20
TOP_N_AFTER_CE = 3

LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

FIRST_THRESHOLD, SECOND_THRESHOLD = 0.72, 0.60
CE_ACCEPT, CE_FLOOR = 0.48, 0.30
NOUN_OVERLAP = 0.45
MAX_DIRECT_WORDS = 250
LOG_LVL = logging.INFO

REFUSAL = (
    "К сожалению, я не нашёл точного ответа на ваш вопрос. "
    "Пожалуйста, обратитесь в службу поддержки «Пятёрочки»."
)

# ——— логирование
logging.basicConfig(
    level=LOG_LVL,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    stream=sys.stdout,
)
for noisy in (
    "httpx",
    "httpcore",
    "sentence_transformers",
    "transformers.generation_utils",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
log = logging.getLogger("FAQ-Bot")

# ——— NLP‑утилиты
morph = MorphAnalyzer()
_word_re = re.compile(r"[а-яёa-z]+", re.I)
STOP_RU = set(get_stop_words("russian"))


def lemmatize(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for w in _word_re.findall(text.lower()):
        if w in STOP_RU:
            continue
        p = morph.parse(w)[0]
        pairs.append((p.normal_form, p.tag.POS or ""))
    return pairs


def extract_nouns(pairs: List[Tuple[str, str]]) -> set[str]:
    return {lemma for lemma, pos in pairs if pos == "NOUN"}


COUNTRIES = r"(?:росси[ия]|рф|казахстан|кз|беларус[ьи]|рб|узбекистан|kg|кыргызстан)"
SPURIOUS_LOC_RE = re.compile(rf"\b(?:в|во)\s+{COUNTRIES}\b", re.I)


class FAQBot:
    """Чат‑бот, поддерживающий гибридный поиск (dense + sparse)."""

    def __init__(self):
        log.info("🔄 инициализация бота…")

        # ① dense‑эмбеддер
        self.emb = SentenceTransformer(EMB_NAME).to(CE_DEVICE)
        self.dim = self.emb.get_sentence_embedding_dimension()
        log.info("📐 embedder %s, dim=%d", EMB_NAME, self.dim)

        # ② cross‑encoder для rerank
        self.ce = CrossEncoder(CE_NAME, device=CE_DEVICE)
        log.info("🔗 cross‑encoder %s загружен на %s", CE_NAME, CE_DEVICE)

        # ③ sparse‑энкодер (optional)
        self._has_sparse = False
        if _HAS_FASTEMBED:
            try:
                # BM25 не требует GPU и весит <10 МБ
                self.sparse_enc = SparseTextEmbedding(model_name="Qdrant/bm25")
                self._has_sparse = True
                log.info("🧩 sparse‑encoder fastembed/BM25 готов")
            except Exception as e:
                log.warning("❗ не удалось инициализировать fastembed: %s", e)
                self._has_sparse = False
        else:
            log.info("ℹ️ fastembed не установлен; работаем без гибридного поиска")

        # ④ Qdrant
        self.qdr = QdrantClient(url="http://localhost:6333", timeout=90)
        self._ensure_collection()

        # ⑤ LLM для генерации кратких ответов
        self.tok = AutoTokenizer.from_pretrained(LLM_NAME, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = pipeline(
            "text-generation",
            model=LLM_NAME,
            tokenizer=self.tok,
            device_map="auto",
            do_sample=True,
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

    # ——— работа с коллекцией
    def _collection_exists(self) -> bool:
        return COLL in [c.name for c in self.qdr.get_collections().collections]

    def _ensure_collection(self):
        """Создать (или проверить) коллекцию с dense + sparse векторами."""
        if not self._collection_exists():
            log.info("🆕 создаём коллекцию %s (dense + sparse)…", COLL)
            self.qdr.create_collection(
                collection_name=COLL,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
        else:
            # быстрая проверка наличия sparse‑хранилища
            info = self.qdr.get_collection(COLL)
            if "sparse" not in info.sparse_vectors_config:
                log.warning(
                    "❗ Коллекция %s уже существует, но без sparse‑вектора. "
                    "Гибридный поиск работать не будет.",
                    COLL,
                )
                self._has_sparse = False

        # загрузим CSV и дольём недостающие точки
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

        dense_texts = [f"{P_PREF}{row.question} {row.content[:512]}" for row in new.itertuples()]
        dense_vecs = self.emb.encode(dense_texts, convert_to_tensor=False, show_progress_bar=False)

        sparse_embs: List[Tuple[List[int], List[float]]] = []
        if self._has_sparse:
            for emb in self.sparse_enc.embed_documents([row.question for row in new.itertuples()]):
                sparse_embs.append((emb.indices.tolist(), emb.values.tolist()))
        else:
            sparse_embs = [([], [])] * len(new)

        up_points: List[models.PointStruct] = []
        for i, row in enumerate(new.itertuples()):
            vectors: Dict[str, Any] = {"dense": dense_vecs[i].tolist()}
            if self._has_sparse:
                idx, val = sparse_embs[i]
                vectors["sparse"] = models.SparseVector(indices=idx, values=val)
            up_points.append(
                models.PointStruct(
                    id=int(row.id),
                    vector=vectors,  # type: ignore[arg-type]
                    payload={"question": row.question, "answer": row.content},
                )
            )

        for i in range(0, len(up_points), 128):
            self.qdr.upsert(COLL, up_points[i : i + 128], wait=True)

        total = self.qdr.count(COLL, exact=True).count
        log.info("📦 всего векторов: %d", total)

    # ——— utils
    @staticmethod
    def _clean_query(q: str) -> str:
        q = SPURIOUS_LOC_RE.sub("", q)
        q = re.sub(r"\s{2,}", " ", q)
        return q.strip()

    def _adaptive_threshold(self, vec: List[float]) -> float:
        base = self.qdr.search(COLL, query_vector=("dense", vec), limit=20)
        if not base:
            return FIRST_THRESHOLD
        scores = sorted(h.score for h in base)
        idx = int(len(scores) * 0.35)
        return float(max(SECOND_THRESHOLD, scores[idx]))

    def _passes_noun_gate(self, q_pairs: List[Tuple[str, str]], txt: str, ce_score: float) -> bool:
        if ce_score < CE_ACCEPT:
            return False
        qn = extract_nouns(q_pairs)
        hn = extract_nouns(lemmatize(txt))
        if not qn:
            return ce_score >= CE_FLOOR
        overlap = len(qn & hn) / len(qn)
        return overlap >= NOUN_OVERLAP

    def _best_paragraph(self, q_pairs: List[Tuple[str, str]], answer: str) -> str:
        paras = [p.strip() for p in re.split(r"\n{2,}|\r?\n", answer) if p.strip()]
        if len(paras) == 1:
            return paras[0]
        qn = extract_nouns(q_pairs)
        best, best_ol = paras[0], 0.0
        for p in paras:
            ol = len(qn & extract_nouns(lemmatize(p))) / max(1, len(qn))
            if ol > best_ol:
                best, best_ol = p, ol
        return best

    # ——— основной метод
    def ask(self, raw_query: str) -> str:
        raw_query = raw_query.strip()
        if not raw_query:
            return REFUSAL

        log.info("💬 вопрос: %s", raw_query)
        query = self._clean_query(raw_query)
        q_pairs = lemmatize(query)
        q_vec = self.emb.encode(Q_PREF + query, convert_to_tensor=False).tolist()

        # ① формируем sparse‑вектор запроса (если доступен)
        sparse_q = None
        if self._has_sparse:
            emb = self.sparse_enc.embed_query(query)
            sparse_q = models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

        # ② выполняем поиск
        hits = []
        try:
            if self._has_sparse and sparse_q and len(sparse_q.indices) > 0:
                # гибридный запрос через Query API
                hits = self.qdr.query_points(
                    collection_name=COLL,
                    prefetch=[
                        models.Prefetch(query=sparse_q, using="sparse", limit=50),
                        models.Prefetch(query=q_vec, using="dense", limit=50),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=CE_TOP_K,
                )
            else:
                # fallback → чистый dense
                hits = self.qdr.search(
                    COLL,
                    query_vector=("dense", q_vec),
                    limit=CE_TOP_K,
                    score_threshold=self._adaptive_threshold(q_vec),
                )
        except UnexpectedResponse as e:
            log.warning("Ответ Qdrant: %s — fallback на dense", e)
            hits = self.qdr.search(
                COLL,
                query_vector=("dense", q_vec),
                limit=CE_TOP_K,
                score_threshold=self._adaptive_threshold(q_vec),
            )

        if not hits:
            return REFUSAL

        # ③ rerank
        ce_pairs = [(query, f"{h.payload['question']} {h.payload['answer']}") for h in hits]
        ce_scores = self.ce.predict(ce_pairs, convert_to_numpy=True, batch_size=16)
        scored = sorted(zip(ce_scores, hits), key=lambda x: x[0], reverse=True)[: TOP_N_AFTER_CE]

        answer: str | None = None
        for best_s, best_hit in scored:
            doc_text = f"{best_hit.payload['question']} {best_hit.payload['answer']}"
            if self._passes_noun_gate(q_pairs, doc_text, best_s):
                answer = best_hit.payload["answer"].strip()
                break
        if answer is None:
            return REFUSAL

        # ④ формируем ответ
        if len(answer.split()) <= MAX_DIRECT_WORDS:
            return answer

        snippet = self._best_paragraph(q_pairs, answer)
        if len(snippet.split()) <= MAX_DIRECT_WORDS:
            return snippet

        context = snippet[:1024]
        msgs = [
            {
                "role": "system",
                "content": "Ты — виртуальный помощник сети «Пятёрочка». "
                "Сконцентрируйся на точном и кратком ответе.",
            },
            {"role": "system", "content": f"КОНТЕКСТ:\n{context}"},
            {"role": "user", "content": raw_query},
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = self.llm(
            prompt,
            max_new_tokens=120,
            stop=[self.tok.eos_token],
            pad_token_id=self.tok.pad_token_id,
            no_repeat_ngram_size=3,
            return_full_text=False,
        )[0]["generated_text"].strip()
        return out or snippet or REFUSAL

    def close(self):
        self.qdr.close()


# ——— мини‑тест
if __name__ == "__main__":
    bot = FAQBot()
    while True:
        test_q = input()
        print("—" * 100)
        print("Вопрос:", test_q)
        print("Ответ :", bot.ask(test_q))
    bot.close()
