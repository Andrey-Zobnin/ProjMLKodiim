from __future__ import annotations

import inspect
import logging
import re
import sys
import warnings
from collections import namedtuple
from typing import List, Tuple, Dict, Any

import os, torch
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words

# ——— fastembed (optional sparse)
try:
    from fastembed import SparseTextEmbedding

    _HAS_FASTEMBED = True
except ImportError:
    _HAS_FASTEMBED = False

# ——— monkey‑patch for Py≥3.11 (sentence‑transformers uses deprecated API)
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")


    def getargspec(func):  # type: ignore[misc]
        f = inspect.getfullargspec(func)
        return ArgSpec(f.args, f.varargs, f.varkw, f.defaults)


    inspect.getargspec = getargspec  # type: ignore[assignment]

# ——— config
CSV = "faq_canonical_expanded.csv"
COLL = "x5_faq_prod_customer_v3"

EMB_NAME = "d0rj/e5-large-en-ru"
Q_PREF, P_PREF = "query: ", "passage: "

CE_NAME = "     "
CE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CE_TOP_K = 20
TOP_N_AFTER_CE = 3

FIRST_THRESHOLD, SECOND_THRESHOLD = 0.72, 0.60
CE_ACCEPT, CE_FLOOR = 0.48, 0.30
NOUN_OVERLAP = 0.45
MAX_DIRECT_WORDS = 250
LOG_LVL = logging.INFO

REFUSAL = (
    "К сожалению, я не нашёл точного ответа на ваш вопрос. "
    "Переключаю на оператора X5"
)

# ——— logging
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

# ——— NLP helpers
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
    """Гибридный FAQ‑бот (dense + sparse)."""

    def __init__(self):
        log.info("🔄 инициализация бота…")

        # ① dense‑эмбеддер
        self.emb = SentenceTransformer(EMB_NAME, device=CE_DEVICE)
        self.dim = self.emb.get_sentence_embedding_dimension()
        log.info("📐 embedder %s, dim=%d", EMB_NAME, self.dim)

        # ② cross‑encoder for reranking
        self.ce = CrossEncoder(CE_NAME, device=CE_DEVICE)
        log.info("🔗 cross‑encoder %s загружен на %s", CE_NAME, CE_DEVICE)

        # ③ sparse‑энкодер (fastembed)
        self._has_sparse = False
        if _HAS_FASTEMBED:
            try:
                self.sparse_enc = SparseTextEmbedding(model_name="Qdrant/bm25")
                self._has_sparse = True
                log.info("🧩 sparse‑encoder fastembed/Qdrant‑bm25 готов")
            except Exception as e:
                log.warning("❗ fastembed не инициализировался: %s", e)
        else:
            log.info("ℹ️ fastembed не установлен")

        # ④ Qdrant
        qdrant_url = os.getenv("QDRANT__URL", "http://qdrant:6333")
        self.qdr = QdrantClient(url=qdrant_url, timeout=90)

        log.info("✅ бот готов")

    # --------------------------------------------------------------------- #
    #                            collection helpers                         #
    # --------------------------------------------------------------------- #
    def _collection_exists(self) -> bool:
        return COLL in [c.name for c in self.qdr.get_collections().collections]

    def _ensure_collection(self):
        """Создать или проверить коллекцию (dense + sparse) и при необходимости upsert‑нуть новые документы."""
        if not self._collection_exists():
            log.info("🆕 создаём коллекцию %s (dense + sparse)…", COLL)
            self.qdr.create_collection(
                collection_name=COLL,
                vectors_config={
                    "dense": models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
        else:
            info = self.qdr.get_collection(COLL)
            try:  # Qdrant ≥1.7
                sparse_cfg = info.config.params.sparse_vectors or {}
            except AttributeError:  # <1.7
                sparse_cfg = getattr(info, "sparse_vectors_config", {}) or {}
            if "sparse" not in sparse_cfg:
                log.warning("❗ Коллекция без sparse‑вектора → гибридный поиск отключён")
                self._has_sparse = False

        # -------- upsert missing documents --------
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

        dense_texts = [f"{P_PREF}{r.question} {r.content[:512]}" for r in new.itertuples()]
        dense_vecs = self.emb.encode(
            dense_texts, convert_to_tensor=False, show_progress_bar=False
        )

        # sparse encodings
        if self._has_sparse:
            texts = [r.question for r in new.itertuples()]
            if hasattr(self.sparse_enc, "embed_documents"):  # ≤0.3
                gen = self.sparse_enc.embed_documents(texts)
            else:  # ≥0.4
                gen = self.sparse_enc.embed(texts, is_query=False)
            sparse_embs = [(e.indices.tolist(), e.values.tolist()) for e in gen]
        else:
            sparse_embs = [([], [])] * len(new)

        up_points: List[models.PointStruct] = []
        for i, r in enumerate(new.itertuples()):  # ← исправление (itertuples)
            vectors: Dict[str, Any] = {"dense": dense_vecs[i].tolist()}
            if self._has_sparse:
                idx, val = sparse_embs[i]
                vectors["sparse"] = models.SparseVector(indices=idx, values=val)
            up_points.append(
                models.PointStruct(
                    id=int(r.id),
                    vector=vectors,  # type: ignore[arg-type]
                    payload={"question": r.question, "answer": r.content},
                )
            )
        for i in range(0, len(up_points), 128):
            self.qdr.upsert(COLL, up_points[i: i + 128], wait=True)

        log.info("📦 всего векторов: %d", self.qdr.count(COLL, exact=True).count)

    # --------------------------------------------------------------------- #
    #                               helpers                                 #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _clean_query(q: str) -> str:
        return re.sub(r"\s{2,}", " ", SPURIOUS_LOC_RE.sub("", q)).strip()

    def _adaptive_threshold(self, vec: List[float]) -> float:
        base = self.qdr.search(COLL, query_vector=("dense", vec), limit=20, with_payload=False)
        if not base:
            return FIRST_THRESHOLD
        scores = sorted(h.score for h in base)
        return float(max(SECOND_THRESHOLD, scores[int(len(scores) * 0.35)]))

    def _passes_noun_gate(self, qp: List[Tuple[str, str]], txt: str, ce: float) -> bool:
        if ce < CE_ACCEPT:
            return False
        qn, hn = extract_nouns(qp), extract_nouns(lemmatize(txt))
        if not qn:
            return ce >= CE_FLOOR
        return len(qn & hn) / len(qn) >= NOUN_OVERLAP

    def _best_paragraph(self, qp: List[Tuple[str, str]], ans: str) -> str:
        paras = [p.strip() for p in re.split(r"\n{2,}|\r?\n", ans) if p.strip()]
        if len(paras) == 1:
            return paras[0]
        qn = extract_nouns(qp)
        return max(paras, key=lambda p: len(qn & extract_nouns(lemmatize(p))) / max(len(qn), 1))

    # --------------------------------------------------------------------- #
    #                              main API                                 #
    # --------------------------------------------------------------------- #
    def ask(self, raw_q: str) -> str:
        raw_q = raw_q.strip()
        if not raw_q:
            return REFUSAL

        log.info("💬 вопрос: %s", raw_q)
        query = self._clean_query(raw_q)
        qp = lemmatize(query)
        q_vec = self.emb.encode(Q_PREF + query, convert_to_tensor=False).tolist()

        # sparse query vector (если есть fastembed)
        sparse_q = None
        if self._has_sparse:
            if hasattr(self.sparse_enc, "embed_query"):  # ≤0.3
                emb = self.sparse_enc.embed_query(query)
            else:  # ≥0.4
                emb = next(self.sparse_enc.embed([query], is_query=True))
            sparse_q = models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

        # -------- поиск --------
        try:
            if self._has_sparse and sparse_q and sparse_q.indices:
                resp = self.qdr.query_points(
                    collection_name=COLL,
                    prefetch=[
                        models.Prefetch(query=sparse_q, using="sparse", limit=50),
                        models.Prefetch(query=q_vec, using="dense", limit=50),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=CE_TOP_K,
                    with_payload=True,
                )
                hits = resp.points if hasattr(resp, "points") else resp
            else:
                hits = self.qdr.search(
                    COLL,
                    query_vector=("dense", q_vec),
                    limit=CE_TOP_K,
                    score_threshold=self._adaptive_threshold(q_vec),
                    with_payload=True,
                )
        except UnexpectedResponse as e:
            log.warning("Qdrant error %s → fallback dense", e)
            hits = self.qdr.search(
                COLL,
                query_vector=("dense", q_vec),
                limit=CE_TOP_K,
                score_threshold=self._adaptive_threshold(q_vec),
                with_payload=True,
            )

        if not hits:
            return REFUSAL

        # поддержка очень старых Qdrant (<1.5), где приходили (id, score) кортежи
        if hits and not hasattr(hits[0], "payload"):
            hits = [
                models.ScoredPoint(id=h[0], score=h[1], payload={}, vector={}) for h in hits
            ]

        # -------- rerank cross‑encoder --------
        ce_pairs = [(query, f"{h.payload['question']} {h.payload['answer']}") for h in hits]
        ce_scores = self.ce.predict(ce_pairs, convert_to_numpy=True, batch_size=16)
        ranked = sorted(zip(ce_scores, hits), key=lambda x: x[0], reverse=True)[: TOP_N_AFTER_CE]

        answer = None
        for s, h in ranked:
            doc = f"{h.payload['question']} {h.payload['answer']}"
            if self._passes_noun_gate(qp, doc, s):
                answer = h.payload["answer"].strip()
                break
        if answer is None:
            return REFUSAL

        # -------- формируем ответ --------
        if len(answer.split()) <= MAX_DIRECT_WORDS:
            return answer

        snippet = self._best_paragraph(qp, answer)
        # Логика LLM‑перефраза удалена — возвращаем лучший абзац как есть
        return snippet or answer or REFUSAL

    # --------------------------------------------------------------------- #
    #                               teardown                                #
    # --------------------------------------------------------------------- #
    def close(self):
        self.qdr.close()


# ——— quick test
if __name__ == "__main__":
    bot = FAQBot()
    try:
        while True:
            question = input("Ваш вопрос: ")
            print("-" * 100)
            print("Ответ :", bot.ask(question))
    except (KeyboardInterrupt, EOFError):
        print("\nЗавершаю работу бота.")
    finally:
        bot.close()
