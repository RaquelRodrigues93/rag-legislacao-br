"""Busca densa (embeddings), lexical (BM25) ou híbrida (Reciprocal Rank Fusion ponderado).

As duas abordagens falham de jeitos diferentes:
- BM25 acerta termos exatos ("superendividamento", "ANPD"), mas não entende paráfrases
  ("meu nome ficou negativado" → "cadastros e bancos de dados de consumidores").
- Embeddings entendem paráfrases, mas podem confundir artigos vizinhos de vocabulário parecido.
Na avaliação com perguntas em linguagem leiga, a busca densa sozinha venceu a híbrida, por isso é
o padrão (ver README). O modo híbrido continua disponível, com peso do BM25 configurável.

Os resultados são agregados por **artigo**: se várias partes do mesmo artigo aparecem, ele conta
uma vez (na melhor posição) e o texto completo do artigo vai para o modelo.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from legisrag.config import settings
from legisrag.ingest import load_chunks
from legisrag.parser import Chunk

Mode = Literal["bm25", "dense", "hybrid"]

_STOPWORDS = set(
    """a ao aos as até com como da das de dela dele do dos e é em entre essa esse esta este
    eu foi for há isso isto já la lhe mais mas me mesmo meu minha na nas nem no nos nós o os ou
    para pela pelas pelo pelos por qual quando que quem se sem ser seu seus sua suas são só também
    te tem ter um uma umas uns você vocês posso pode podem sobre qualquer""".split()
)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", _strip_accents(text.lower()))
    out = []
    for t in tokens:
        if t in _STOPWORDS or len(t) < 2:
            continue
        # Stemming mínimo: remove plural simples ("consumidores" -> "consumidor").
        if len(t) > 5 and t.endswith("es") and t[-3] in "rsz":
            t = t[:-2]
        elif len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        out.append(t)
    return out


class Embedder:
    """Wrapper do sentence-transformers que aplica os prefixos exigidos pelos modelos E5."""

    def __init__(self, model_name: str = settings.embedding_model):
        self.model_name = model_name
        self._needs_prefix = "e5" in model_name.lower()

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer  # import pesado, só quando usar

        return SentenceTransformer(self.model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=len(texts) > 64
        ).astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" if self._needs_prefix else t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}" if self._needs_prefix else text])[0]


@dataclass
class Article:
    """Artigo completo (todas as partes reunidas) — a unidade entregue ao modelo."""

    article_id: str
    law_name: str
    law_number: str
    article: str
    hierarchy: str
    text: str
    url: str

    @property
    def citation_label(self) -> str:
        return f"{self.law_name}, {self.article}"


@dataclass
class SearchResult:
    article: Article
    score: float
    rank: int


def _merge_articles(chunks: list[Chunk]) -> dict[str, Article]:
    articles: dict[str, Article] = {}
    for c in chunks:
        if c.article_id not in articles:
            articles[c.article_id] = Article(
                c.article_id, c.law_name, c.law_number, c.article, c.hierarchy, c.text, c.url
            )
        else:
            # Partes seguintes repetem o caput na primeira linha; não duplicar.
            continuation = c.text.split("\n", 1)[1] if "\n" in c.text else c.text
            articles[c.article_id].text += "\n" + continuation
    return articles


def build_index(embedder: Embedder | None = None) -> None:
    chunks = load_chunks()
    embedder = embedder or Embedder()
    embeddings = embedder.embed_passages([c.index_text for c in chunks])
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(settings.index_dir / "embeddings.npy", embeddings)
    meta = {"embedding_model": embedder.model_name, "chunk_ids": [c.id for c in chunks]}
    (settings.index_dir / "meta.json").write_text(json.dumps(meta, indent=2))


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder):
        if len(chunks) != len(embeddings):
            raise ValueError("Índice desatualizado: rode `legisrag index` novamente.")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])
        self.articles = _merge_articles(chunks)

    @classmethod
    def load(cls) -> HybridRetriever:
        meta_path = settings.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("Índice não encontrado. Rode `legisrag index` primeiro.")
        meta = json.loads(meta_path.read_text())
        chunks = load_chunks()
        if [c.id for c in chunks] != meta["chunk_ids"]:
            raise ValueError("chunks.jsonl mudou desde a indexação: rode `legisrag index`.")
        embeddings = np.load(settings.index_dir / "embeddings.npy")
        return cls(chunks, embeddings, Embedder(meta["embedding_model"]))

    # --- rankings individuais (listas de índices de chunk, do melhor para o pior) ---

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores) if scores[i] > 0]

    def _dense_ranking(self, query: str) -> list[int]:
        sims = self.embeddings @ self.embedder.embed_query(query)
        return [int(i) for i in np.argsort(-sims)]

    def _to_article_ranking(self, chunk_ranking: list[int]) -> list[str]:
        seen: dict[str, None] = {}
        for i in chunk_ranking:
            seen.setdefault(self.chunks[i].article_id, None)
        return list(seen)

    def search(
        self, query: str, k: int = settings.top_k, mode: Mode | None = None
    ) -> list[SearchResult]:
        mode = mode or settings.search_mode  # type: ignore[assignment]
        rankings: list[list[str]] = []
        weights: list[float] = []
        if mode in ("dense", "hybrid"):
            rankings.append(self._to_article_ranking(self._dense_ranking(query)))
            weights.append(1.0)
        if mode in ("bm25", "hybrid"):
            rankings.append(self._to_article_ranking(self._bm25_ranking(query)))
            weights.append(settings.bm25_weight if mode == "hybrid" else 1.0)
        if not rankings:
            raise ValueError(f"Modo de busca inválido: {mode!r}")

        fused = reciprocal_rank_fusion(rankings, k=settings.rrf_k, weights=weights)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            SearchResult(article=self.articles[aid], score=score, rank=r + 1)
            for r, (aid, score) in enumerate(top)
        ]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """score(d) = Σ w_i / (k + rank_i(d)). Itens ausentes de um ranking não somam nada nele."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return scores
