import numpy as np
import pytest

from legisrag.evaluation import load_dataset, retrieval_metrics
from legisrag.parser import Chunk
from legisrag.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize


def test_tokenize_normalizes_accents_plurals_and_stopwords():
    tokens = tokenize("Os consumidores têm direitos básicos")
    assert tokens == ["consumidor", "direito", "basico"]


def test_rrf_rewards_agreement_between_rankings():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["a"] > scores["c"] > 0
    assert scores["d"] == pytest.approx(scores["c"])


def test_rrf_weights():
    scores = reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[1.0, 0.5])
    assert scores["a"] == pytest.approx(2 * scores["b"])


def test_retrieval_metrics():
    m = retrieval_metrics(["x", "a", "y", "b"], {"a", "b"}, ks=(1, 3))
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0
    assert m["recall@3"] == 0.5
    assert m["mrr"] == 0.5


class FakeEmbedder:
    """Embeddings determinísticos por palavra-chave, para testar sem baixar modelos."""

    model_name = "fake"
    VOCAB = ["dados", "consumidor", "multa"]

    def _vec(self, text: str) -> np.ndarray:
        v = np.array([text.lower().count(w) for w in self.VOCAB], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def _chunk(cid: str, article_id: str, text: str) -> Chunk:
    return Chunk(cid, article_id, "t", "Lei", "Lei 1", article_id, "", text, "http://x")


@pytest.fixture
def retriever() -> HybridRetriever:
    chunks = [
        _chunk("a1", "a1", "Proteção de dados pessoais e dados sensíveis."),
        _chunk("a2", "a2", "Direitos do consumidor."),
        _chunk("a2#p2", "a2", "Direitos do consumidor. […]\nmais direitos do consumidor."),
        _chunk("a3", "a3", "Multa de até dois por cento."),
        # Documentos extras: o BM25 Okapi zera o IDF de termos presentes em metade do corpus.
        *(_chunk(f"x{i}", f"x{i}", f"Disposição transitória número {i}.") for i in range(4)),
    ]
    emb = FakeEmbedder()
    return HybridRetriever(chunks, np.stack([emb._vec(c.index_text) for c in chunks]), emb)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_returns_articles_not_chunks(retriever, mode):
    results = retriever.search("direitos do consumidor", k=3, mode=mode)
    ids = [r.article.article_id for r in results]
    assert ids[0] == "a2"
    assert len(ids) == len(set(ids))  # partes do mesmo artigo agregadas


def test_article_parts_are_merged_without_repeating_caput(retriever):
    text = retriever.articles["a2"].text
    assert text == "Direitos do consumidor.\nmais direitos do consumidor."


def test_dataset_is_well_formed():
    items = load_dataset()
    assert len({it.id for it in items}) == len(items)
    assert sum(it.in_scope for it in items) >= 40
    assert any(not it.in_scope for it in items)
