"""Testa a API HTTP com um pipeline falso (sem índice e sem chamar o Claude)."""

import anthropic
import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from legisrag import api
from legisrag.generator import Answer, Citation, CredentialsError
from legisrag.retriever import Article, SearchResult


def _answer() -> Answer:
    art = Article(
        "cdc-art-49",
        "CDC",
        "Lei 8.078/1990",
        "Art. 49",
        "Capítulo VI",
        "O consumidor pode desistir do contrato, no prazo de 7 dias...",
        "http://x#art49",
    )
    return Answer(
        text="Sim, em até 7 dias.[1]",
        citations=[Citation(1, "CDC, Art. 49", "pode desistir do contrato", art.url)],
        sources=[SearchResult(article=art, score=0.9, rank=1)],
    )


class FakePipeline:
    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error, self.calls = result, error, []

    def ask(self, question, k):
        self.calls.append((question, k))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def use_pipeline(monkeypatch):
    def _set(pipeline):
        monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
        return TestClient(api.app)

    return _set


def test_health(use_pipeline):
    assert use_pipeline(FakePipeline()).get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_citations_and_sources(use_pipeline):
    pipeline = FakePipeline(_answer())
    resp = use_pipeline(pipeline).post("/ask", json={"question": "Posso devolver?", "k": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("Sim")
    assert body["abstained"] is False
    assert body["citations"][0] == {
        "source": 1,
        "label": "CDC, Art. 49",
        "cited_text": "pode desistir do contrato",
        "url": "http://x#art49",
    }
    assert body["sources"][0]["label"] == "CDC, Art. 49"
    assert pipeline.calls == [("Posso devolver?", 3)]


@pytest.mark.parametrize("payload", [{"question": "a"}, {"question": "ok ok", "k": 0}, {}])
def test_ask_validates_input(use_pipeline, payload):
    assert use_pipeline(FakePipeline(_answer())).post("/ask", json=payload).status_code == 422


def _api_error(cls, status: int):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("erro", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (CredentialsError(), 503),
        (_api_error(anthropic.RateLimitError, 429), 429),
        (_api_error(anthropic.InternalServerError, 500), 502),
    ],
)
def test_ask_maps_errors_to_http_status(use_pipeline, error, status):
    resp = use_pipeline(FakePipeline(error=error)).post("/ask", json={"question": "Posso?"})
    assert resp.status_code == status
