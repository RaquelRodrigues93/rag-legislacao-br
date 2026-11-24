"""Testa o tratamento da resposta do Claude sem chamar a API (cliente falso)."""

from types import SimpleNamespace as NS

from legisrag.generator import Generator
from legisrag.retriever import Article, SearchResult


def _result(aid: str, label: str) -> SearchResult:
    art = Article(aid, "CDC", "Lei 8.078/1990", label, "Capítulo VI", "texto", f"http://x#{aid}")
    return SearchResult(article=art, score=1.0, rank=1)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _client(response):
    messages = FakeMessages(response)
    return NS(beta=NS(messages=messages)), messages


def _usage():
    return NS(input_tokens=100, output_tokens=20)


def test_citations_are_mapped_to_sources_and_marked_in_text():
    cit = NS(type="char_location", document_index=1, cited_text=" O consumidor pode desistir ")
    response = NS(
        stop_reason="end_turn",
        usage=_usage(),
        content=[
            NS(type="thinking", thinking=""),
            NS(type="text", text="Sim, em até 7 dias.", citations=[cit]),
            NS(type="text", text=" Vale para compras online.", citations=None),
        ],
    )
    client, messages = _client(response)
    results = [_result("cdc-art-18", "Art. 18"), _result("cdc-art-49", "Art. 49")]

    ans = Generator(client=client, model="claude-opus-5-5").answer("posso devolver?", results)

    assert ans.text == "Sim, em até 7 dias.[2] Vale para compras online."
    assert ans.citations[0].label == "CDC, Art. 49"
    assert ans.citations[0].cited_text == "O consumidor pode desistir"
    assert ans.cited_article_ids == {"cdc-art-49"}
    assert not ans.abstained

    docs = messages.kwargs["messages"][0]["content"][:2]
    assert all(d["type"] == "document" and d["citations"] == {"enabled": True} for d in docs)
    assert messages.kwargs["fallbacks"] == "default"


def test_abstention_is_detected():
    response = NS(
        stop_reason="end_turn",
        usage=_usage(),
        content=[NS(type="text", text="Não encontrei nos artigos a resposta.", citations=None)],
    )
    client, _ = _client(response)
    gen = Generator(client=client, model="claude-opus-5-5")
    ans = gen.answer("homicídio?", [_result("a", "Art. 1º")])
    assert ans.abstained


def test_refusal_is_handled():
    response = NS(stop_reason="refusal", usage=_usage(), content=[])
    client, _ = _client(response)
    ans = Generator(client=client, model="claude-opus-5-5").answer("?", [_result("a", "Art. 1º")])
    assert ans.text.startswith("Não foi possível")


def test_no_fallback_param_for_models_without_support():
    response = NS(stop_reason="end_turn", usage=_usage(), content=[])
    client, messages = _client(response)
    Generator(client=client, model="claude-haiku-5-5").answer("?", [_result("a", "Art. 1º")])
    assert "fallbacks" not in messages.kwargs


def test_no_api_call_without_results():
    client, messages = _client(None)
    ans = Generator(client=client).answer("?", [])
    assert ans.abstained and messages.kwargs is None
