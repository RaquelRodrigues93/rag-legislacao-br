"""Geração da resposta com Claude usando a API de Citations.

Cada artigo recuperado vai como um bloco `document` com `citations.enabled`. O modelo devolve a
resposta em blocos de texto, e cada trecho fundamentado traz o `cited_text` exato do artigo. Isso
dá citações **verificáveis**: a referência aponta para um trecho literal da lei, não para um
"[1]" que o modelo pode ter inventado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from legisrag.config import settings
from legisrag.retriever import SearchResult

SYSTEM_PROMPT = """\
Você responde perguntas sobre legislação brasileira para pessoas leigas, usando apenas os \
trechos de lei fornecidos como documentos nesta conversa.

- Fundamente cada afirmação nos documentos e cite o artigo correspondente.
- Se os documentos não trouxerem a resposta, comece a resposta exatamente com "Não encontrei" \
e diga em uma frase o que faltou. Não complete com conhecimento próprio: quem usa o sistema \
precisa poder conferir cada informação na fonte.
- Escreva em português claro e direto, em poucos parágrafos curtos. Explique termos técnicos \
quando aparecerem.
"""

ABSTAIN_PREFIX = "Não encontrei"

# Modelos que aceitam o parâmetro server-side `fallbacks` (beta). Se o modelo recusar a
# requisição por política de segurança, a API refaz a chamada em um modelo de fallback.
_FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class CredentialsError(RuntimeError):
    """Nenhuma credencial da API Anthropic configurada."""

    def __init__(self) -> None:
        super().__init__(
            "Credenciais da API do Claude não encontradas. Defina ANTHROPIC_API_KEY no "
            "arquivo .env (veja .env.example)."
        )


@dataclass
class Citation:
    source: int  # número da fonte (1-based), igual à posição em `sources`
    label: str
    cited_text: str
    url: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[SearchResult] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def abstained(self) -> bool:
        return self.text.strip().startswith(ABSTAIN_PREFIX)

    @property
    def cited_article_ids(self) -> set[str]:
        return {self.sources[c.source - 1].article.article_id for c in self.citations}


def _documents(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": r.article.text},
            "title": r.article.citation_label,
            "context": f"{r.article.law_number} — {r.article.hierarchy}",
            "citations": {"enabled": True},
        }
        for r in results
    ]


class Generator:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        self.client = client or anthropic.Anthropic()
        self.model = model or settings.claude_model

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(text=f"{ABSTAIN_PREFIX} nenhum artigo relacionado à pergunta.")

        kwargs: dict = {}
        if self.model in _FALLBACK_MODELS:
            kwargs = {"betas": [_FALLBACK_BETA], "fallbacks": "default"}

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                output_config={"effort": settings.claude_effort},
                messages=[
                    {
                        "role": "user",
                        "content": [*_documents(results), {"type": "text", "text": question}],
                    }
                ],
                **kwargs,
            )
        except TypeError as e:
            # O SDK só detecta a falta de credenciais no momento da requisição.
            if "authentication" in str(e):
                raise CredentialsError() from e
            raise

        answer = Answer(
            text="",
            sources=results,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        if response.stop_reason == "refusal":
            answer.text = "Não foi possível responder a esta pergunta."
            return answer

        parts: list[str] = []
        for block in response.content:
            if block.type != "text":
                continue  # thinking / fallback blocks
            parts.append(block.text)
            markers = []
            for cit in block.citations or []:
                if cit.type != "char_location":
                    continue
                src = cit.document_index + 1
                article = results[cit.document_index].article
                answer.citations.append(
                    Citation(src, article.citation_label, cit.cited_text.strip(), article.url)
                )
                if src not in markers:
                    markers.append(src)
            if markers:
                parts.append("".join(f"[{m}]" for m in markers))
        answer.text = "".join(parts).strip()
        return answer
