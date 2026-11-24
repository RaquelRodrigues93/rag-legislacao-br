"""API HTTP. Rode com: uvicorn legisrag.api:app --reload"""

from __future__ import annotations

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from legisrag.generator import CredentialsError
from legisrag.pipeline import get_pipeline

load_dotenv()

app = FastAPI(
    title="legisrag",
    description="Perguntas e respostas sobre LGPD e CDC com citações verificáveis.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(6, ge=1, le=15)


class CitationOut(BaseModel):
    source: int
    label: str
    cited_text: str
    url: str


class SourceOut(BaseModel):
    rank: int
    label: str
    hierarchy: str
    url: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[CitationOut]
    sources: list[SourceOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        ans = get_pipeline().ask(req.question, k=req.k)
    except CredentialsError as e:
        raise HTTPException(503, str(e)) from e
    except anthropic.AuthenticationError as e:
        raise HTTPException(503, "Chave da API do Claude inválida.") from e
    except anthropic.RateLimitError as e:
        raise HTTPException(429, "Limite de requisições da API do Claude atingido.") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(503, "Não foi possível conectar à API do Claude.") from e
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Erro da API do Claude ({e.status_code}).") from e

    return AskResponse(
        answer=ans.text,
        abstained=ans.abstained,
        citations=[CitationOut(**vars(c)) for c in ans.citations],
        sources=[
            SourceOut(
                rank=r.rank,
                label=r.article.citation_label,
                hierarchy=r.article.hierarchy,
                url=r.article.url,
            )
            for r in ans.sources
        ],
    )
