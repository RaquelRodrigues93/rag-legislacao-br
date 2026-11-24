"""Configuração central. Tudo pode ser sobrescrito por variáveis de ambiente."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("LEGISRAG_DATA_DIR", ROOT / "data"))
    # Modelo de embeddings multilíngue local (gratuito, roda em CPU).
    # Escolhido pela avaliação (ver README): e5-large teve o melhor MRR. e5-base é a opção leve.
    embedding_model: str = os.getenv("LEGISRAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    # dense | hybrid | bm25 — com perguntas em linguagem leiga, a busca densa venceu a híbrida.
    search_mode: str = os.getenv("LEGISRAG_SEARCH_MODE", "dense")
    # Peso do ranking BM25 no RRF quando search_mode=hybrid (o denso tem peso 1).
    bm25_weight: float = float(os.getenv("LEGISRAG_BM25_WEIGHT", "0.3"))
    claude_model: str = os.getenv("LEGISRAG_CLAUDE_MODEL", "claude-opus-5-5")
    # low | medium | high — perguntas sobre artigos de lei raramente precisam de mais que medium.
    claude_effort: str = os.getenv("LEGISRAG_CLAUDE_EFFORT", "medium")
    top_k: int = int(os.getenv("LEGISRAG_TOP_K", "6"))
    # Constante do Reciprocal Rank Fusion (60 é o valor do artigo original).
    rrf_k: int = int(os.getenv("LEGISRAG_RRF_K", "60"))

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "processed" / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()
