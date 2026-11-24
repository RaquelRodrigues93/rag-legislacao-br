"""Ponto de entrada único: pergunta -> busca -> resposta citada."""

from __future__ import annotations

from functools import lru_cache

from legisrag.config import settings
from legisrag.generator import Answer, Generator
from legisrag.retriever import HybridRetriever, Mode


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, k: int = settings.top_k, mode: Mode | None = None) -> Answer:
        results = self.retriever.search(question, k=k, mode=mode)
        return self.generator.answer(question, results)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(HybridRetriever.load(), Generator())
