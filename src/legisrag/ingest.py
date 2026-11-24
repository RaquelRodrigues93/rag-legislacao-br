"""Baixa as leis do Planalto e gera data/processed/chunks.jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from legisrag.config import settings
from legisrag.parser import Chunk, decode_html, parse_law
from legisrag.sources import SOURCES, LawSource

log = logging.getLogger(__name__)

# O servidor do Planalto derruba conexões de clientes sem User-Agent de navegador.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def fetch(source: LawSource, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{source.slug}.html"
    if cached.exists():
        return decode_html(cached.read_bytes())
    log.info("Baixando %s", source.url)
    resp = requests.get(source.url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return decode_html(resp.content)


def ingest(out_path: Path | None = None) -> list[Chunk]:
    out_path = out_path or settings.chunks_path
    raw_dir = settings.data_dir / "raw"
    chunks: list[Chunk] = []
    for source in SOURCES:
        law_chunks = parse_law(fetch(source, raw_dir), source)
        n_articles = len({c.article_id for c in law_chunks})
        log.info("%s: %d artigos, %d chunks", source.name, n_articles, len(law_chunks))
        chunks.extend(law_chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or settings.chunks_path
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
