"""CLI: legisrag ingest | index | ask | eval."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import typer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = typer.Typer(add_completion=False, help="RAG sobre legislação brasileira (LGPD e CDC).")


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def ingest() -> None:
    """Baixa as leis do Planalto e gera data/processed/chunks.jsonl."""
    from legisrag.ingest import ingest as run

    chunks = run()
    typer.echo(f"{len(chunks)} chunks de {len({c.article_id for c in chunks})} artigos.")


@app.command()
def index() -> None:
    """Calcula os embeddings dos chunks (data/index/)."""
    from legisrag.retriever import build_index

    build_index()
    typer.echo("Índice criado.")


@app.command()
def search(
    question: str,
    k: int = typer.Option(6, "--k"),
    mode: str = typer.Option(None, "--mode", help="dense | hybrid | bm25 (padrão: config)"),
) -> None:
    """Mostra só os artigos recuperados (sem chamar o Claude)."""
    from legisrag.retriever import HybridRetriever

    for r in HybridRetriever.load().search(question, k=k, mode=mode):  # type: ignore[arg-type]
        typer.echo(f"{r.rank}. {r.article.citation_label}  (score {r.score:.4f})")


@app.command()
def ask(question: str, k: int = typer.Option(6, "--k")) -> None:
    """Responde a pergunta com citações."""
    from legisrag.generator import CredentialsError
    from legisrag.pipeline import get_pipeline

    try:
        ans = get_pipeline().ask(question, k=k)
    except CredentialsError as e:
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"\n{ans.text}\n")
    seen = set()
    for c in ans.citations:
        if (c.source, c.cited_text) in seen:
            continue
        seen.add((c.source, c.cited_text))
        typer.echo(f'[{c.source}] {c.label}: "{c.cited_text}"\n    {c.url}')


@app.command(name="eval")
def evaluate(
    generation: bool = typer.Option(
        False, "--generation", help="Também avalia a geração (chama a API do Claude; tem custo)."
    ),
    name: str = typer.Option("latest", "--name", help="Nome do arquivo em eval/results/."),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="Resultado salvo para comparar; sai com erro se o modo padrão piorar.",
        exists=True,
        dir_okay=False,
    ),
    tolerance: float = typer.Option(
        0.02, "--tolerance", help="Queda máxima aceita em cada métrica antes de acusar regressão."
    ),
    summary_file: Path | None = typer.Option(
        None, "--summary-file", help="Anexa um resumo em Markdown (ex.: $GITHUB_STEP_SUMMARY)."
    ),
) -> None:
    """Roda a avaliação de recuperação (e opcionalmente de geração)."""
    from legisrag import evaluation as ev
    from legisrag.config import settings
    from legisrag.retriever import HybridRetriever

    mode = settings.search_mode
    retriever = HybridRetriever.load()
    report = ev.evaluate_retrieval(retriever)
    typer.echo(f"\nRecuperação ({report['n_questions']} perguntas):\n")
    typer.echo(ev.retrieval_table(report))
    typer.echo(f"\nPerguntas sem artigo relevante no top 6 (modo {mode}):")
    for miss in report["misses"][mode]:
        typer.echo(f"  ✗ {miss['id']}: {miss['question']}")
    out = {"retrieval": report}

    regressions = None
    if baseline is not None:
        regressions = ev.find_regressions(
            report, json.loads(baseline.read_text()), mode, tolerance=tolerance
        )

    if generation:
        from legisrag.generator import Generator
        from legisrag.pipeline import RAGPipeline

        gen = ev.evaluate_generation(RAGPipeline(retriever, Generator()))
        typer.echo("\nGeração:\n" + json.dumps(gen["summary"], indent=2, ensure_ascii=False))
        out["generation"] = gen

    typer.echo(f"\nResultados salvos em {ev.save(out, name)}")

    if summary_file is not None:
        with summary_file.open("a", encoding="utf-8") as f:
            f.write(ev.markdown_summary(report, mode, regressions, out.get("generation")))

    if regressions:
        typer.secho(f"\nRegressão em relação a {baseline}:", fg="red", err=True)
        for r in regressions:
            typer.secho(f"  {r}", fg="red", err=True)
        raise typer.Exit(1)
    if regressions is not None:
        typer.echo(f"\nSem regressão em relação a {baseline} (tolerância {tolerance}).")


if __name__ == "__main__":
    app()
