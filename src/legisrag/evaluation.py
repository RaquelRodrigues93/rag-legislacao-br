"""Avaliação automatizada.

- **Recuperação** (gratuita, determinística): para cada pergunta sabemos quais artigos são
  relevantes. Medimos Hit@k (achou algum relevante entre os k primeiros), Recall@k e MRR, para
  cada modo de busca. É o que diz se uma mudança no chunking/índice ajudou ou piorou.
- **Geração** (chama a API, tem custo): mede se a resposta cita um artigo relevante e se o
  sistema se abstém nas perguntas fora do escopo, em vez de inventar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from legisrag.config import ROOT
from legisrag.retriever import HybridRetriever, Mode

DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


@dataclass
class EvalItem:
    id: str
    question: str
    relevant: list[str]

    @property
    def in_scope(self) -> bool:
        return bool(self.relevant)


def load_dataset(path: Path = DATASET) -> list[EvalItem]:
    with path.open(encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def retrieval_metrics(ranked: list[str], relevant: set[str], ks=(1, 3, 6)) -> dict[str, float]:
    m: dict[str, float] = {}
    for k in ks:
        top = ranked[:k]
        m[f"hit@{k}"] = float(any(a in relevant for a in top))
        m[f"recall@{k}"] = len(relevant & set(top)) / len(relevant)
    first = next((i for i, a in enumerate(ranked, 1) if a in relevant), None)
    m["mrr"] = 1.0 / first if first else 0.0
    return m


def evaluate_retrieval(
    retriever: HybridRetriever, modes: tuple[Mode, ...] = ("bm25", "hybrid", "dense")
) -> dict:
    items = [it for it in load_dataset() if it.in_scope]
    report: dict = {"n_questions": len(items), "modes": {}, "misses": {}}
    for mode in modes:
        totals: dict[str, float] = {}
        misses = []
        for it in items:
            ranked = [r.article.article_id for r in retriever.search(it.question, k=10, mode=mode)]
            m = retrieval_metrics(ranked, set(it.relevant))
            for key, v in m.items():
                totals[key] = totals.get(key, 0.0) + v
            if not m["hit@6"]:
                misses.append({"id": it.id, "question": it.question, "got": ranked[:6]})
        report["modes"][mode] = {k: round(v / len(items), 3) for k, v in totals.items()}
        report["misses"][mode] = misses
    return report


def evaluate_generation(pipeline) -> dict:
    items = load_dataset()
    rows = []
    for it in items:
        ans = pipeline.ask(it.question)
        rows.append(
            {
                "id": it.id,
                "in_scope": it.in_scope,
                "abstained": ans.abstained,
                "cited_relevant": bool(ans.cited_article_ids & set(it.relevant)),
                "n_citations": len(ans.citations),
                "input_tokens": ans.input_tokens,
                "output_tokens": ans.output_tokens,
                "answer": ans.text,
            }
        )
    in_scope = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    summary = {
        "n_questions": len(rows),
        # Entre as perguntas no escopo: a resposta citou literalmente um artigo relevante?
        "citation_accuracy": round(sum(r["cited_relevant"] for r in in_scope) / len(in_scope), 3),
        # Perguntas no escopo em que o sistema disse "não encontrei" (deveria responder).
        "false_abstention_rate": round(sum(r["abstained"] for r in in_scope) / len(in_scope), 3),
        # Perguntas fora do escopo em que o sistema corretamente se absteve.
        "oos_abstention_rate": (
            round(sum(r["abstained"] for r in oos) / len(oos), 3) if oos else None
        ),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / len(rows)),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in rows) / len(rows)),
    }
    return {"summary": summary, "rows": rows}


def save(report: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return path


def retrieval_table(report: dict) -> str:
    cols = ["hit@1", "hit@3", "hit@6", "recall@6", "mrr"]
    lines = ["| modo | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, m in report["modes"].items():
        lines.append(f"| {mode} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |")
    return "\n".join(lines)


# Métricas verificadas contra o baseline. Hit@1 fica de fora: oscila demais com 47 perguntas.
GATED_METRICS = ("hit@3", "hit@6", "recall@6", "mrr")


def find_regressions(
    report: dict,
    baseline: dict,
    mode: str,
    # 0.02: tolera variação numérica entre máquinas (um artigo do 1º para o 2º lugar tira 0.011
    # do MRR), mas acusa uma pergunta a menos no Hit@3/Hit@6 (1/47 = 0.021).
    tolerance: float = 0.02,
    metrics: tuple[str, ...] = GATED_METRICS,
) -> list[str]:
    """Métricas do `mode` que caíram mais que `tolerance` em relação ao baseline.

    `report` é a saída de `evaluate_retrieval`; `baseline` é um arquivo salvo por `save`.
    """
    base = baseline["retrieval"]["modes"].get(mode)
    if base is None:
        raise ValueError(f"O baseline não tem resultados para o modo {mode!r}.")
    current = report["modes"][mode]
    return [
        f"{mode} {m}: {current[m]:.3f} (baseline {base[m]:.3f})"
        for m in metrics
        if current[m] < base[m] - tolerance
    ]


def markdown_summary(
    report: dict, mode: str, regressions: list[str] | None = None, generation: dict | None = None
) -> str:
    """Resumo da avaliação em Markdown (usado no resumo da execução da CI)."""
    parts = [
        f"## Avaliação de recuperação ({report['n_questions']} perguntas)",
        retrieval_table(report),
    ]
    if regressions is not None:
        if regressions:
            parts.append("**Regressão em relação ao baseline:**")
            parts.append("\n".join(f"- {r}" for r in regressions))
        else:
            parts.append("Nenhuma regressão em relação ao baseline.")
    misses = report["misses"][mode]
    if misses:
        parts.append(
            f"<details><summary>Sem artigo relevante no top 6 (modo {mode}): "
            f"{len(misses)}</summary>\n"
        )
        parts.append("\n".join(f"- `{m['id']}` {m['question']}" for m in misses))
        parts.append("</details>")
    if generation is not None:
        parts.append("## Avaliação de geração")
        parts.append("\n".join(f"- **{k}**: {v}" for k, v in generation["summary"].items()))
    return "\n\n".join(parts) + "\n"
