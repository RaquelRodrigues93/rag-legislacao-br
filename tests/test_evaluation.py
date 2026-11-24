"""Comparação com o baseline e resumo em Markdown usados pela CI de avaliação."""

import pytest

from legisrag.evaluation import find_regressions, markdown_summary


def _metrics(**overrides) -> dict:
    m = {
        "hit@1": 0.7,
        "recall@1": 0.6,
        "hit@3": 0.85,
        "recall@3": 0.8,
        "hit@6": 0.9,
        "recall@6": 0.9,
        "mrr": 0.8,
    }
    return m | overrides


def _report(dense: dict, misses: list | None = None) -> dict:
    return {
        "n_questions": 47,
        "modes": {"bm25": _metrics(mrr=0.5), "dense": dense},
        "misses": {"bm25": [], "dense": misses or []},
    }


BASELINE = {"retrieval": _report(_metrics())}


def test_no_regression_when_metrics_hold_or_improve():
    assert find_regressions(_report(_metrics(mrr=0.85)), BASELINE, "dense") == []


def test_drop_within_tolerance_is_accepted():
    assert find_regressions(_report(_metrics(mrr=0.789)), BASELINE, "dense") == []


def test_drop_beyond_tolerance_is_reported():
    found = find_regressions(_report(_metrics(mrr=0.75, **{"hit@6": 0.87})), BASELINE, "dense")
    assert found == ["dense hit@6: 0.870 (baseline 0.900)", "dense mrr: 0.750 (baseline 0.800)"]


def test_hit_at_1_is_not_gated():
    assert find_regressions(_report(_metrics(**{"hit@1": 0.5})), BASELINE, "dense") == []


def test_missing_mode_in_baseline_is_an_error():
    with pytest.raises(ValueError, match="hybrid"):
        find_regressions(_report(_metrics()), BASELINE, "hybrid")


def test_markdown_summary():
    misses = [{"id": "cdc-12", "question": "Venda casada é proibida?", "got": []}]
    md = markdown_summary(_report(_metrics(), misses), "dense", ["dense mrr: 0.750"])
    assert "| dense | 0.70 | 0.85 | 0.90 | 0.90 | 0.80 |" in md
    assert "- dense mrr: 0.750" in md
    assert "`cdc-12` Venda casada é proibida?" in md

    assert "Nenhuma regressão" in markdown_summary(_report(_metrics()), "dense", [])
    assert "baseline" not in markdown_summary(_report(_metrics()), "dense", None)
