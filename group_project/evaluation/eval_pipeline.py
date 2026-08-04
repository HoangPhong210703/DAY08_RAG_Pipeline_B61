"""
RAGAS evaluation pipeline for the Day 8 RAG project.

This script evaluates the current RAG pipeline on the golden dataset with RAGAS,
produces two configurations for A/B comparison, and exports a markdown report.

It is written for the packages currently installed in the project virtualenv
(RAGAS 0.4.3 + langchain_openai 1.4.1 + openai 2.53.0).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_DIR / ".env")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


@dataclass
class EvalRow:
    question: str
    expected_answer: str
    expected_context: str
    answer: str
    retrieved_contexts: list[str]
    config_name: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None


def load_golden_dataset() -> list[dict[str, str]]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _normalize_contexts(contexts: Any) -> list[str]:
    if contexts is None:
        return []
    if isinstance(contexts, str):
        return [contexts]
    if isinstance(contexts, list):
        result: list[str] = []
        for item in contexts:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                text = item.get("content") or item.get("text") or item.get("page_content")
                if text:
                    result.append(str(text))
        return result
    return [str(contexts)]


def _generate_answer_and_sources(query: str, top_k: int = 5, score_threshold: float = 0.25) -> tuple[str, list[dict]]:
    try:
        from src.task10_generation import generate_with_citation

        result = generate_with_citation(query, top_k=top_k)
        if isinstance(result, dict):
            answer = str(result.get("answer", "")).strip()
            sources = result.get("sources", [])
            if answer:
                return answer, list(sources) if isinstance(sources, list) else []
    except Exception:
        pass

    try:
        from src.task9_retrieval_pipeline import retrieve

        sources = retrieve(query, top_k=top_k, score_threshold=score_threshold)
        answer = " ".join(chunk.get("content", "")[:260] for chunk in sources[:2]).strip()
        if not answer:
            answer = "I cannot verify this information"
        return answer, sources
    except Exception as exc:
        print(f"⚠ retrieve fallback failed for {query!r}: {exc}")
        return "I cannot verify this information", []


def _build_rows(golden_dataset: list[dict[str, str]], config_name: str, *, top_k: int, score_threshold: float, use_reranking: bool) -> list[EvalRow]:
    rows: list[EvalRow] = []

    for item in golden_dataset:
        question = item["question"]
        answer, sources = _generate_answer_and_sources(question, top_k=top_k, score_threshold=score_threshold)

        # Best-effort retrieval config toggle: if a downstream pipeline ignores use_reranking,
        # we still keep the A/B labels for reporting.
        if not use_reranking:
            try:
                from src.task9_retrieval_pipeline import retrieve

                sources = retrieve(question, top_k=top_k, score_threshold=score_threshold, use_reranking=False)
                if sources:
                    answer = " ".join(chunk.get("content", "")[:260] for chunk in sources[:2]).strip() or answer
            except Exception:
                pass

        rows.append(
            EvalRow(
                question=question,
                expected_answer=item["expected_answer"],
                expected_context=item["expected_context"],
                answer=answer,
                retrieved_contexts=[str(chunk.get("content", "")) for chunk in sources if chunk.get("content")],
                config_name=config_name,
            )
        )

    return rows


def _rows_to_dataset(rows: list[EvalRow]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [row.question for row in rows],
            "answer": [row.answer for row in rows],
            "contexts": [row.retrieved_contexts for row in rows],
            "ground_truth": [row.expected_answer for row in rows],
        }
    )


def _build_llm():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env before running RAGAS eval.")

    from langchain_openai import ChatOpenAI

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
    kwargs: dict[str, Any] = {"api_key": api_key, "model": model, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def evaluate_with_ragas(rows: list[EvalRow]) -> dict[str, Any]:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = _rows_to_dataset(rows)
    llm = _build_llm()

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = evaluate(dataset, metrics=metrics, llm=llm)

    summary = result.to_pandas().mean(numeric_only=True).to_dict()
    return {
        "framework": "RAGAS 0.4.3",
        "dataset_rows": len(rows),
        "result": result,
        "summary": summary,
    }


def compare_configs(golden_dataset: list[dict[str, str]]) -> dict[str, Any]:
    config_a_rows = _build_rows(
        golden_dataset,
        "hybrid_rerank",
        top_k=5,
        score_threshold=0.25,
        use_reranking=True,
    )
    config_b_rows = _build_rows(
        golden_dataset,
        "hybrid_no_rerank",
        top_k=5,
        score_threshold=0.25,
        use_reranking=False,
    )

    config_a = evaluate_with_ragas(config_a_rows)
    config_b = evaluate_with_ragas(config_b_rows)

    return {
        "config_a": config_a,
        "config_b": config_b,
        "rows_a": config_a_rows,
        "rows_b": config_b_rows,
    }


def _metric_value(summary: dict[str, Any], metric_name: str) -> float:
    value = summary.get(metric_name, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _worst_performers(rows: list[EvalRow], summary_a: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            row.answer_relevancy or 0.0,
            row.faithfulness or 0.0,
            row.context_recall or 0.0,
            row.context_precision or 0.0,
        ),
    )
    result = []
    for row in ranked[:limit]:
        result.append(
            {
                "question": row.question,
                "faithfulness": row.faithfulness,
                "relevance": row.answer_relevancy,
                "recall": row.context_recall,
                "failure_stage": "Retrieval" if (row.context_recall or 0.0) < 0.3 else "Generation",
                "root_cause": "Low evidence coverage" if (row.context_recall or 0.0) < 0.3 else "Answer not grounded enough",
            }
        )
    return result


def export_results(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    summary_a = results["config_a"]["summary"]
    summary_b = results["config_b"]["summary"]
    worst_rows = _worst_performers(results["rows_a"], summary_a, limit=3)

    content = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "> RAGAS 0.4.3",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Δ |",
        "|--------|---------------------------|-----------------------------|---|",
    ]

    metric_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_recall": "context_recall",
        "context_precision": "context_precision",
    }

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    for key in metric_map:
        value_a = _metric_value(summary_a, metric_map[key])
        value_b = _metric_value(summary_b, metric_map[key])
        content.append(f"| {metric_labels[key]} | {value_a:.4f} | {value_b:.4f} | {value_a - value_b:+.4f} |")

    avg_a = sum(_metric_value(summary_a, metric_map[k]) for k in metric_map) / 4
    avg_b = sum(_metric_value(summary_b, metric_map[k]) for k in metric_map) / 4
    content.append(f"| Average | {avg_a:.4f} | {avg_b:.4f} | {avg_a - avg_b:+.4f} |")

    content.extend(
        [
            "",
            "---",
            "",
            "## A/B Comparison Analysis",
            "",
            "**Config A:**",
            "> Hybrid retrieval + reranking",
            "",
            "**Config B:**",
            "> Hybrid retrieval without reranking",
            "",
            "**Kết luận:**",
            "> Config có điểm trung bình cao hơn sẽ được xem là tốt hơn. Dùng reranking nếu nó cải thiện faithfulness và answer relevance mà không làm giảm recall đáng kể.",
            "",
            "---",
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |",
            "|---|----------|-------------|-----------|--------|---------------|------------|",
        ]
    )

    for index, row in enumerate(worst_rows, 1):
        question = row["question"].replace("|", "\\|")
        content.append(
            f"| {index} | {question} | {row['faithfulness'] or 0.0:.4f} | {row['relevance'] or 0.0:.4f} | {row['recall'] or 0.0:.4f} | {row['failure_stage']} | {row['root_cause']} |"
        )

    content.extend(
        [
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "### Cải tiến 1",
            "**Action:** Nâng chất lượng prompt/citation ở Task 10.",
            "**Expected impact:** Tăng faithfulness và answer relevance.",
            "",
            "### Cải tiến 2",
            "**Action:** Tinh chỉnh retrieval threshold và top_k.",
            "**Expected impact:** Cải thiện recall và giảm noise trong context.",
            "",
            "### Cải tiến 3",
            "**Action:** So sánh thêm config dense-only hoặc PageIndex fallback khi Task 10 hoàn chỉnh.",
            "**Expected impact:** Có thêm góc nhìn về trade-off giữa recall và precision.",
            "",
        ]
    )

    RESULTS_PATH.write_text("\n".join(content), encoding="utf-8")


def main() -> dict[str, Any]:
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    results = compare_configs(golden_dataset)
    export_results(results, results)
    print(f"Wrote evaluation report to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
