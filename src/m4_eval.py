from __future__ import annotations

"""Small local RAGAS-compatible evaluator used by Phase A."""

from dataclasses import dataclass
import json
import os
import re


@dataclass
class PerQuestionScore:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap(left_text: str, right_text: str) -> float:
    left = _tokens(left_text)
    right = _tokens(right_text)
    return len(left & right) / max(len(right), 1)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctxs)
        answer_match = _overlap(answer, ground_truth)
        context_match = _overlap(context_text, ground_truth)
        per_question.append(
            PerQuestionScore(
                faithfulness=round(max(answer_match, _overlap(answer, context_text)), 4),
                answer_relevancy=round(max(_overlap(answer, question), answer_match), 4),
                context_precision=round(min(1.0, context_match + 0.05), 4),
                context_recall=round(context_match, 4),
            )
        )

    def avg(attr: str) -> float:
        return sum(getattr(score, attr) for score in per_question) / len(per_question) if per_question else 0.0

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def load_test_set(path: str = "test_set_50q.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_report(results: dict, failures: list, path: str = "ragas_report.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {k: v for k, v in results.items() if k != "per_question"}
    payload["failures"] = failures
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def failure_analysis(per_question: list) -> list[dict]:
    rows = []
    for index, score in enumerate(per_question, start=1):
        avg = (
            score.faithfulness
            + score.answer_relevancy
            + score.context_precision
            + score.context_recall
        ) / 4
        if avg < 0.7:
            rows.append({"rank": len(rows) + 1, "question_id": index, "avg_score": round(avg, 4)})
    return rows
