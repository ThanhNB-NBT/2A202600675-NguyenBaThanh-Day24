from __future__ import annotations

"""Phase A: RAGAS-style production evaluation."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError("answers_50q.json not found. Run: python setup_answers.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        groups.setdefault(item["distribution"], []).append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    from src.m4_eval import evaluate_ragas

    raw = evaluate_ragas(
        [a["question"] for a in answers],
        [a["answer"] for a in answers],
        [a.get("contexts", []) for a in answers],
        [a["ground_truth"] for a in answers],
    )
    return [
        RagasResult(
            question_id=a["id"],
            distribution=a["distribution"],
            question=a["question"],
            answer=a["answer"],
            contexts=a.get("contexts", []),
            ground_truth=a["ground_truth"],
            faithfulness=float(pq.faithfulness),
            answer_relevancy=float(pq.answer_relevancy),
            context_precision=float(pq.context_precision),
            context_recall=float(pq.context_recall),
        )
        for a, pq in zip(answers, raw.get("per_question", []))
    ]


def bottom_10(results: list[RagasResult]) -> list[dict]:
    rows = []
    for rank, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[r.worst_metric]
        rows.append(
            {
                "rank": rank,
                "question_id": r.question_id,
                "distribution": r.distribution,
                "question": r.question,
                "avg_score": round(r.avg_score, 4),
                "worst_metric": r.worst_metric,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return rows


def cluster_analysis(results: list[RagasResult]) -> dict:
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for r in results:
        matrix[r.worst_metric][r.distribution] += 1

    dists = ["factual", "multi_hop", "adversarial"]
    dominant_dist = max(dists, key=lambda d: sum(row[d] for row in matrix.values()))
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": (
            f"{dominant_dist} has the most failures; {dominant_metric} is the weakest "
            f"metric. Suggested fix: {DIAGNOSTIC_TREE[dominant_metric][1]}."
        ),
    }


def save_phase_a_report(results: list[RagasResult], clusters: dict, path: str = "reports/ragas_50q.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per_dist = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness": sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy": sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall": sum(r.context_recall for r in subset) / len(subset),
                "avg_score": sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_failure_analysis(b10: list[dict], clusters: dict, path: str = "analysis/failure_clusters.md") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Failure Clusters\n\n")
        f.write(clusters["insight"] + "\n\n")
        for item in b10:
            f.write(
                f"- #{item['rank']} Q{item['question_id']} ({item['distribution']}): "
                f"{item['worst_metric']} - {item['diagnosis']}. Fix: {item['suggested_fix']}\n"
            )


if __name__ == "__main__":
    test_set = load_test_set_50q()
    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"{dist}: {len(qs)}")

    results = run_ragas_50q(load_answers())
    clusters = cluster_analysis(results)
    b10 = bottom_10(results)
    save_phase_a_report(results, clusters)
    save_failure_analysis(b10, clusters)
    print(f"Phase A report saved: {len(results)} questions")
