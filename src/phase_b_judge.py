from __future__ import annotations

"""Phase B: pairwise judge, swap check, Cohen kappa, bias report."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _score(question: str, answer: str) -> float:
    q = _tokens(question)
    a = _tokens(answer)
    overlap = len(q & a) / max(len(q), 1)
    numbers = 0.15 if re.search(r"\d", answer) else 0.0
    policy_words = {"khong", "không", "bat", "bắt", "buoc", "buộc", "v2024", "hien", "hiện"}
    policy = 0.1 if a & policy_words else 0.0
    brevity = max(0.0, 0.15 - max(len(answer) - 500, 0) / 4000)
    return round(min(1.0, 0.45 + overlap * 0.45 + numbers + policy + brevity), 3)


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = _score(question, answer_a)
    score_b = _score(question, answer_b)
    if abs(score_a - score_b) < 0.03:
        winner = "tie"
        reasoning = "Answers are too close to call with the local rubric."
    else:
        winner = "A" if score_a > score_b else "B"
        reasoning = f"Answer {winner} is stronger on relevance, specificity, and policy cues."
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": score_a, "B": score_b}}


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]
    final = pass1["winner"] if pass1["winner"] == winner_pass2 else "tie"
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=pass1["winner"] == winner_pass2,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels):
        raise ValueError("label lists must have the same length")
    if not judge_labels:
        return 0.0
    n = len(judge_labels)
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    labels = set(judge_labels) | set(human_labels)
    p_e = sum((judge_labels.count(label) / n) * (human_labels.count(label) / n) for label in labels)
    return round((p_o - p_e) / (1 - p_e), 6) if p_e != 1 else 1.0


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "No judge results.",
        }

    position_bias_count = sum(not r.position_consistent for r in judge_results)
    a_wins_a_longer = sum(r.final_winner == "A" and len(r.answer_a) > len(r.answer_b) for r in judge_results)
    b_wins_b_longer = sum(r.final_winner == "B" and len(r.answer_b) > len(r.answer_a) for r in judge_results)
    decisive = sum(r.final_winner != "tie" for r in judge_results)
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0
    position_bias_rate = position_bias_count / total
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": "Use swap-and-average." if position_bias_rate > 0.3 else "Position bias is low.",
    }


if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)

    results = [
        swap_and_average(
            item["question"],
            item["model_answer"],
            "Khong du thong tin de ket luan chinh sach nay.",
        )
        for item in human_data
    ]
    judge_labels = [1 if r.final_winner == "A" else 0 for r in results]
    human_labels = [item["human_label"] for item in human_data]
    report = {
        "results": [asdict(r) for r in results],
        "cohen_kappa": cohen_kappa(judge_labels, human_labels),
        "bias_report": bias_report(results),
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/bias_report.md", "w", encoding="utf-8") as f:
        f.write("# Judge Bias Report\n\n")
        f.write(json.dumps(report["bias_report"], ensure_ascii=False, indent=2))
    print("Phase B report saved")
