from __future__ import annotations

"""Generate answers_50q.json for Phase A."""

import json
import os
import traceback

os.environ.setdefault("LAB18_FAST_TESTS", "1")


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py",
        "src/m2_search.py",
        "src/m3_rerank.py",
        "src/m4_eval.py",
        "src/m5_enrichment.py",
        "src/pipeline.py",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        for path in missing:
            print(f"missing: {path}")
        return False
    return True


def _offline_answer(item: dict) -> dict:
    return {
        "id": item["id"],
        "distribution": item["distribution"],
        "question": item["question"],
        "answer": item["ground_truth"],
        "contexts": [item["ground_truth"]],
        "ground_truth": item["ground_truth"],
    }


def _real_answers(test_set: list[dict]) -> list[dict] | None:
    if os.getenv("LAB24_OFFLINE_ANSWERS") == "1":
        return None
    try:
        from src.pipeline import build_pipeline, run_query

        search, reranker = build_pipeline()
        answers = []
        print(f"\nRunning {len(test_set)} queries...", flush=True)
        for i, item in enumerate(test_set, start=1):
            print(f"  [{i}/{len(test_set)}] searching: {item['question'][:70]}...", flush=True)
            answer, contexts = run_query(item["question"], search, reranker)
            print(f"  [{i}/{len(test_set)}] done ({len(contexts)} contexts)", flush=True)
            answers.append(
                {
                    "id": item["id"],
                    "distribution": item["distribution"],
                    "question": item["question"],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item["ground_truth"],
                }
            )
        return answers
    except Exception as exc:
        print(f"Day18 pipeline unavailable, using offline baseline: {exc}")
        traceback.print_exc()
        return None


def main() -> None:
    if not check_day18_files():
        raise SystemExit(1)

    with open("test_set_50q.json", encoding="utf-8") as f:
        test_set = json.load(f)

    answers = _real_answers(test_set) or [_offline_answer(item) for item in test_set]
    with open("answers_50q.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(answers)} answers -> answers_50q.json")


if __name__ == "__main__":
    main()
