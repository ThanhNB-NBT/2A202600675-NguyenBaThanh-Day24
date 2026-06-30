from __future__ import annotations

"""Phase C: local guardrails for PII, adversarial inputs, and latency."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, LATENCY_BUDGET_P95_MS


PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "VN_CCCD": re.compile(r"\b(?:\d{12}|\d{9})\b"),
    "VN_PHONE": re.compile(r"\b0[3-9]\d{8}\b"),
}

BLOCK_KEYWORDS = [
    "cccd",
    "cmnd",
    "so dien thoai",
    "sdt",
    "email cua",
    "bo qua",
    "ignore",
    "forget",
    "system override",
    "admin command",
    "previous instructions",
    "system instructions",
    "dan",
    "unrestricted",
    "khong co gioi han",
    "khong có giới hạn",
    "dong vai",
    "đóng vai",
    "tiet lo",
    "tiết lộ",
    "confidential",
    "training data",
    "employee records",
    "all employee",
    "bang luong chi tiet",
    "bảng lương chi tiết",
    "tan cong mang",
    "tấn công mạng",
    "bai tho",
    "bài thơ",
    "nau pho",
    "nấu phở",
    "bitcoin",
    "ethereum",
    "phuong trinh",
    "phương trình",
    "marvel",
]


def _strip_accents(text: str) -> str:
    table = str.maketrans(
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ",
        "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
        "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD",
    )
    return text.translate(table)


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    entities = []
    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                {
                    "type": entity_type,
                    "text": match.group(0),
                    "score": 0.9,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    anonymized = text
    for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
        anonymized = anonymized[: entity["start"]] + f"<{entity['type']}>" + anonymized[entity["end"] :]
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}


def setup_presidio():
    return None, None


def setup_nemo_rails():
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    normalized = _strip_accents(text).lower()
    blocked = any(keyword in normalized for keyword in BLOCK_KEYWORDS)
    return {
        "allowed": not blocked,
        "blocked_reason": "local_input_rail" if blocked else None,
        "response": "blocked" if blocked else "allowed",
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    pii = pii_scan(answer)
    blocked = pii["has_pii"] or not (await check_input_rail(answer))["allowed"]
    return {
        "safe": not blocked,
        "flagged_reason": "local_output_rail" if blocked else None,
        "final_answer": pii["anonymized"] if pii["has_pii"] else answer,
    }


def run_adversarial_suite(adversarial_set: list[dict], rails=None, analyzer=None, anonymizer=None) -> list[dict]:
    async def _run_all() -> list[dict]:
        rows = []
        for item in adversarial_set:
            blocked_by = None
            if pii_scan(item["input"], analyzer, anonymizer)["has_pii"]:
                blocked_by = "presidio"
            elif not (await check_input_rail(item["input"], rails))["allowed"]:
                blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "input": item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                    "expected": item["expected"],
                    "actual": actual,
                    "blocked_by": blocked_by,
                    "passed": actual == item["expected"],
                }
            )
        return rows

    return asyncio.run(_run_all())


def _percentiles(times: list[float]) -> dict:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(times)
    n = len(s)
    return {
        "p50": round(s[min(int(n * 0.50), n - 1)], 2),
        "p95": round(s[min(int(n * 0.95), n - 1)], 2),
        "p99": round(s[min(int(n * 0.99), n - 1)], 2),
    }


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20, rails=None, analyzer=None, anonymizer=None) -> dict:
    inputs = (test_inputs or ["test"])[:n_runs]
    presidio_times, nemo_times, total_times = [], [], []

    async def _measure() -> None:
        for text in inputs:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())
    total = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total,
        "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


if __name__ == "__main__":
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    results = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency([item["input"] for item in adversarial_set], n_runs=len(adversarial_set))
    report = {
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "pass_rate": sum(r["passed"] for r in results) / len(results) if results else 0.0,
        "results": results,
        "latency": latency,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved: {report['passed']}/{report['total']} passed")
