from __future__ import annotations

"""Module 5: Enrichment Pipeline."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_MODEL, openai_client

_OPENAI_FAILED = False


@dataclass
class EnrichedChunk:
    """Enriched chunk."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _json_from_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.I | re.M).strip()
    return json.loads(content)


def summarize_chunk(text: str) -> str:
    """Create a short chunk summary."""
    global _OPENAI_FAILED
    if OPENAI_API_KEY and not _OPENAI_FAILED and os.getenv("LAB18_FAST_TESTS") != "1":
        try:
            resp = openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Tom tat doan van sau trong 1-2 cau ngan bang tieng Viet."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            _OPENAI_FAILED = True
            print(f"  OpenAI summarize failed: {e}", flush=True)

    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that the chunk can answer."""
    global _OPENAI_FAILED
    if OPENAI_API_KEY and not _OPENAI_FAILED and os.getenv("LAB18_FAST_TESTS") != "1":
        try:
            resp = openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": f"Tao {n_questions} cau hoi ma doan van co the tra loi. Moi dong mot cau."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            lines = resp.choices[0].message.content.strip().splitlines()
            return [q.strip().lstrip("0123456789.-) ") for q in lines if q.strip()][:n_questions]
        except Exception as e:
            _OPENAI_FAILED = True
            print(f"  OpenAI HyQA failed: {e}", flush=True)

    return [f"{s.rstrip('.')}?" for s in _sentences(text)[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a short context line to the chunk."""
    global _OPENAI_FAILED
    if OPENAI_API_KEY and not _OPENAI_FAILED and os.getenv("LAB18_FAST_TESTS") != "1":
        try:
            resp = openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Viet 1 cau ngan mo ta ngu canh cua doan van. Chi tra ve 1 cau."},
                    {"role": "user", "content": f"Tai lieu: {document_title}\n\nDoan van:\n{text}"},
                ],
                max_tokens=80,
            )
            return f"{resp.choices[0].message.content.strip()}\n\n{text}"
        except Exception as e:
            _OPENAI_FAILED = True
            print(f"  OpenAI contextual failed: {e}", flush=True)

    prefix = f"Trich tu {document_title}. " if document_title else "Ngu canh tai lieu. "
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    """Extract lightweight metadata for retrieval filters."""
    global _OPENAI_FAILED
    if OPENAI_API_KEY and not _OPENAI_FAILED and os.getenv("LAB18_FAST_TESTS") != "1":
        try:
            resp = openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": 'Chi tra ve JSON: {"topic":"...","entities":["..."],"category":"policy|hr|it|finance","language":"vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return _json_from_response(resp.choices[0].message.content)
        except Exception as e:
            _OPENAI_FAILED = True
            print(f"  OpenAI metadata failed: {e}", flush=True)

    low = text.lower()
    category = "hr" if any(w in low for w in ["nghi", "nhan vien", "luong", "phep"]) else "policy"
    return {"topic": summarize_chunk(text)[:80], "entities": [], "category": category, "language": "vi"}


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary, questions, context, and metadata."""
    global _OPENAI_FAILED
    if OPENAI_API_KEY and not _OPENAI_FAILED and os.getenv("LAB18_FAST_TESTS") != "1":
        try:
            resp = openai_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": (
                        'Chi tra ve JSON: {"summary":"...","questions":["..."],'
                        '"context":"...","metadata":{"topic":"...","entities":["..."],'
                        '"category":"policy|hr|it|finance","language":"vi|en"}}'
                    )},
                    {"role": "user", "content": f"Tai lieu: {source}\n\nDoan van:\n{text}"},
                ],
                max_tokens=400,
            )
            return _json_from_response(resp.choices[0].message.content)
        except Exception as e:
            _OPENAI_FAILED = True
            print(f"  Enrichment API failed: {e}", flush=True)

    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trich tu {source}." if source else "Ngu canh tai lieu.",
        "metadata": extract_metadata(text),
    }


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Run enrichment over chunks."""
    methods = methods or ["combined"]
    enriched = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context = result.get("context", "")
            enriched_text = f"{context}\n\n{text}" if context else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print(summarize_chunk(sample))
    print(generate_hypothesis_questions(sample))
    print(contextual_prepend(sample, "So tay nhan vien"))
    print(extract_metadata(sample))
