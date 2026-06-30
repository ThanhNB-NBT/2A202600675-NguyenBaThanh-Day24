"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
import sys
import time

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        return None

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

load_dotenv()

# --- API Keys ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or NVIDIA_API_KEY
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"
NAIVE_COLLECTION = "lab24_naive"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0"))

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge ---
JUDGE_MODEL = os.getenv("JUDGE_MODEL", OPENAI_MODEL)

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer

NVIDIA_CHAT_RPM = int(os.getenv("NVIDIA_CHAT_RPM", "10"))
NVIDIA_MAX_RETRIES = int(os.getenv("NVIDIA_MAX_RETRIES", "5"))
_LAST_CHAT_CALL = 0.0


def openai_client():
    from openai import OpenAI

    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return RateLimitedOpenAI(OpenAI(**kwargs))


class RateLimitedOpenAI:
    def __init__(self, client):
        self._client = client
        self.embeddings = client.embeddings
        self.chat = RateLimitedChat(client.chat)

    def __getattr__(self, name):
        return getattr(self._client, name)


class RateLimitedChat:
    def __init__(self, chat):
        self.completions = RateLimitedCompletions(chat.completions)


class RateLimitedCompletions:
    def __init__(self, completions):
        self._completions = completions

    def create(self, *args, **kwargs):
        global _LAST_CHAT_CALL
        delay = 60.0 / max(NVIDIA_CHAT_RPM, 1)
        for attempt in range(NVIDIA_MAX_RETRIES + 1):
            wait = delay - (time.monotonic() - _LAST_CHAT_CALL)
            if wait > 0:
                print(f"  [LLM] rate limit sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
            try:
                result = self._completions.create(*args, **kwargs)
                _LAST_CHAT_CALL = time.monotonic()
                return result
            except Exception as exc:
                _LAST_CHAT_CALL = time.monotonic()
                if "429" not in str(exc) or attempt >= NVIDIA_MAX_RETRIES:
                    raise
                backoff = min(60.0, delay * (attempt + 1))
                print(f"  [LLM] 429 retry in {backoff:.1f}s ({attempt + 1}/{NVIDIA_MAX_RETRIES})", flush=True)
                time.sleep(backoff)
