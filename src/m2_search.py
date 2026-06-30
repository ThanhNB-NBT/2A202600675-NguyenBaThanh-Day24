from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import math
import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    from underthesea import word_tokenize
    try:
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            segmented = segment_vietnamese(chunk["text"])
            tokens = [t.lower() for t in segmented.split() if t.strip()]
            self.corpus_tokens.append(tokens)
            
        from rank_bm25 import BM25Okapi
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = [t.lower() for t in segment_vietnamese(query).split() if t.strip()]
        scores = self.bm25.get_scores(tokenized_query)
        
        # Lọc scores[i] > 0 để bỏ docs không liên quan
        top_indices = sorted(
            [i for i in range(len(scores)) if scores[i] > 0],
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[idx]),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        print(f"  [M2] Connecting Qdrant {QDRANT_HOST}:{QDRANT_PORT}...", flush=True)
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            print(f"  [M2] Loading embedding model: {EMBEDDING_MODEL}...", flush=True)
            if EMBEDDING_MODEL.startswith("nvidia/"):
                self._encoder = NvidiaEmbedding()
                print("  [M2] NVIDIA embedding API ready.", flush=True)
                return self._encoder
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
                print("  [M2] Embedding model loaded.", flush=True)
            except BaseException as exc:
                print(f"  [M2] Embedding model failed: {type(exc).__name__}: {exc}", flush=True)
                print("  [M2] Falling back to local hash embeddings.", flush=True)
                self._encoder = HashEmbedding()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        print(f"  [M2] Encoding {len(texts)} chunks...", flush=True)
        vectors = self._get_encoder().encode(texts, show_progress_bar=True, input_type="passage")
        vector_size = EMBEDDING_DIM or len(vectors[0])
        print(f"  [M2] Recreating Qdrant collection: {collection} (dim={vector_size})...", flush=True)
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print("  [M2] Collection ready.", flush=True)
        print("  [M2] Encoding done; building Qdrant points...", flush=True)
        
        points = []
        for i, (v, c) in enumerate(zip(vectors, chunks)):
            points.append(PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]}
            ))
        print(f"  [M2] Upserting {len(points)} points to Qdrant...", flush=True)
        self.client.upsert(collection_name=collection, points=points)
        print("  [M2] Dense index complete.", flush=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query, input_type="query").tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        )
        return [
            SearchResult(
                text=pt.payload["text"],
                score=pt.score,
                metadata={k: v for k, v in pt.payload.items() if k != "text"},
                method="dense"
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                            top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
            
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["result"].metadata,
            method="hybrid"
        )
        for item in sorted_items
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        print("  [M2] Building BM25 index...", flush=True)
        self.bm25.index(chunks)
        print("  [M2] BM25 index complete.", flush=True)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")


class HashEmbedding:
    # ponytail: fallback keeps the lab runnable; use SentenceTransformer when the local runtime supports it.
    def encode(self, texts, show_progress_bar: bool = False, input_type: str = "passage"):
        single = isinstance(texts, str)
        items = [texts] if single else texts
        vectors = [self._encode_one(text) for text in items]
        return vectors[0] if single else vectors

    def _encode_one(self, text: str):
        vector = [0.0] * EMBEDDING_DIM
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return HashVector(v / norm for v in vector)


class HashVector(list):
    def tolist(self):
        return list(self)


class NvidiaEmbedding:
    # ponytail: avoids local torch crashes; switch EMBEDDING_MODEL to a local model if needed.
    def encode(self, texts, show_progress_bar: bool = False, input_type: str = "passage"):
        from config import openai_client

        single = isinstance(texts, str)
        items = [texts] if single else texts
        client = openai_client()
        vectors = []
        batch_size = 16
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch, extra_body={"input_type": input_type})
            vectors.extend(HashVector(item.embedding) for item in response.data)
            print(f"  [M2] NVIDIA embedded {min(start + batch_size, len(items))}/{len(items)}", flush=True)
        return vectors[0] if single else vectors
