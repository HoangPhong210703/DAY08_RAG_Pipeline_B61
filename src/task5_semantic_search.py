"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import math

from .task4_chunking_indexing import chunk_documents, embed_query, embed_texts, get_collection, load_documents

_local_corpus: list[dict] | None = None
_local_vectors: list[list[float]] | None = None


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip() or top_k <= 0:
        return []

    # Chroma/remote embeddings are opt-in so importing the lab never requires
    # a service, API key, or a stale index created by another configuration.
    use_chroma = __import__("os").getenv("RAG_USE_CHROMA", "").lower() in {"1", "true", "yes"}
    if use_chroma:
        try:
            return _search_chroma(query, top_k)
        except (ImportError, RuntimeError, KeyError, ValueError, OSError) as exc:
            # A local deterministic path keeps retrieval usable offline.
            print(f"  ⚠ Chroma semantic search unavailable ({exc}); dùng local index.")

    return _search_local(query, top_k)


def _search_chroma(query: str, top_k: int) -> list[dict]:
    query_vector = embed_query(query)

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results.get("documents") or not results["documents"][0]:
        return []

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, 1.0 - dist)  # cosine distance → similarity
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


def _search_local(query: str, top_k: int) -> list[dict]:
    """Dense-like retrieval fallback using the same local hash embeddings."""
    global _local_corpus, _local_vectors
    if _local_corpus is None:
        _local_corpus = chunk_documents(load_documents())
        _local_vectors = embed_texts([item["content"] for item in _local_corpus])
    if not _local_corpus or _local_vectors is None:
        return []

    query_vector = embed_query(query)
    scored = []
    for index, (item, vector) in enumerate(zip(_local_corpus, _local_vectors)):
        score = sum(a * b for a, b in zip(query_vector, vector))
        # Cosine similarity is exposed on [0, 1], matching the Chroma path.
        scored.append((max(0.0, min(1.0, score)), index))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    return [
        {
            "content": _local_corpus[index]["content"],
            "score": round(score, 4),
            "metadata": dict(_local_corpus[index].get("metadata", {})),
        }
        for score, index in scored[:top_k]
    ]


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
