"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re

from .task4_chunking_indexing import chunk_documents, load_documents

# Cùng corpus (đã re-split 800/100) với Task 4/5 — để lexical và semantic search so
# sánh được trên đúng 1 tập chunk khi merge ở Task 9 (hybrid retrieval).
_bm25 = None
_corpus: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Lowercase + tách từ bằng \\w+ (bỏ dấu câu dính vào từ, tốt hơn split() thô)."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_index():
    """Lazy-build BM25 index (cached) từ cùng nguồn dữ liệu với Task 4."""
    global _bm25, _corpus
    if _bm25 is None:
        docs = load_documents()
        _corpus = chunk_documents(docs)
        _bm25 = build_bm25_index(_corpus)
    return _bm25, _corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25, corpus = _get_index()
    scores = bm25.get_scores(_tokenize(query))

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            break  # ranked_indices đã sort giảm dần, không còn gì > 0 phía sau
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"],
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("bảo hiểm xã hội thai sản", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
