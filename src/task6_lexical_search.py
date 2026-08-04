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
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return _SimpleBM25(tokenized_corpus)
    return BM25Okapi(tokenized_corpus)


class _SimpleBM25:
    """BM25Okapi-compatible fallback for the minimal lab environment."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        from collections import Counter
        import math

        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(tokens) for tokens in corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if corpus else 0.0
        document_frequency = Counter(token for tokens in corpus for token in set(tokens))
        self.idf = {
            token: math.log(1.0 + (len(corpus) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        from collections import Counter

        scores: list[float] = []
        for tokens, doc_length in zip(self.corpus, self.doc_lengths):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_length / (self.avgdl or 1.0)
                )
                score += self.idf.get(token, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            scores.append(score)
        return scores


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
    if not query or not query.strip() or top_k <= 0:
        return []

    bm25, corpus = _get_index()
    if not corpus:
        return []
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
