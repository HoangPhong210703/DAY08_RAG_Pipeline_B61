"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import math
import re


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if top_k <= 0 or not candidates:
        return []
    query_tokens = _tokens(query)
    scored = []
    for position, candidate in enumerate(candidates):
        content_tokens = _tokens(candidate.get("content", ""))
        overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
        phrase_bonus = 0.1 if query.strip().casefold() in candidate.get("content", "").casefold() else 0.0
        original = float(candidate.get("score", 0.0) or 0.0)
        relevance = min(1.0, 0.75 * overlap + 0.15 * phrase_bonus + 0.10 * max(0.0, original))
        item = dict(candidate)
        item["score"] = round(relevance, 6)
        scored.append((relevance, position, item))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [item for _, _, item in scored[:top_k]]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []
    lambda_param = max(0.0, min(1.0, lambda_param))
    vectors = [candidate.get("embedding") for candidate in candidates]
    missing = [index for index, vector in enumerate(vectors) if not vector]
    if missing:
        from .task4_chunking_indexing import embed_texts

        generated = embed_texts([candidates[index].get("content", "") for index in missing])
        for index, vector in zip(missing, generated):
            vectors[index] = vector

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_index = None
        best_mmr = float("-inf")
        for index in remaining:
            vector = vectors[index] or []
            relevance = _cosine(query_embedding, vector)
            redundancy = max(
                (_cosine(vector, vectors[selected_index] or []) for selected_index in selected),
                default=0.0,
            )
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * redundancy
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)

    results = []
    for index in selected:
        item = dict(candidates[index])
        item["score"] = round(
            lambda_param * _cosine(query_embedding, vectors[index] or []), 6
        )
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0:
        return []
    k = max(1, k)
    rrf_scores: dict[str, float] = {}   # stable document key -> fused score
    content_map: dict[str, dict] = {}   # stable key -> full dict (first occurrence wins)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = _document_key(item)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, score in sorted_items[:top_k]:
        item = dict(content_map[key])
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        from .task4_chunking_indexing import embed_query

        return rerank_mmr(embed_query(query), candidates, top_k)
    elif method == "rrf":
        # Trường hợp gọi rerank() với 1 danh sách phẳng (không phải merge nhiều
        # ranker): coi candidates là 1 ranked list duy nhất (sort theo score gốc để
        # xác định rank), rồi áp công thức RRF — đây là trường hợp suy biến của RRF
        # (1 input list). Để MERGE nhiều ranked lists (dense + sparse ở Task 9),
        # gọi rerank_rrf([list1, list2, ...]) trực tiếp thay vì qua đây.
        ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
        return rerank_rrf([ranked], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _document_key(item: dict) -> str:
    metadata = item.get("metadata") or {}
    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if source is not None and chunk_index is not None:
        return f"{source}|{chunk_index}"
    return str(item.get("content", ""))


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
