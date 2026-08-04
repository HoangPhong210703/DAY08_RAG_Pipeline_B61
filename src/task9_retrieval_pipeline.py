"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Đã tự đo điểm cosine top-1 của semantic_search trên corpus lao-động thật (OpenAI
# text-embedding-3-small), 5 câu liên quan vs 5 câu lạc đề:
#   - Liên quan (thai sản, công đoàn, trợ cấp thất nghiệp, giờ làm việc, ATVSLĐ):
#     0.353 – 0.486
#   - Lạc đề nhưng vẫn là câu tiếng Việt mạch lạc (nấu ăn, cây cảnh, world cup, sửa xe):
#     0.328 – 0.417  ← CHỒNG LẤN gần hết với nhóm "liên quan" ở trên
#   - Gibberish thật sự ("xyzabc123nonsense"): 0.167
# → Với embedding/corpus này, cosine similarity KHÔNG tách được "liên quan" khỏi
#   "lạc đề nhưng mạch lạc" — chỉ tách được "văn bản có nghĩa" khỏi "gibberish". Đây là
#   giới hạn thật của phương pháp (không phải bug), nên threshold được đặt ngay trên mức
#   gibberish để fallback chỉ kích hoạt cho input thực sự vô nghĩa/không parse được.
SCORE_THRESHOLD = 0.25   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    mode: str = "hybrid",  # "hybrid" | "semantic" | "lexical" — cho phép so sánh A/B trên UI
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline (mode="hybrid", mặc định):
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    mode="semantic" / "lexical": chỉ chạy 1 nhánh, KHÔNG merge/rerank — trả điểm gốc
    (cosine hoặc BM25) chưa qua RRF, để so sánh trực tiếp 2 phương pháp trên UI.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không (chỉ áp dụng ở mode="hybrid")
        mode: "hybrid" | "semantic" | "lexical"

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' | 'semantic' | 'lexical' | 'pageindex'
        }
    """
    dense_results = semantic_search(query, top_k=top_k * 2) if mode in ("hybrid", "semantic") else []
    sparse_results = lexical_search(query, top_k=top_k * 2) if mode in ("hybrid", "lexical") else []

    if mode == "semantic":
        merged = [{**r, "source": "semantic"} for r in dense_results]
    elif mode == "lexical":
        merged = [{**r, "source": "lexical"} for r in sparse_results]
    else:
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
        for item in merged:
            item["source"] = "hybrid"

    # Rerank (giữ 'source' — rerank() copy nguyên dict, chỉ ghi đè 'score'). Chỉ áp
    # dụng ở mode="hybrid" — semantic/lexical-only là để xem điểm gốc, không rerank.
    if mode == "hybrid" and use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Fallback threshold DÙNG ĐIỂM COSINE GỐC (dense_results), KHÔNG PHẢI RRF — chỉ có
    # ý nghĩa khi có dense_results (mode="lexical" không tính cosine nên bỏ qua bước này).
    if dense_results:
        best_score = dense_results[0]["score"]
        if best_score < score_threshold:
            print(f"  ⚠ Semantic best score ({best_score:.3f}) < threshold ({score_threshold}) → thử PageIndex fallback")
            try:
                fallback = pageindex_search(query, top_k=top_k)
            except Exception as e:
                print(f"  ⚠ PageIndex fallback không khả dụng ({e}), giữ kết quả {mode}")
                fallback = []
            if fallback:
                return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
