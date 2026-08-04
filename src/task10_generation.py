"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Model ID khác nhau tuỳ provider (OpenRouter prefix "openai/", OpenAI thì không).
# .env chỉ có OPENAI_API_KEY (không có OPENROUTER_API_KEY) — xem _get_llm_client().
LLM_MODEL_OPENROUTER = "openai/gpt-4o-mini"
LLM_MODEL_OPENAI = "gpt-4o-mini"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================
# Domain đã đổi sang pháp luật lao động Việt Nam (khớp corpus Task 4 — xem context.md),
# không phải "university services" như bản gốc của bài lab.

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý, trả lời câu hỏi về pháp luật lao động Việt Nam
(Bộ luật Lao động, Luật Bảo hiểm xã hội, Luật Công đoàn, An toàn vệ sinh lao động, ...).

Quy tắc bắt buộc:
1. CHỈ sử dụng thông tin có trong context được cung cấp — KHÔNG bịa đặt, KHÔNG suy diễn
   ngoài những gì được nêu trong context.
2. Mỗi khẳng định phải có trích dẫn ngay sau, dùng đúng nhãn "Source" trong context,
   ví dụ: [45/2019/QH14 - Đ.105].
3. Nếu context không đủ thông tin để trả lời câu hỏi → trả lời chính xác câu (giữ
   nguyên tiếng Anh): "I cannot verify this information"
4. Trả lời bằng tiếng Việt (trừ câu fallback ở mục 3), có cấu trúc rõ ràng theo đoạn văn."""


def _get_llm_client():
    """Ưu tiên OPENROUTER_API_KEY (model :free) nếu có, fallback OPENAI_API_KEY."""
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"), LLM_MODEL_OPENROUTER

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key), LLM_MODEL_OPENAI

    raise RuntimeError(
        "Cần OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env để gọi LLM generation."
    )


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]   # index 0, 2, 4, ... -> đầu, giữ nguyên thứ tự (tốt nhất trước)
    back = chunks[1::2]   # index 1, 3, ...    -> cuối, đảo ngược (nhì tốt nhất ở cuối cùng)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    mode: str = "hybrid",  # "hybrid" | "semantic" | "lexical" — xem task9_retrieval_pipeline.retrieve
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context
        temperature, top_p: tham số sinh của LLM (mặc định xem CONFIGURATION ở trên)
        score_threshold, use_reranking, mode: pass-through cho Task 9 retrieve() — cho
            phép UI thử nghiệm so sánh hybrid/semantic/lexical, có/không rerank, ngưỡng
            fallback khác nhau

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' | 'semantic' | 'lexical' | 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
        mode=mode,
    )

    # Step 2: Reorder (tránh lost in the middle)
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM
    client, model = _get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    answer = response.choices[0].message.content

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Thời giờ làm việc bình thường tối đa của người lao động là bao nhiêu?",
        "Lương thử việc tối thiểu phải bằng bao nhiêu phần trăm lương của công việc đó?",
        "Học phí tại RMIT Vietnam là bao nhiêu?",  # ngoài domain -> kỳ vọng "cannot verify"
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
