"""
RAG Chatbot — Pháp luật Lao động Việt Nam
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Domain đã đổi từ "university services" (bản gốc bài lab) sang pháp luật lao động Việt
Nam — khớp corpus thật đang được index ở Task 4 (ragvbpl.sqlite, xem context.md).

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Labor Law RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("⚖️ Pháp Luật Lao Động RAG")
    st.caption("Trợ lý hỏi đáp về pháp luật lao động Việt Nam (Bộ luật Lao động, BHXH, Công đoàn, ATVSLĐ)")


    # no need for suggestions
    # st.subheader("💡 Câu hỏi gợi ý")
    # suggestions = [
    #     "Thời giờ làm việc bình thường tối đa của người lao động là bao nhiêu?",
    #     "Lương thử việc tối thiểu phải bằng bao nhiêu phần trăm lương của công việc đó?",
    #     "Người lao động muốn đơn phương chấm dứt hợp đồng lao động thì phải báo trước bao lâu?",
    #     "Làm thêm giờ vào ngày lễ thì được trả ít nhất bao nhiêu phần trăm lương?",
    #     "Người lao động làm đủ 12 tháng được nghỉ phép năm bao nhiêu ngày?",
    # ]
    # for s in suggestions:
    #     if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
    #         st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")

    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    mode_label = st.radio(
        "Chế độ truy xuất",
        ["Hybrid (Semantic + BM25)", "Chỉ Semantic", "Chỉ Lexical (BM25)"],
        help="So sánh Hybrid retrieval với từng phương pháp riêng lẻ trên cùng 1 câu hỏi.",
    )
    mode = {
        "Hybrid (Semantic + BM25)": "hybrid",
        "Chỉ Semantic": "semantic",
        "Chỉ Lexical (BM25)": "lexical",
    }[mode_label]

    use_reranking = st.checkbox(
        "Bật RRF Reranking", value=True,
        help="Chỉ áp dụng ở chế độ Hybrid.", disabled=(mode != "hybrid"),
    )

    score_threshold = st.slider(
        "Ngưỡng fallback PageIndex (cosine)", 0.0, 1.0, 0.25, step=0.05,
        help="Nếu điểm cosine cao nhất của Semantic Search thấp hơn ngưỡng này, hệ thống thử fallback sang PageIndex.",
    )

    with st.expander("🎛️ Tham số LLM"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)
        top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, step=0.05)

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("⚖️ Pháp Luật Lao Động RAG Chatbot")
st.caption("Hệ thống hỏi đáp pháp luật lao động Việt Nam (Bộ luật Lao động, BHXH, Công đoàn, An toàn vệ sinh lao động)")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
                    st.text(src.get("content", "")[:300] + "...")
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

# Xử lý khi bấm nút gợi ý hoặc nhập câu hỏi mới
user_input = st.chat_input("Nhập câu hỏi của bạn về pháp luật lao động...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    score_threshold=score_threshold,
                    use_reranking=use_reranking,
                    mode=mode,
                )
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", mode)

            except NotImplementedError:
                answer = "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành `src/task10_generation.py` để kết nối pipeline vào UI!"
                sources = []
                retrieval_source = None
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []
                retrieval_source = None

            st.markdown(answer)

            if retrieval_source:
                st.caption(f"🔎 Nguồn truy xuất: `{retrieval_source}`")

            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        doc_type = meta.get("type", "unknown")
                        score = src.get("score", 0)
                        st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
                        st.text(src.get("content", "")[:300] + "...")
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
