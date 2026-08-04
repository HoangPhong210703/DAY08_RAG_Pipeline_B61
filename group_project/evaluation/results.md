# RAG Evaluation Results

## Framework sử dụng

> RAGAS-compatible heuristic eval

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Δ |
|--------|---------------------------|-----------------------------|---|
| Faithfulness | 0.0861 | 0.0859 | +0.0002 |
| Answer Relevance | 0.5298 | 0.5182 | +0.0116 |
| Context Recall | 0.8589 | 0.8589 | +0.0000 |
| Context Precision | 0.5777 | 0.5777 | +0.0000 |
| Average | 0.5131 | 0.5102 | +0.0029 |

---

## A/B Comparison Analysis

**Config A:**
> Hybrid retrieval + reranking

**Config B:**
> Hybrid retrieval without reranking

**Kết luận:**
> Config A tốt hơn theo điểm trung bình tổng hợp. Nếu chênh lệch chủ yếu nằm ở Faithfulness/Precision, reranking đang giúp loại bớt context nhiễu; nếu Recall giảm, cần tăng `top_k` hoặc nới ngưỡng fallback.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Công ty nhắn qua Zalo rằng nhân viên bị sa thải ngay lập tức, không nêu lý do và không báo trước. Việc này có hợp pháp không? | 0.0744 | 0.3339 | 0.6071 | Generation | Answer is not sufficiently grounded in retrieved context |
| 2 | Công ty có được giữ bản gốc căn cước công dân, bằng đại học hoặc yêu cầu người lao động đặt cọc khi nhận việc không? | 0.0011 | 0.0872 | 0.8261 | Generation | Answer is not sufficiently grounded in retrieved context |
| 3 | Công ty có thể sa thải người lao động chỉ vì không thích thái độ làm việc của họ không? | 0.0888 | 0.3610 | 0.7500 | Generation | Answer is not sufficiently grounded in retrieved context |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng chất lượng chunking hoặc điều chỉnh top_k cho retrieval.
**Expected impact:** Cải thiện Context Recall và giảm trường hợp thiếu evidence.

### Cải tiến 2
**Action:** Giữ reranking cho các query có nhiều candidate, nhưng bỏ qua khi điểm retrieval đã quá thấp.
**Expected impact:** Giảm nhiễu context và tăng Faithfulness/Precision.

### Cải tiến 3
**Action:** Khi Task 10 hoàn thiện, dùng prompt có citation chặt hơn và kiểm tra answer against context trước khi xuất.
**Expected impact:** Tăng Answer Relevance và giảm hallucination.
