"""
Task 8 — PageIndex Vectorless RAG (local structure-aware fallback).

PageIndex SDK thật (pageindex.ai) cần đăng ký tài khoản + API key trả phí/giới hạn free
tier — không có sẵn trong `.env` của bài lab này (PAGEINDEX_API_KEY để trống). Thay vì
giả lập, triển khai lại đúng NGUYÊN LÝ mà PageIndex mô tả (xem LAB_GUIDE.md — "Vectorless
RAG: đọc hiểu tài liệu theo chương, mục và tiêu đề mà KHÔNG chunking"):

    - Điều hướng theo cấu trúc tài liệu (document → chapter → article), so khớp query
      với TIÊU ĐỀ (table-of-contents), không phải nội dung đầy đủ.
    - Trả về NGUYÊN VĂN article_text của node khớp nhất — không chunk, không embedding.

Khác biệt với Task 5 (dense/embedding) và Task 6 (BM25 trên nội dung đầy đủ của chunk
800 ký tự): ở đây match chỉ diễn ra trên tiêu đề (document title + chapter title +
article title), và nguồn dữ liệu là bảng `parsed_articles` — mức "1 Điều = 1 row, giữ
nguyên toàn văn", CHƯA bị chunk như bảng `chunks` dùng ở Task 4/5/6.

Nếu sau này có PAGEINDEX_API_KEY thật, có thể thay `_get_structure_index`/`pageindex_search`
bằng lời gọi PageIndex SDK — xem https://github.com/VectifyAI/PageIndex. Lưu ý API
`/retrieval` của PageIndex đã deprecated (vẫn hoạt động) — response nằm trong
"retrieved_nodes" → "relevant_contents": list[list[{section_title, relevant_content}]].
"""

import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")  # không dùng ở local fallback này
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
SQLITE_PATH = STANDARDIZED_DIR / "legal" / "ragvbpl.sqlite"

_structure_index: list[dict] | None = None  # cache — tránh đọc lại SQLite mỗi lần gọi


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _load_structure_index() -> list[dict]:
    """
    Đọc cây cấu trúc (document → chapter → article) từ bảng `parsed_articles`
    (JOIN `documents`), lọc domain "labor" (khớp Task 4/6). Giữ nguyên toàn văn
    `article_text` — không chunk.
    """
    if not SQLITE_PATH.exists():
        return []

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT pa.document_number, pa.article_no, pa.chapter_no, pa.chapter_title,
               pa.article_title, pa.article_text, d.title AS document_title
        FROM parsed_articles pa
        JOIN documents d ON pa.document_number = d.document_number
        WHERE d.legal_domains LIKE '%"labor"%'
        """
    ).fetchall()
    conn.close()

    nodes = []
    for r in rows:
        title_path = " - ".join(
            p for p in [r["document_title"], r["chapter_title"], r["article_title"]] if p
        )
        nodes.append({
            "title_tokens": _tokenize(title_path),
            "content": f"{title_path}\n{r['article_text']}",
            "metadata": {
                "source": f"{r['document_number']} - Đ.{r['article_no']}",
                "type": "legal",
                "document_number": r["document_number"],
                "title": r["document_title"],
                "chapter_no": r["chapter_no"] or "",
                "chapter_title": r["chapter_title"] or "",
                "article_no": r["article_no"],
                "article_title": r["article_title"] or "",
            },
        })
    return nodes


def _get_structure_index() -> list[dict]:
    global _structure_index
    if _structure_index is None:
        _structure_index = _load_structure_index()
    return _structure_index


def upload_documents():
    """
    Ở local fallback này, "upload" = build cây cấu trúc trong bộ nhớ từ SQLite (không có
    bước gửi file lên server). Với PageIndex SDK thật, hàm này sẽ submit PDF lên
    pageindex.ai và lưu lại doc_id cho từng tài liệu.
    """
    index = _get_structure_index()
    print(f"✓ Đã dựng cây cấu trúc: {len(index)} article-nodes (domain: labor)")
    return index


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval: so khớp query với TIÊU ĐỀ (document/chapter/article title) theo
    tỷ lệ trùng từ khoá — không dùng embedding, không chunk. Trả về nguyên văn
    article_text của (các) node khớp nhất.

    Dùng làm fallback khi hybrid search (Task 9) không có kết quả đủ tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if not query or not query.strip() or top_k <= 0:
        return []
    index = _get_structure_index()
    query_tokens = _tokenize(query)
    if not query_tokens or not index:
        return []

    scored = []
    for node in index:
        overlap = query_tokens & node["title_tokens"]
        if not overlap:
            continue
        # Tỷ lệ số từ query khớp được trong tiêu đề — không bị "loãng" bởi tiêu đề dài.
        score = len(overlap) / len(query_tokens)
        scored.append((score, node))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, node in scored[:top_k]:
        results.append({
            "content": node["content"],
            "score": round(score, 4),
            "metadata": node["metadata"],
            "source": "pageindex",
        })
    return results


if __name__ == "__main__":
    upload_documents()
    print()
    for q in ["thời gian nghỉ thai sản", "quyền của công đoàn", "an toàn vệ sinh lao động"]:
        print(f"Query: {q}")
        for r in pageindex_search(q, top_k=3):
            print(f"  [{r['score']:.3f}] {r['metadata']['source']} - {r['content'][:80]}")
        print()
