"""
Task 4 — Chunking & Indexing vào Vector Store.

Nguồn dữ liệu:
    - Legal: bảng `chunks` trong data/standardized/legal/ragvbpl.sqlite. Đây là kết quả
      của pipeline `data_ingestion/` (riêng biệt với Task 1-3) — mỗi row đã là 1 đơn vị
      pháp lý structure-aware (1 Điều, hoặc 1 Khoản/Điểm nếu Điều quá dài), với citation
      header (tên văn bản + số Điều) đã được prepend vào `text` để cung cấp ngữ cảnh cho
      embedding. Không cần đọc lại qua Markdown — dùng thẳng SQLite.
    - News: markdown files trong data/standardized/news/ (Task 3, sinh ra sau).

Cấu hình đã chọn (khớp yêu cầu CP2 trong LAB_GUIDE.md):
    - Chunking: RecursiveCharacterTextSplitter, size=800, overlap=100 — các chunk pháp lý
      gốc dài trung bình ~1200 ký tự (tối đa ~2500), nên vẫn cần re-split để tương thích
      context window nhỏ khi generation; overlap=100 (~12%) đủ giữ liên tục ngữ nghĩa qua
      ranh giới câu mà không làm phình số lượng chunk.
    - Embedding: OpenAI `text-embedding-3-small` (1536 dim) via API — tránh tải/chạy model
      lớn (BAAI/bge-m3, ~2GB) cục bộ trên máy không có GPU; chất lượng multilingual đủ tốt
      cho tiếng Việt lẫn tiếng Anh, latency thấp vì chạy qua API thay vì CPU inference.
    - Vector store: ChromaDB, persistent local tại chroma_db/, cosine similarity.

Yêu cầu: biến môi trường OPENAI_API_KEY (trong .env, xem .env.example).

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import hashlib
import os
import sqlite3
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
SQLITE_PATH = STANDARDIZED_DIR / "legal" / "ragvbpl.sqlite"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI API — multilingual, không cần GPU cục bộ
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 100  # số text/request tới OpenAI Embeddings API

VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"

_openai_client = None  # cache — tránh khởi tạo lại client mỗi lần gọi


# =============================================================================
# SHARED HELPERS (dùng lại ở Task 5, 9, ...)
# =============================================================================

def get_openai_client():
    """Trả về OpenAI client đã khởi tạo (cached), đọc OPENAI_API_KEY từ .env."""
    global _openai_client
    if _openai_client is None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY chưa được thiết lập — thêm vào file .env (xem .env.example)."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một danh sách text qua OpenAI Embeddings API, theo batch, có retry khi rate-limited."""
    from openai import RateLimitError

    client = get_openai_client()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                break
            except RateLimitError:
                wait = 2 ** attempt
                print(f"  ⚠ Rate limited, retry sau {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("OpenAI Embeddings API: hết lượt retry do rate limit.")
        embeddings.extend(item.embedding for item in resp.data)

    return embeddings


def embed_query(text: str) -> list[float]:
    """Embed 1 query string (dùng ở Task 5 semantic_search)."""
    return embed_texts([text])[0]


def get_collection():
    """Trả về ChromaDB collection (persistent, tạo mới nếu chưa có)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def _load_legal_from_sqlite() -> list[dict]:
    """
    Đọc TOÀN BỘ bảng `chunks` (đã structure-aware chunked) từ ragvbpl.sqlite.

    Trước đây lọc theo `legal_domains LIKE '%"labor"%'`, nhưng domain tag chỉ được gán
    bằng keyword-match trên title/lĩnh vực/ngành của CHÍNH văn bản đó (xem
    `data_ingestion/scope.py::classify_domains`) — không lan sang các văn bản vào corpus
    qua relationship expansion (nghị định/thông tư hướng dẫn/sửa đổi một luật lao động
    seed) nhưng không lặp lại từ khoá "lao động" trong tiêu đề riêng của chúng. Kết quả:
    95/148 documents (64%) không có domain tag nào — KHÔNG phải do mất dữ liệu lúc
    chunking (chunk-level và document-level "labor" tag khớp chính xác 881/881), mà do
    bước classify domain ở thượng nguồn vốn đã không đầy đủ. Không lọc domain ở đây nữa
    — index toàn bộ corpus.
    """
    if not SQLITE_PATH.exists():
        return []

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM chunks").fetchall()
    conn.close()

    documents = []
    for row in rows:
        r = dict(row)
        source_label = f"{r['document_number']} - Đ.{r['article_no']}"
        if r.get("clause_no"):
            source_label += f" K.{r['clause_no']}"
        documents.append({
            "content": r["text"],
            "metadata": {
                "source": source_label,
                "type": "legal",
                "document_number": r.get("document_number") or "",
                "document_type": r.get("document_type") or "",
                "title": r.get("title") or "",
                "issuing_authority": r.get("issuing_authority") or "",
                "validity_status": r.get("validity_status") or "",
                "legal_domains": r.get("legal_domains") or "",
                "chapter_no": r.get("chapter_no") or "",
                "chapter_title": r.get("chapter_title") or "",
                "article_no": r.get("article_no") or 0,
                "article_title": r.get("article_title") or "",
                "clause_no": r.get("clause_no") or "",
                "point": r.get("point") or "",
                "source_url": r.get("source_url") or "",
                "sqlite_chunk_id": r.get("chunk_id") or "",
            },
        })
    return documents


def _load_news_markdown() -> list[dict]:
    """Đọc markdown files trong data/standardized/news/ (Task 3)."""
    news_dir = STANDARDIZED_DIR / "news"
    if not news_dir.exists():
        return []

    documents = []
    for md_file in news_dir.rglob("*.md"):
        documents.append({
            "content": md_file.read_text(encoding="utf-8"),
            "metadata": {"source": md_file.name, "type": "news"},
        })
    return documents


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ nguồn: legal (SQLite, đã chunk sẵn) + news (Markdown, Task 3).

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    return _load_legal_from_sqlite() + _load_news_markdown()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Re-split mỗi document theo CHUNK_SIZE/CHUNK_OVERLAP (RecursiveCharacterTextSplitter).

    Với legal, mỗi "document" đầu vào đã là 1 đơn vị Điều/Khoản/Điểm — bước này chỉ
    cắt tiếp nếu vượt CHUNK_SIZE, để đảm bảo context window nhỏ khi generation.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk cuối cùng.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng OpenAI text-embedding-3-small.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks qua OpenAI API "
          f"({(len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE} requests)...")
    embeddings = embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def _chunk_id(chunk: dict, fallback_index: int) -> str:
    """ID ổn định qua các lần reindex (upsert-idempotent), tránh ký tự lạ trong id Chroma."""
    meta = chunk["metadata"]
    base = meta.get("sqlite_chunk_id") or meta.get("source") or f"doc-{fallback_index}"
    raw = f"{base}|{meta.get('chunk_index', fallback_index)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def index_to_vectorstore(chunks: list[dict]):
    """Upsert toàn bộ chunks (đã có 'embedding') vào ChromaDB collection."""
    collection = get_collection()

    BATCH = 500
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        ids = [_chunk_id(c, start + j) for j, c in enumerate(batch)]
        collection.upsert(
            ids=ids,
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print(f"✓ Indexed to vector store ({CHROMA_DIR})")


if __name__ == "__main__":
    run_pipeline()
