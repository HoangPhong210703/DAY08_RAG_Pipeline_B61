"""SQLite store for the canonical corpus (POC).

Tables: ``documents`` (canonical metadata + content + provenance), ``relationships``, and
``articles_ref`` (Pháp điển validation reference). See schema in ``doc/data-src.md`` §6.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ragvbpl.models import ArticleRef, CanonicalDocument, Relationship

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_number            TEXT PRIMARY KEY,
    canonical_document_id      TEXT,
    document_type              TEXT,
    title                      TEXT,
    issuing_authority          TEXT,
    issued_date                TEXT,
    effective_date             TEXT,
    expiry_date                TEXT,
    validity_status            TEXT,
    legal_domains              TEXT,   -- JSON array
    canonical_content          TEXT,
    canonical_content_source   TEXT,
    metadata_source            TEXT,
    structure_validation_source TEXT,
    source_datasets            TEXT,   -- JSON array
    official_source_url        TEXT,
    content_hash               TEXT,
    content_conflict           INTEGER,
    parse_validation_status    TEXT
);
CREATE TABLE IF NOT EXISTS relationships (
    doc_number        TEXT,
    other_doc_number  TEXT,
    relationship      TEXT
);
CREATE TABLE IF NOT EXISTS articles_ref (
    record_id         TEXT PRIMARY KEY,
    article_id        TEXT,
    article_title     TEXT,
    topic_title_vi    TEXT,
    subject_title_vi  TEXT,
    content_text      TEXT,
    source_url        TEXT
);
CREATE TABLE IF NOT EXISTS review_queue (
    document_number   TEXT,
    reason            TEXT
);
CREATE TABLE IF NOT EXISTS parsed_articles (
    document_number   TEXT,
    article_no        INTEGER,
    article_title     TEXT,
    chapter_no        TEXT,
    chapter_title     TEXT,
    section_no        TEXT,
    section_title     TEXT,
    n_khoan           INTEGER,
    n_diem            INTEGER,
    char_len          INTEGER,
    article_text      TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    document_number   TEXT,
    document_type     TEXT,
    title             TEXT,
    issuing_authority TEXT,
    validity_status   TEXT,
    legal_domains     TEXT,
    chapter_no        TEXT,
    chapter_title     TEXT,
    article_no        INTEGER,
    article_title     TEXT,
    clause_no         TEXT,
    point             TEXT,
    source_url        TEXT,
    canonical_source  TEXT,
    char_len          INTEGER,
    text              TEXT
);
"""


def connect(sqlite_path: str | Path) -> sqlite3.Connection:
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def upsert_documents(conn: sqlite3.Connection, docs: list[CanonicalDocument]) -> None:
    rows = []
    review = []
    for d in docs:
        rows.append(
            (
                d.document_number,
                d.canonical_document_id,
                d.document_type,
                d.title,
                d.issuing_authority,
                d.issued_date,
                d.effective_date,
                d.expiry_date,
                d.validity_status,
                json.dumps(d.legal_domains, ensure_ascii=False),
                d.canonical_content,
                d.canonical_content_source,
                d.metadata_source,
                d.structure_validation_source,
                json.dumps(d.source_datasets, ensure_ascii=False),
                d.official_source_url,
                d.content_hash,
                int(d.content_conflict),
                d.parse_validation_status,
            )
        )
        if d.canonical_document_id is None:
            review.append((d.document_number, "missing key field for canonical_document_id"))
        if not d.canonical_content:
            review.append((d.document_number, "no content (empty source body)"))

    # Each ingestion rebuilds the whole canonical set — wipe first so stale rows don't accumulate.
    conn.execute("DELETE FROM documents")
    conn.execute("DELETE FROM review_queue")
    conn.executemany(
        """INSERT OR REPLACE INTO documents VALUES
           (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    if review:
        conn.executemany("INSERT INTO review_queue VALUES (?,?)", review)
    conn.commit()


def replace_relationships(conn: sqlite3.Connection, rels: list[Relationship]) -> None:
    conn.execute("DELETE FROM relationships")
    conn.executemany(
        "INSERT INTO relationships VALUES (?,?,?)",
        [(r.doc_number, r.other_doc_number, r.relationship) for r in rels],
    )
    conn.commit()


def replace_articles_ref(conn: sqlite3.Connection, articles: list[ArticleRef]) -> None:
    conn.execute("DELETE FROM articles_ref")
    conn.executemany(
        "INSERT OR REPLACE INTO articles_ref VALUES (?,?,?,?,?,?,?)",
        [
            (a.record_id, a.article_id, a.article_title, a.topic_title_vi,
             a.subject_title_vi, a.content_text, a.source_url)
            for a in articles
        ],
    )
    conn.commit()


def replace_parsed_articles(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Rows are (document_number, article_no, article_title, chapter_no, chapter_title,
    section_no, section_title, n_khoan, n_diem, char_len, article_text)."""
    conn.execute("DELETE FROM parsed_articles")
    conn.executemany(
        "INSERT INTO parsed_articles VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()


def update_parse_status(conn: sqlite3.Connection, statuses: dict[str, str]) -> None:
    """Set documents.parse_validation_status for each {document_number: status}."""
    conn.executemany(
        "UPDATE documents SET parse_validation_status=? WHERE document_number=?",
        [(st, dn) for dn, st in statuses.items()],
    )
    conn.commit()


def iter_documents_with_content(conn: sqlite3.Connection):
    """Yield (document_number, canonical_content) for docs that have content."""
    cur = conn.execute(
        "SELECT document_number, canonical_content FROM documents "
        "WHERE canonical_content IS NOT NULL AND length(canonical_content) > 0"
    )
    yield from cur


def iter_passed_articles(conn: sqlite3.Connection):
    """Yield (meta_dict, article_dict) for every parsed Điều of parse-passed documents."""
    import json
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """SELECT p.*, d.document_type, d.title, d.issuing_authority, d.validity_status,
                  d.legal_domains, d.official_source_url, d.canonical_content_source
           FROM parsed_articles p JOIN documents d USING (document_number)
           WHERE d.parse_validation_status = 'passed'
           ORDER BY p.document_number, p.article_no"""
    )
    for r in cur:
        meta = {
            "document_number": r["document_number"],
            "document_type": r["document_type"],
            "title": r["title"],
            "issuing_authority": r["issuing_authority"],
            "validity_status": r["validity_status"],
            "legal_domains": json.loads(r["legal_domains"] or "[]"),
            "official_source_url": r["official_source_url"],
            "canonical_content_source": r["canonical_content_source"],
        }
        art = {
            "article_no": r["article_no"],
            "article_title": r["article_title"],
            "chapter_no": r["chapter_no"],
            "chapter_title": r["chapter_title"],
            "article_text": r["article_text"],
        }
        yield meta, art


def count_passed_articles(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT count(*) FROM parsed_articles p JOIN documents d USING (document_number) "
        "WHERE d.parse_validation_status='passed'"
    ).fetchone()[0]


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def iter_chunks(conn: sqlite3.Connection, docs: list[str] | None = None):
    """Yield each chunk row (sqlite3.Row) for indexing; optionally only for ``docs``."""
    conn.row_factory = sqlite3.Row
    if docs:
        ph = ",".join("?" * len(docs))
        yield from conn.execute(
            f"SELECT * FROM chunks WHERE document_number IN ({ph}) "
            "ORDER BY document_number, article_no", docs,
        )
    else:
        yield from conn.execute("SELECT * FROM chunks ORDER BY document_number, article_no")


def replace_chunks(conn: sqlite3.Connection, chunks) -> None:
    import json
    conn.execute("DELETE FROM chunks")
    conn.executemany(
        "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (c.chunk_id, c.document_number, c.document_type, c.title, c.issuing_authority,
             c.validity_status, json.dumps(c.legal_domains, ensure_ascii=False),
             c.chapter_no, c.chapter_title, c.article_no, c.article_title, c.clause_no,
             c.point, c.source_url, c.canonical_source, c.char_len, c.text)
            for c in chunks
        ],
    )
    conn.commit()
