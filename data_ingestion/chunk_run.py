"""M4a — chunking only.

Reads parse-passed documents from SQLite, builds Điều/Khoản/Điểm chunks, and writes them to the
``chunks`` table. Run this first and inspect before the (slow) embedding step.

Run:  python -m ragvbpl.chunk.run
"""

from __future__ import annotations

import argparse
import logging

from ragvbpl.config import load_settings
from ragvbpl.progress import PctBar
from ragvbpl.store import db

from .chunker import chunk_article

log = logging.getLogger("ragvbpl.chunk")


def run() -> dict:
    s = load_settings()
    sqlite_path = s.config.paths.get("sqlite_path", "data/ragvbpl.sqlite")
    max_chars = s.config.chunking.get("max_chars", 2000)

    # unit: "article" → Điều-only (one chunk per Điều); "clause" → split by Khoản/Điểm.
    split_clauses = s.config.chunking.get("unit", "article") != "article"

    conn = db.connect(sqlite_path)
    n_art = db.count_passed_articles(conn)

    chunks = []
    bar = PctBar("Chunking", n_art)
    for i, (meta, art) in enumerate(db.iter_passed_articles(conn), 1):
        chunks.extend(chunk_article(meta, art, max_chars=max_chars, split_clauses=split_clauses))
        bar.update(i)
    bar.close(n_art)

    db.replace_chunks(conn, chunks)
    conn.close()

    sizes = [c.char_len for c in chunks] or [0]
    summary = {
        "unit": "clause" if split_clauses else "article",
        "passed_articles": n_art,
        "chunks": len(chunks),
        "avg_chars": sum(sizes) // len(sizes),
        "max_chars": max(sizes),
        "sqlite_path": sqlite_path,
    }
    log.info("Done: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG-vbpl M4a chunking")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(run())


if __name__ == "__main__":
    main()
