"""M3 parser orchestrator.

Reads canonical_content from SQLite, parses Điều/Khoản/Điểm, writes the parsed_articles table, and
sets documents.parse_validation_status (passed | failed | no_content).

Run:  python -m ragvbpl.parse.run
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from ragvbpl.config import load_settings
from ragvbpl.store import db

from .articles import parse_document, validate

log = logging.getLogger("ragvbpl.parse")


def run() -> dict:
    s = load_settings()
    sqlite_path = s.config.paths.get("sqlite_path", "data/ragvbpl.sqlite")
    conn = db.connect(sqlite_path)

    rows: list[tuple] = []
    statuses: dict[str, str] = {}
    status_counts: Counter = Counter()
    total_articles = 0

    # Docs without content → no_content.
    for (dn,) in conn.execute(
        "SELECT document_number FROM documents "
        "WHERE canonical_content IS NULL OR length(canonical_content)=0"
    ).fetchall():
        statuses[dn] = "no_content"
        status_counts["no_content"] += 1

    for dn, content in db.iter_documents_with_content(conn):
        articles = parse_document(content)
        status, issue = validate(articles)
        statuses[dn] = status
        status_counts[status] += 1
        total_articles += len(articles)
        if issue:
            log.info("  %-16s %s (%d Điều) %s", dn, status, len(articles), issue)
        for a in articles:
            rows.append((
                dn, a.article_no, a.article_title,
                a.chapter_no, a.chapter_title, a.section_no, a.section_title,
                a.n_khoan, a.n_diem, len(a.article_text), a.article_text,
            ))

    db.replace_parsed_articles(conn, rows)
    db.update_parse_status(conn, statuses)
    conn.close()

    summary = {
        "documents": len(statuses),
        "parsed_articles": len(rows),
        "total_dieu": total_articles,
        **{f"status:{k}": v for k, v in sorted(status_counts.items())},
        "sqlite_path": sqlite_path,
    }
    log.info("Done: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG-vbpl M3 parser")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(run())


if __name__ == "__main__":
    main()
