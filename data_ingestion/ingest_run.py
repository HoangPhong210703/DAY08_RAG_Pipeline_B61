"""M1 ingestion orchestrator.

Flow (spec §5, steps 1–2):
  main corpus → scope-filter → attach clean-laws content → clean HTML for the rest →
  hash content → load article-level reference → write everything to SQLite.

Run:  python -m ragvbpl.ingest.run
"""

from __future__ import annotations

import argparse
import logging
import re

from ragvbpl.config import load_settings
from ragvbpl.models import CanonicalDocument
from ragvbpl.normalize.fields import make_canonical_id
from ragvbpl.normalize.text import content_hash
from ragvbpl.store import db

from . import article_level, corpus1, corpus2, scope
from .clean_html import html_to_text

log = logging.getLogger("ragvbpl.ingest")

# Document types (folded) treated as foundational laws → prefer clean-laws content.
_LAW_TYPES = {"luat", "bo luat"}

# Seeds whose content is unusable in corpus 1 & 2 → reconstruct from the Pháp điển đề mục's foundational
# (LQ) article series. Maps document_number → "<chủ đề>.<đề mục>" prefix. (45/2019 = đề mục 20.2, the
# Labor Code: 220 LQ articles; both other sources are corrupted/empty.)
PHAPDIEN_RECONSTRUCT: dict[str, str] = {"45/2019/QH14": "20.2"}

# Hand-verified metadata for curated seeds that are absent from the main corpus (UTS_VLC has their
# content/title but no metadata). Dates checked against chinhphu.vn / thuvienphapluat.vn.
MISSING_SEED_META: dict[str, dict] = {
    "124/2025/QH15": {
        "document_type": "Luật",
        "title": "Luật Giáo dục nghề nghiệp",
        "issuing_authority": "Quốc hội",
        "issued_date": "2025-12-10",
        "effective_date": "2026-01-01",
        "legal_domains": ["labor"],
    },
    "51/2024/QH15": {
        "document_type": "Luật",
        "title": "Luật sửa đổi, bổ sung một số điều của Luật Bảo hiểm y tế",
        "issuing_authority": "Quốc hội",
        "issued_date": "2024-11-27",
        "effective_date": "2025-07-01",
        "legal_domains": ["social_insurance_employment"],
    },
}


def _is_law(doc: CanonicalDocument) -> bool:
    from .scope import _fold

    return _fold(doc.document_type) in _LAW_TYPES


_DIEU_LINE = re.compile(r"(?mi)^[#*>\s]*điều\s+\d+")


def _has_dieu(text: str) -> bool:
    """A real law/decree body has at least one 'Điều N' at a line start. Used to reject corrupted
    clean-laws records (UTS_VLC has e.g. 45/2019/QH14 = a Lào Cai plan, not the Labor Code)."""
    return bool(_DIEU_LINE.search(text))


def _pick_content(
    doc: CanonicalDocument, clean_index: dict, raw_html: dict[str, str]
) -> tuple[str | None, str | None]:
    """Return (canonical_content, source_dataset) per the source-priority rules.

    For laws, prefer UTS_VLC but only if its content looks valid (contains Điều); otherwise fall back
    to the main corpus. Returns (None, None) when no usable content exists (e.g. empty placeholder
    HTML), so we don't store empty/misleading content.
    """
    # Cleaned main-corpus HTML (fallback for laws; primary for decrees/circulars).
    html_text = None
    html = raw_html.get(doc.document_number or "")
    if html:
        t = html_to_text(html)
        if t.strip():
            html_text = t

    if _is_law(doc):
        hit = corpus2.lookup(clean_index, doc.document_number)
        uts = (hit.get("content") if hit else None) or ""
        if uts.strip() and _has_dieu(uts):
            return uts, corpus2.DATASET
        # UTS missing or corrupted → fall back to the main corpus.
        if html_text:
            return html_text, corpus1._DATASET
        return None, None

    if html_text:
        return html_text, corpus1._DATASET
    return None, None


def _reconstruct_from_phapdien(articles, demuc: str) -> str | None:
    """Assemble a document's text from the Pháp điển đề mục's LQ (foundational-law) article series.

    Each row 'Điều {demuc}.LQ.{N}' → 'Điều {N}. {title}\\n{body}', ordered by N. Returns None if the
    series isn't present.
    """
    pat = re.compile(rf"^Điều {re.escape(demuc)}\.LQ\.(\d+)$")
    items: list[tuple[int, str]] = []
    for a in articles:
        m = pat.match(a.article_id or "")
        if not m:
            continue
        n = int(m.group(1))
        prefix = f"{a.article_id}. "
        title = a.article_title[len(prefix):] if (a.article_title or "").startswith(prefix) else (a.article_title or "")
        items.append((n, f"Điều {n}. {title}\n{a.content_text or ''}".strip()))
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    return "\n\n".join(t for _, t in items)


def _contribute_missing_seeds(
    documents: list[CanonicalDocument], clean_index: dict, structure_src: str
) -> list[CanonicalDocument]:
    """Build documents for curated seeds absent from the main corpus, sourced from UTS_VLC content +
    the hand-verified MISSING_SEED_META. Returns the docs added."""
    present = {d.document_number for d in documents}
    added: list[CanonicalDocument] = []
    for number in scope.SEED_DOC_NUMBERS - present:
        hit = corpus2.lookup(clean_index, number)
        if not (hit and (hit.get("content") or "").strip()):
            continue  # not in UTS_VLC either → genuinely unavailable
        meta = MISSING_SEED_META.get(number, {})
        doc = CanonicalDocument(
            document_number=number,
            document_type=meta.get("document_type") or "Luật",
            title=meta.get("title") or hit.get("title"),
            issuing_authority=meta.get("issuing_authority"),
            issued_date=meta.get("issued_date"),
            effective_date=meta.get("effective_date"),
            validity_status="active",
            legal_domains=meta.get("legal_domains", []),
            canonical_content=hit["content"],
            canonical_content_source=corpus2.DATASET,
            metadata_source=corpus2.DATASET,
            structure_validation_source=structure_src,
            source_datasets=[corpus2.DATASET],
            content_hash=content_hash(hit["content"]),
        )
        doc.canonical_document_id = make_canonical_id(
            doc.document_type, doc.document_number, doc.issuing_authority, doc.issued_date
        )
        added.append(doc)
    return added


def run(expand: bool = True, article_domain_filter: bool = True) -> dict:
    s = load_settings()
    sources = s.config.sources
    sqlite_path = s.config.paths.get("sqlite_path", "data/ragvbpl.sqlite")

    log.info("Loading main corpus (scope filtering)…")
    main = corpus1.load(sources.get("main", corpus1._DATASET), expand=expand)
    log.info("  selected %d documents, %d relationships", len(main.documents), len(main.relationships))

    log.info("Loading clean-laws index…")
    clean_index = corpus2.load_index(sources.get("clean_laws", corpus2.DATASET))

    log.info("Selecting canonical content…")
    for doc in main.documents:
        content, src = _pick_content(doc, clean_index, main.raw_html)
        doc.canonical_content = content
        doc.canonical_content_source = src
        doc.content_hash = content_hash(content) if content else None
        doc.structure_validation_source = sources.get("article_level", article_level.DATASET)
        if src and src not in doc.source_datasets:
            doc.source_datasets.append(src)

    structure_src = sources.get("article_level", article_level.DATASET)
    contributed = _contribute_missing_seeds(main.documents, clean_index, structure_src)
    if contributed:
        main.documents.extend(contributed)
        log.info("  contributed %d missing-seed docs from clean-laws: %s",
                 len(contributed), [d.document_number for d in contributed])

    log.info("Loading article-level reference…")
    articles = article_level.load(
        sources.get("article_level", article_level.DATASET), domain_filter=article_domain_filter
    )
    log.info("  %d article reference rows", len(articles))

    # Reconstruct content for seeds unusable in corpus 1 & 2, from the Pháp điển LQ series.
    by_number = {d.document_number: d for d in main.documents}
    for number, demuc in PHAPDIEN_RECONSTRUCT.items():
        doc = by_number.get(number)
        if doc is None or doc.canonical_content:
            continue
        text = _reconstruct_from_phapdien(articles, demuc)
        if text:
            doc.canonical_content = text
            doc.canonical_content_source = article_level.DATASET
            doc.content_hash = content_hash(text)
            if article_level.DATASET not in doc.source_datasets:
                doc.source_datasets.append(article_level.DATASET)
            log.info("  reconstructed %s from Pháp điển đề mục %s (%d chars)", number, demuc, len(text))

    log.info("Writing to SQLite at %s …", sqlite_path)
    conn = db.connect(sqlite_path)
    db.upsert_documents(conn, main.documents)
    db.replace_relationships(conn, main.relationships)
    db.replace_articles_ref(conn, articles)
    conn.close()

    summary = {
        "documents": len(main.documents),
        "relationships": len(main.relationships),
        "articles_ref": len(articles),
        "with_clean_law_content": sum(
            1 for d in main.documents if d.canonical_content_source == corpus2.DATASET
        ),
        "needs_review": sum(1 for d in main.documents if d.canonical_document_id is None),
        "sqlite_path": sqlite_path,
    }
    log.info("Done: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG-vbpl M1 ingestion")
    ap.add_argument("--no-expand", action="store_true", help="skip relationship expansion")
    ap.add_argument("--all-articles", action="store_true", help="don't domain-filter Pháp điển")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    summary = run(expand=not args.no_expand, article_domain_filter=not args.all_articles)
    print(summary)


if __name__ == "__main__":
    main()
