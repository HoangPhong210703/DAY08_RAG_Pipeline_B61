"""Step 6 — structure-aware chunking.

One Điều = one chunk; a long Điều is split by Khoản; a long Khoản by Điểm; an over-long Điểm is
hard-split. Each chunk carries its parent document/article metadata + a citation header (prepended to
the text to give embeddings context). See ``data-src.md`` §5 step 6.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_RE_KHOAN = re.compile(r"^(\d+)\.\s")
_RE_DIEM = re.compile(r"^([a-zđ])\)\s", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    document_number: str
    document_type: str | None
    title: str | None
    issuing_authority: str | None
    validity_status: str
    legal_domains: list[str]
    chapter_no: str
    chapter_title: str
    article_no: int
    article_title: str
    clause_no: str | None     # Khoản
    point: str | None         # Điểm
    source_url: str | None
    canonical_source: str | None
    text: str                 # citation header + body (what gets embedded)
    char_len: int = field(default=0)


def _segment(text: str, pat: re.Pattern) -> list[tuple[str | None, str]]:
    """Split text into (label, segment) by lines matching pat. Text before the first marker → label None."""
    label: str | None = None
    buf: list[str] = []
    segs: list[tuple[str | None, str]] = []
    for ln in text.split("\n"):
        m = pat.match(ln.strip())
        if m:
            if buf:
                segs.append((label, "\n".join(buf).strip()))
            label, buf = m.group(1), [ln]
        else:
            buf.append(ln)
    if buf:
        segs.append((label, "\n".join(buf).strip()))
    return [(lbl, t) for lbl, t in segs if t]


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [text]


def chunk_article(meta: dict, art: dict, max_chars: int = 2000, split_clauses: bool = True) -> list[Chunk]:
    """Return the chunks for one parsed Điều (``art``) of document ``meta``.

    split_clauses=True  → one Điều = one chunk; long Điều split by Khoản, then Điểm (clause-aware).
    split_clauses=False → Điều-only: one chunk per Điều, long ones size-capped only (no Khoản/Điểm).
    """
    header = f"{meta.get('title') or ''} - Điều {art['article_no']}. {art.get('article_title') or ''}".strip(" -")
    body = art.get("article_text") or ""

    units: list[tuple[str | None, str | None, str]] = []  # (clause_no, point, text)
    if not split_clauses or len(body) <= max_chars:
        units.append((None, None, body))
    else:
        for kno, kseg in _segment(body, _RE_KHOAN):
            if kno is None or len(kseg) <= max_chars:
                units.append((kno, None, kseg))
            else:
                for pno, pseg in _segment(kseg, _RE_DIEM):
                    units.append((kno, pno, pseg))

    # Enforce the size cap on EVERY unit — incl. unlabeled intro / no-Khoản blocks (tables, appendices)
    # that segmentation can't split. Without this a single Điều can become a 250k-char chunk.
    capped: list[tuple[str | None, str | None, str]] = []
    for kno, pno, seg in units:
        if len(seg) <= max_chars:
            capped.append((kno, pno, seg))
        else:
            capped.extend((kno, pno, hs) for hs in _hard_split(seg, max_chars))
    units = capped

    chunks: list[Chunk] = []
    for idx, (kno, pno, seg) in enumerate(units):
        text = f"{header}\n{seg}".strip()
        raw_id = f"{meta['document_number']}|{art['article_no']}|{kno}|{pno}|{idx}"
        chunks.append(Chunk(
            chunk_id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24],
            document_number=meta["document_number"],
            document_type=meta.get("document_type"),
            title=meta.get("title"),
            issuing_authority=meta.get("issuing_authority"),
            validity_status=meta.get("validity_status") or "unknown",
            legal_domains=meta.get("legal_domains") or [],
            chapter_no=art.get("chapter_no") or "",
            chapter_title=art.get("chapter_title") or "",
            article_no=art["article_no"],
            article_title=art.get("article_title") or "",
            clause_no=kno,
            point=pno,
            source_url=meta.get("official_source_url"),
            canonical_source=meta.get("canonical_content_source"),
            text=text,
            char_len=len(text),
        ))
    return chunks
