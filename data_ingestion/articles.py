"""Step 5 — Điều/Khoản/Điểm parser.

Parses a document's canonical_content into Điều-level units, tracking the surrounding Chương/Mục and
counting Khoản/Điểm within each Điều. Handles both clean Markdown (UTS_VLC) and cleaned-HTML plain
text (main corpus). Validation is self-consistency only (sequential Điều numbering).

Vietnamese legal hierarchy: Văn bản → Chương → Mục → Điều → Khoản → Điểm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Header detection (applied to a markdown-stripped, lowercased-insensitive line).
_RE_CHUONG = re.compile(r"^chương\s+([ivxlcdm]+|\d+)\b\.?\s*(.*)$", re.IGNORECASE)
_RE_MUC = re.compile(r"^mục\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
_RE_DIEU = re.compile(r"^điều\s+(\d+)\b(.*)$", re.IGNORECASE)
# Within an Điều body:
_RE_KHOAN = re.compile(r"^\d+\.\s+\S")          # "1. ..."
_RE_DIEM = re.compile(r"^[a-zđ]\)\s+\S", re.IGNORECASE)  # "a) ..."

_MD = re.compile(r"[#*>`]+")  # markdown emphasis/heading/quote markers to strip for matching


def _is_dieu_header(rest: str) -> bool:
    """Distinguish a real 'Điều N' header from an inline cross-reference.

    Header: 'Điều 3. Giải thích…' (punctuation) or 'Điều 16 Tên điều' (capitalized title) or bare
    'Điều 2' (title on next line). Cross-reference: 'Điều 2 của Luật này', 'Điều 5, Điều 6' — the text
    after the number starts with a lowercase word / comma / digit.
    """
    rest = rest.strip()
    if not rest:
        return True
    return rest[0] in ".:-)" or rest[0].isupper()


@dataclass
class ParsedArticle:
    article_no: int
    article_title: str = ""
    chapter_no: str = ""
    chapter_title: str = ""
    section_no: str = ""
    section_title: str = ""
    _lines: list[str] = field(default_factory=list)

    @property
    def article_text(self) -> str:
        return "\n".join(self._lines).strip()

    @property
    def n_khoan(self) -> int:
        return sum(1 for ln in self._lines if _RE_KHOAN.match(ln))

    @property
    def n_diem(self) -> int:
        return sum(1 for ln in self._lines if _RE_DIEM.match(ln))


_RE_BARE_DIEU = re.compile(r"^điều\s*(\d*)$", re.IGNORECASE)  # digits may be 0 (all on next lines)


def _stitch_headers(text: str) -> str:
    """Repair Điều headers split across lines by bad source formatting.

    Some UTS_VLC records render 'Điều 20. Tiêu đề' as three lines: 'Điều 2' / '0' / '. Tiêu đề'.
    Merge a bare 'Điều N' with following pure-digit lines (the rest of the number) and a leading-'.'
    line. Pure-digit-only continuation lines are unambiguous (a Khoản is 'N. text', never bare 'N').
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _RE_BARE_DIEU.match(lines[i].strip())
        if not m:
            out.append(lines[i])
            i += 1
            continue
        num = m.group(1)
        j = i + 1
        while j < len(lines) and lines[j].strip().isdigit():
            num += lines[j].strip()
            j += 1
        if not num:  # "Điều" with no number anywhere → not a header, leave as-is
            out.append(lines[i])
            i += 1
            continue
        if j < len(lines) and lines[j].strip().startswith("."):
            out.append(f"Điều {num}{lines[j].strip()}")
            j += 1
        else:
            out.append(f"Điều {num}.")
        i = j
    return "\n".join(out)


_LQUOTE, _RQUOTE = "“", "”"  # “ ” — wrap amended text quoted from other laws


def _quote_depths(lines: list[str]) -> list[int]:
    """Quote-nesting depth at the START of each line (using “…” pairs).

    Amendment articles (e.g. Luật Đầu tư Điều 75–77) quote whole 'Điều N' of the laws they amend;
    those quoted headers must not be mistaken for this document's own articles. Only applied when
    quotes are balanced — otherwise depth-tracking would wrongly suppress real headers."""
    depths, d = [], 0
    for raw in lines:
        depths.append(d)
        d += raw.count(_LQUOTE) - raw.count(_RQUOTE)
    if d != 0:  # unbalanced → don't trust it; treat everything as top-level
        return [0] * len(lines)
    return depths


def parse_document(text: str | None) -> list[ParsedArticle]:
    """Split ``text`` into Điều units with their Chương/Mục context."""
    if not text:
        return []
    text = _stitch_headers(text)
    articles: list[ParsedArticle] = []
    cur: ParsedArticle | None = None
    expect_title = False
    chapter = ("", "")
    section = ("", "")

    lines = text.split("\n")
    depths = _quote_depths(lines)
    for idx, raw in enumerate(lines):
        s = raw.strip()
        h = _MD.sub(" ", s).strip()  # header-detection view
        if not h:
            if cur is not None:
                cur._lines.append("")
            continue

        # Inside a quoted amendment block: keep the text as body, but never as a structural header.
        if depths[idx] > 0:
            if cur is not None:
                cur._lines.append(s)
            continue

        m_dieu = _RE_DIEU.match(h)
        if m_dieu and _is_dieu_header(m_dieu.group(2)):
            if cur is not None:
                articles.append(cur)
            title = m_dieu.group(2).strip().lstrip(".:-) ").strip()
            cur = ParsedArticle(
                article_no=int(m_dieu.group(1)),
                article_title=title,
                chapter_no=chapter[0], chapter_title=chapter[1],
                section_no=section[0], section_title=section[1],
            )
            expect_title = not title  # title may be on the next line
            continue

        m_ch = _RE_CHUONG.match(h)
        if m_ch:
            chapter = (m_ch.group(1), m_ch.group(2).strip())
            continue
        m_muc = _RE_MUC.match(h)
        if m_muc:
            section = (m_muc.group(1), m_muc.group(2).strip())
            continue

        if cur is not None:
            if expect_title and not cur.article_title:
                cur.article_title = s
                expect_title = False
            else:
                cur._lines.append(s)

    if cur is not None:
        articles.append(cur)
    return articles


def validate(articles: list[ParsedArticle]) -> tuple[str, str]:
    """Self-consistency check. Returns (status, issue) where status is passed|failed."""
    if not articles:
        return "failed", "no_dieu_found"
    nums = [a.article_no for a in articles]
    if nums == list(range(1, len(nums) + 1)):
        return "passed", ""
    issues = []
    if len(set(nums)) != len(nums):
        issues.append("duplicate_dieu")
    if nums != sorted(nums):
        issues.append("out_of_order")
    if sorted(set(nums)) != list(range(min(nums), max(nums) + 1)):
        issues.append("gaps")
    if nums[0] != 1:
        issues.append("not_starting_at_1")
    return "failed", ",".join(issues) or "nonstandard_numbering"
