"""Text normalization helpers (Step 2)."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_text(s: str | None) -> str:
    """NFC-normalize Unicode and collapse runs of whitespace. Required before hash-compare/dedup."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    return _WS.sub(" ", s).strip()


_INLINE_WS = re.compile(r"[ \t ​]+")  # spaces/tabs/nbsp/zero-width — NOT newlines


def normalize_lines(s: str | None) -> str:
    """NFC-normalize and collapse inline whitespace, but PRESERVE line breaks.

    Used for document bodies, where line structure is semantic (the Điều/Khoản parser relies on
    "Điều N" appearing at the start of a line). Collapses runs of blank lines to one.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    out: list[str] = []
    for ln in s.splitlines():
        ln = _INLINE_WS.sub(" ", ln).strip()
        if ln or (out and out[-1]):  # drop leading/consecutive blanks
            out.append(ln)
    return "\n".join(out).strip()


def content_hash(s: str | None) -> str:
    """SHA256 of the normalized text — used to detect content conflicts across sources."""
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()


def fold_ascii(s: str | None) -> str:
    """Lowercase + strip accents for tolerant matching.

    Note: Vietnamese 'đ'/'Đ' have no NFD decomposition, so map them to 'd' explicitly before
    stripping combining marks — otherwise 'lao động' → 'lao đong' and ascii keywords miss.
    """
    s = normalize_text(s).casefold().replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")
