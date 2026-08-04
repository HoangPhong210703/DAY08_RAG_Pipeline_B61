"""Metadata field normalization + canonical-id construction (Step 2 / spec §4)."""

from __future__ import annotations

import hashlib
import re

from .text import fold_ascii as _fold
from .text import normalize_text

# --- dates -------------------------------------------------------------------

_DMY = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$")
_ISO = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")


def parse_vn_date(s: str | None) -> str | None:
    """Parse the main corpus' ``DD/MM/YYYY`` dates to ISO ``YYYY-MM-DD``. Returns None if unparseable."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    if _ISO.match(s):
        return s
    m = _DMY.match(s)
    if not m:
        return None
    d, mth, y = m.groups()
    return f"{y}-{int(mth):02d}-{int(d):02d}"


# --- field normalizers (for the dedup key) -----------------------------------


def normalize_doc_number(s: str | None) -> str | None:
    """e.g. ' 45/2019/qh14 ' -> '45/2019/QH14'."""
    if not s:
        return None
    s = normalize_text(s).upper().replace(" ", "")
    return s or None


def normalize_authority(s: str | None) -> str | None:
    return _fold(s) or None


def normalize_doc_type(s: str | None) -> str | None:
    return _fold(s) or None


def make_canonical_id(
    document_type: str | None,
    document_number: str | None,
    issuing_authority: str | None,
    issued_date: str | None,
) -> str | None:
    """SHA256 of the normalized composite key. Returns None if any field is missing → manual review."""
    dt = normalize_doc_type(document_type)
    dn = normalize_doc_number(document_number)
    ia = normalize_authority(issuing_authority)
    if not (dt and dn and ia and issued_date):
        return None
    key = "|".join([dt, dn.casefold(), ia, issued_date])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
