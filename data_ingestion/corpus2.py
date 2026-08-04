"""Clean-laws loader: undertheseanlp/UTS_VLC.

Provides clean Markdown full-text for foundational laws/codes. Has no issuing_authority/issued_date,
so it's keyed by normalized document number and attached to main-corpus metadata (spec: content from
clean-laws, metadata from main corpus).
"""

from __future__ import annotations

from ragvbpl.normalize.fields import normalize_doc_number

from .progress import track

DATASET = "undertheseanlp/UTS_VLC"
SPLIT = "2026"  # verified in-force snapshot (Constitution + 305 Laws/Codes)
_TOTAL = 306    # rows in the 2026 split (for the progress bar)


def load_index(dataset_name: str = DATASET, split: str = SPLIT) -> dict[str, dict]:
    """Return ``{normalized_document_number: {title, type, content}}`` for fast lookup."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    index: dict[str, dict] = {}
    for row in track(ds, "Corpus2 laws", _TOTAL):
        num = normalize_doc_number(row.get("id"))
        if not num:
            continue
        index[num] = {
            "title": row.get("title"),
            "type": row.get("type"),       # code | law | constitution
            "content": row.get("content"),  # cleaned Markdown
        }
    return index


def lookup(index: dict[str, dict], document_number: str | None) -> dict | None:
    return index.get(normalize_doc_number(document_number)) if document_number else None
