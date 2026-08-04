"""Article-level loader: tmquan/phapdien-moj-gov-vn (``articles`` subset).

Kept as a separate validation reference (NOT merged into content). Filtered to the POC domains by the
Vietnamese topic/subject titles. Used later (M3) to validate the Điều parser.
"""

from __future__ import annotations

from ragvbpl.models import ArticleRef
from ragvbpl.normalize.text import fold_ascii as _fold

from .progress import track
from .scope import DOMAIN_KEYWORDS

DATASET = "tmquan/phapdien-moj-gov-vn"
SUBSET = "articles"
SPLIT = "train"
_TOTAL = 66000  # approx rows in the articles subset (for the progress bar)

_ALL_KEYWORDS = tuple(kw for kws in DOMAIN_KEYWORDS.values() for kw in kws)


def _in_domain(row: dict) -> bool:
    hay = _fold(f"{row.get('topic_title_vi', '')} {row.get('subject_title_vi', '')}")
    return any(kw in hay for kw in _ALL_KEYWORDS)


def load(dataset_name: str = DATASET, *, domain_filter: bool = True) -> list[ArticleRef]:
    """Stream the articles subset, optionally keeping only POC-domain topics."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, SUBSET, split=SPLIT, streaming=True)
    out: list[ArticleRef] = []
    for row in track(ds, "Corpus3 articles", _TOTAL):
        if domain_filter and not _in_domain(row):
            continue
        out.append(
            ArticleRef(
                record_id=row.get("record_id", ""),
                article_id=row.get("article_id", ""),
                article_title=row.get("article_title"),
                topic_title_vi=row.get("topic_title_vi"),
                subject_title_vi=row.get("subject_title_vi"),
                content_text=row.get("content_text"),
                source_url=row.get("source_url"),
            )
        )
    return out
