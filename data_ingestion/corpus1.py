"""Main corpus loader: th1nhng0/vietnamese-legal-documents.

Joins the ``metadata`` / ``content`` / ``relationships`` configs by ``id``, scope-filters, and maps
Vietnamese fields to :class:`CanonicalDocument`. Returns documents that still need content cleaning
and clean-laws attachment (done in :mod:`ragvbpl.ingest.run`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragvbpl.models import CanonicalDocument, Relationship
from ragvbpl.normalize.fields import make_canonical_id, normalize_doc_number, parse_vn_date

from . import scope
from .progress import track

_DATASET = "th1nhng0/vietnamese-legal-documents"

# Approx row counts per config (HF dataset card) — only used to render % progress bars.
_TOTALS = {"metadata": 153420, "content": 178665, "relationships": 897890}

# Order matters: "het hieu luc mot phan" (partially amended, still in force) must be checked before the
# plain "het hieu luc" (fully expired), since the former contains the latter as a substring.
_VALIDITY_MAP = {
    "con hieu luc": "active",
    "chua co hieu luc": "active",
    "het hieu luc mot phan": "active",
    "het hieu luc": "expired",
}


@dataclass
class MainCorpusResult:
    documents: list[CanonicalDocument]
    relationships: list[Relationship]
    raw_html: dict[str, str] = field(default_factory=dict)  # document_number -> content_html


def _load(name: str, config: str):
    from datasets import Features, Value, load_dataset

    # Streaming: reads parquet shards over HTTP row-by-row; nothing persisted between runs.
    kwargs = {}
    if config == "content":
        # content_html is stored as Arrow large_string; the declared schema says string, and casting
        # a batch back to 32-bit string offsets overflows (>2 GB). Keep it large_string to avoid the cast.
        kwargs["features"] = Features({"id": Value("int64"), "content_html": Value("large_string")})
    return load_dataset(name, config, split="data", streaming=True, **kwargs)


def _validity(row: dict) -> str:
    # Curated seeds are independently verified as in force — override the corpus' unreliable status
    # (e.g. it flags current laws 41/2024, 61/2020 as "Hết hiệu lực toàn bộ").
    if scope.is_seed(row):
        return "active"
    s = scope._fold(row.get("tinh_trang_hieu_luc"))
    for key, val in _VALIDITY_MAP.items():
        if key in s:
            return val
    return "unknown"


def _to_canonical(row: dict) -> CanonicalDocument:
    number = normalize_doc_number(row.get("so_ky_hieu"))
    issued = parse_vn_date(row.get("ngay_ban_hanh"))
    doc = CanonicalDocument(
        document_number=number,
        document_type=(row.get("loai_van_ban") or None),
        title=(row.get("title") or None),
        issuing_authority=(row.get("co_quan_ban_hanh") or None),
        issued_date=issued,
        effective_date=parse_vn_date(row.get("ngay_co_hieu_luc")),
        expiry_date=parse_vn_date(row.get("ngay_het_hieu_luc")),
        validity_status=_validity(row),
        legal_domains=scope.classify_domains(row),
        metadata_source=_DATASET,
        source_datasets=[_DATASET],
        official_source_url=(row.get("nguon_thu_thap") or None),
    )
    doc.canonical_document_id = make_canonical_id(
        doc.document_type, doc.document_number, doc.issuing_authority, doc.issued_date
    )
    return doc


def _as_int(v) -> int | None:
    """Relationship ids are mixed int/str in the source; coerce to int for matching."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load(dataset_name: str = _DATASET, *, expand: bool = True) -> MainCorpusResult:
    """Scope-filter the main corpus and return canonical documents + relationships + raw HTML."""
    # Pass 1 — candidate pool (loose) and selection. Curated seeds are ALWAYS included even if the
    # corpus flags them out-of-pool (its validity status is unreliable, e.g. labels a current law
    # "Hết hiệu lực toàn bộ"); we've verified seeds independently.
    pool: dict[int, dict] = {}
    selected: set[int] = set()
    for row in track(_load(dataset_name, "metadata"), "Corpus1 metadata", _TOTALS["metadata"]):
        seed = scope.is_seed(row)
        if not (scope.in_pool(row) or seed):
            continue
        pool[row["id"]] = row
        if seed or scope.matches_scope(row):
            selected.add(row["id"])

    # Pass 2 — relationship expansion (one hop, BIDIRECTIONAL): pull amenders/guidance docs in the
    # pool. Edges may be recorded from either side (a decree's "hướng dẫn" edge points to the law), so
    # we expand from a fixed base set in both directions.
    if expand:
        base = set(selected)
        add: set[int] = set()
        for r in track(_load(dataset_name, "relationships"), "Corpus1 rel (expand)", _TOTALS["relationships"]):
            if not scope.expands_scope(r.get("relationship")):
                continue
            a, b = _as_int(r["doc_id"]), _as_int(r["other_doc_id"])
            if a in base and b in pool:
                add.add(b)
            if b in base and a in pool:
                add.add(a)
        selected |= add

    docs = {i: _to_canonical(pool[i]) for i in selected}
    num_by_id = {i: d.document_number for i, d in docs.items() if d.document_number}

    # Content join (raw HTML, keyed by document_number for downstream cleaning).
    raw_html: dict[str, str] = {}
    for c in track(_load(dataset_name, "content"), "Corpus1 content", _TOTALS["content"]):
        if c["id"] in selected and num_by_id.get(c["id"]):
            raw_html[num_by_id[c["id"]]] = c.get("content_html") or ""

    # Relationships among selected docs (mapped to document numbers).
    rels: list[Relationship] = []
    for r in track(_load(dataset_name, "relationships"), "Corpus1 rel (edges)", _TOTALS["relationships"]):
        a, b = num_by_id.get(_as_int(r["doc_id"])), num_by_id.get(_as_int(r["other_doc_id"]))
        if a and b:
            rels.append(Relationship(doc_number=a, other_doc_number=b, relationship=r.get("relationship") or ""))

    return MainCorpusResult(documents=list(docs.values()), relationships=rels, raw_html=raw_html)
