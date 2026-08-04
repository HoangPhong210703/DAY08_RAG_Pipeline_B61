"""POC scope filtering for the main corpus (spec §5, Step 1).

Strategy = seed list ∪ keyword match ∪ relationship expansion, restricted to central, in-force
Luật/Bộ luật/Nghị định/Thông tư. Tune the constants below to grow/shrink the corpus.
"""

from __future__ import annotations

from ragvbpl.normalize.fields import normalize_doc_number
from ragvbpl.normalize.text import fold_ascii

# Curated foundational laws (by số ký hiệu), verified against UTS_VLC (vbpl.vn) + web. EDIT ME.
# Their amendments and implementing decrees/circulars are pulled automatically via relationship
# expansion, so list only base/current laws here.
SEED_DOC_NUMBERS: set[str] = {
    # --- Lao động ---
    "45/2019/QH14",   # Bộ luật Lao động
    "84/2015/QH13",   # Luật An toàn, vệ sinh lao động
    "50/2024/QH15",   # Luật Công đoàn
    "69/2020/QH14",   # Luật NLĐ VN đi làm việc ở nước ngoài theo hợp đồng
    "124/2025/QH15",  # Luật Giáo dục nghề nghiệp
    # --- BHXH & việc làm ---
    "41/2024/QH15",   # Luật Bảo hiểm xã hội
    "74/2025/QH15",   # Luật Việc làm
    "25/2008/QH12",   # Luật Bảo hiểm y tế
    "46/2014/QH13",   # Luật sửa đổi, bổ sung Luật BHYT
    "51/2024/QH15",   # Luật BHYT sửa đổi
    # --- Doanh nghiệp & hộ kinh doanh ---
    "59/2020/QH14",   # Luật Doanh nghiệp
    "61/2020/QH14",   # Luật Đầu tư
    "64/2020/QH14",   # Luật Đầu tư theo phương thức PPP
    "04/2017/QH14",   # Luật Hỗ trợ DN nhỏ và vừa
    "17/2023/QH15",   # Luật Hợp tác xã
    "142/2025/QH15",  # Luật Phục hồi, phá sản
    "23/2018/QH14",   # Luật Cạnh tranh
    "54/2019/QH14",   # Luật Chứng khoán
    "69/2014/QH13",   # Luật QL, sử dụng vốn NN đầu tư vào SXKD tại DN
    # --- Thuế & kế toán ---
    "38/2019/QH14",   # Luật Quản lý thuế
    "48/2024/QH15",   # Luật Thuế giá trị gia tăng
    "88/2015/QH13",   # Luật Kế toán
    "67/2011/QH12",   # Luật Kiểm toán độc lập
    # --- Thương mại & thị trường ---
    "36/2005/QH11",   # Luật Thương mại
    "22/2023/QH15",   # Luật Đấu thầu
    "19/2023/QH15",   # Luật Bảo vệ quyền lợi người tiêu dùng
    "20/2023/QH15",   # Luật Giao dịch điện tử
    "50/2005/QH11",   # Luật Sở hữu trí tuệ
    "16/2023/QH15",   # Luật Giá
    # --- Tài chính / bất động sản ---
    "32/2024/QH15",   # Luật Các tổ chức tín dụng
    "29/2023/QH15",   # Luật Kinh doanh bất động sản
}

# Map POC domain code -> keywords matched (accent-insensitive) against lĩnh vực / ngành / title.
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "labor": ("lao dong", "tien luong", "an toan", "cong doan"),
    "social_insurance_employment": (
        "bao hiem xa hoi",
        "bao hiem that nghiep",
        "bao hiem y te",
        "viec lam",
    ),
    "enterprise": ("doanh nghiep", "dau tu", "ho kinh doanh", "kinh doanh"),
}

# Document types kept in scope (folded loại văn bản).
DOC_TYPES_IN_SCOPE: set[str] = {"luat", "bo luat", "nghi dinh", "thong tu"}

# Relationship labels (folded substrings) that pull a related doc into scope. Note the corpus uses the
# abbreviated guidance label "Văn bản HD, QĐ chi tiết" → matched via "chi tiet".
EXPAND_RELATIONSHIPS: tuple[str, ...] = ("sua doi", "bo sung", "huong dan", "chi tiet", "thay the")

# Validity strings (folded) that count as "in force". "Hết hiệu lực một phần" means the law is still
# in force except for amended parts → keep it. (Order matters: check it before plain "het hieu luc".)
_IN_FORCE = ("con hieu luc", "chua co hieu luc", "het hieu luc mot phan")
# Central-scope strings (folded). Empty/unknown is allowed through (kept loose for the POC).
_CENTRAL = ("toan quoc",)


_fold = fold_ascii  # accent/đ-insensitive folding, shared with the rest of the pipeline


def is_in_force(row: dict) -> bool:
    status = _fold(row.get("tinh_trang_hieu_luc"))
    if not status:
        return False
    return any(k in status for k in _IN_FORCE)


def is_central(row: dict) -> bool:
    scope = _fold(row.get("pham_vi"))
    if not scope:
        return True  # unknown → don't exclude in the POC
    return any(k in scope for k in _CENTRAL)


def doc_type_in_scope(row: dict) -> bool:
    return _fold(row.get("loai_van_ban")) in DOC_TYPES_IN_SCOPE


def classify_domains(row: dict) -> list[str]:
    """Return the POC domain codes whose keywords appear in lĩnh vực/ngành/title."""
    hay = _fold(" ".join(str(row.get(f, "") or "") for f in ("linh_vuc", "nganh", "title")))
    return [dom for dom, kws in DOMAIN_KEYWORDS.items() if any(kw in hay for kw in kws)]


def is_seed(row: dict) -> bool:
    return normalize_doc_number(row.get("so_ky_hieu")) in SEED_DOC_NUMBERS


def in_pool(row: dict) -> bool:
    """Loose candidate filter: central, in force, and an in-scope document type."""
    return doc_type_in_scope(row) and is_in_force(row) and is_central(row)


def matches_scope(row: dict) -> bool:
    """Tight filter: curated seed laws only.

    Keyword matching is intentionally NOT used for selection (it pulled off-domain laws like atomic
    energy via "an toàn"). Amendments + implementing decrees/circulars enter via relationship
    expansion. classify_domains is still used to *tag* legal_domains, just not to select.
    """
    return is_seed(row)


def expands_scope(relationship: str | None) -> bool:
    rel = _fold(relationship)
    return any(k in rel for k in EXPAND_RELATIONSHIPS)
