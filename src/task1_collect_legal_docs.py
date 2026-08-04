"""
Task 1 — Thu thập văn bản chính sách/quy định.

Nguồn dữ liệu: dataset công khai trên HuggingFace `tmquan/vbpl-vn`
(https://huggingface.co/datasets/tmquan/vbpl-vn) — bản số hoá Cơ sở dữ liệu Quốc gia
về pháp luật (vbpl.vn, Bộ Tư pháp). Đây CHÍNH LÀ nguồn gốc của
`data/standardized/legal/ragvbpl.sqlite` đã có sẵn trong repo (xem `data_ingestion/`).

Script này KHÔNG dùng để nạp dữ liệu cho vector store — Task 4 đọc thẳng từ
`ragvbpl.sqlite` (đã chunk sẵn, chi tiết hơn nhiều so với 5 file này). Mục đích của
script này chỉ là thoả yêu cầu Task 1: có ≥3 file PDF/DOCX gốc trong
`data/landing/legal/` để chứng minh bước "thu thập văn bản".

Cách lấy dữ liệu: gọi HuggingFace Datasets Server REST API (không cần cài `datasets`/
`huggingface_hub`, không cần tải parquet) để lấy vài row theo `item_id`, sau đó dựng
PDF thật (không phải file .txt đổi tên) bằng `fpdf2` — đúng gợi ý fallback trong comment
gốc của file này ("nếu trang là HTML thuần, convert nội dung text thành PDF bằng fpdf2").
"""

import re
import unicodedata

import requests
from fpdf import FPDF
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "tmquan/vbpl-vn"
HF_CONFIG = "documents"
HF_SPLIT = "train"

# Chọn thủ công 5 văn bản đa dạng loại/năm/địa phương từ dataset (đã khảo sát trước
# qua datasets-server API), độ dài vừa phải (~2000-4200 ký tự) để PDF gọn 1-2 trang.
ITEM_IDS = ["100013", "100024", "100031", "100067", "100074"]

# Font hệ thống hỗ trợ tiếng Việt (Windows) — core fonts của fpdf2 (Helvetica/Times)
# không có dấu tiếng Việt.
_WINDOWS_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\seguisym.ttf"),
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def fetch_documents(item_ids: list[str]) -> dict[str, dict]:
    """
    Quét bảng `documents` qua Datasets Server REST API (phân trang bằng offset/length)
    cho tới khi tìm đủ các item_id cần lấy. Tránh dùng /filter — endpoint đó từ chối
    cú pháp where-clause đơn giản (đã thử, luôn trả "invalid symbols").
    """
    wanted = set(item_ids)
    found: dict[str, dict] = {}
    offset = 0
    page_size = 100
    max_offset = 5000  # đủ để phủ toàn bộ ITEM_IDS đã chọn (đều nằm trong ~700 row đầu)

    while wanted - found.keys() and offset < max_offset:
        resp = requests.get(
            DATASETS_SERVER_URL,
            params={
                "dataset": HF_DATASET,
                "config": HF_CONFIG,
                "split": HF_SPLIT,
                "offset": offset,
                "length": page_size,
            },
            timeout=30,
        )
        resp.raise_for_status()
        for item in resp.json()["rows"]:
            row = item["row"]
            if row.get("item_id") in wanted:
                found[row["item_id"]] = row
        offset += page_size

    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Không tìm thấy các item_id sau trong {HF_DATASET}: {missing}")
    return found


def _slugify(text: str, max_len: int = 60) -> str:
    """Chuyển tiêu đề tiếng Việt có dấu -> ascii-slug an toàn cho tên file."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-")


def _pick_unicode_font() -> Path:
    for candidate in _WINDOWS_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy font Unicode (arial.ttf) để render tiếng Việt vào PDF."
    )


def render_pdf(row: dict, output_path: Path):
    """Dựng 1 file PDF thật từ nội dung markdown của document (fpdf2 + font Unicode)."""
    pdf = FPDF()
    pdf.add_page()
    font_path = _pick_unicode_font()
    pdf.add_font("Vietnamese", "", str(font_path))
    pdf.set_font("Vietnamese", size=14)

    title = row.get("title") or "Untitled"
    doc_number = ", ".join(row.get("doc_number") or []) or "N/A"
    authority = row.get("issuing_authority") or "N/A"
    issue_date = row.get("issue_date") or "N/A"
    source_url = row.get("source_url") or "N/A"

    pdf.multi_cell(0, 8, title)
    pdf.ln(2)

    pdf.set_font("Vietnamese", size=10)
    meta_lines = (
        f"So hieu: {doc_number}\n"
        f"Co quan ban hanh: {authority}\n"
        f"Ngay ban hanh: {issue_date}\n"
        f"Nguon: {source_url}\n"
        f"Dataset: {HF_DATASET} (HuggingFace)"
    )
    pdf.multi_cell(0, 6, meta_lines)
    pdf.ln(4)

    pdf.set_font("Vietnamese", size=11)
    body = row.get("markdown") or ""
    pdf.multi_cell(0, 6, body)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"  ✓ Đã tạo: {output_path} ({output_path.stat().st_size} bytes)")


def collect_all():
    setup_directory()

    print(f"Fetching {len(ITEM_IDS)} documents from {HF_DATASET} ...")
    rows_by_id = fetch_documents(ITEM_IDS)

    for item_id in ITEM_IDS:
        row = rows_by_id[item_id]
        doc_type = row.get("doc_type") or "van-ban"
        number_slug = _slugify((row.get("doc_number") or ["unknown"])[0])
        title_slug = _slugify(row.get("title") or "untitled")
        filename = f"{doc_type}-{number_slug}-{title_slug}.pdf"

        render_pdf(row, DATA_DIR / filename)


if __name__ == "__main__":
    collect_all()
