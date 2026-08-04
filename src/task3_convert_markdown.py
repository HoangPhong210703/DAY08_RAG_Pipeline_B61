"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import zlib
from pathlib import Path

try:
    from markitdown import MarkItDown
except ImportError:  # Optional dependency; PDF fallback below is self-contained.
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown() if MarkItDown is not None else None

    if not legal_dir.exists():
        return
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            text = ""
            if md is not None:
                try:
                    text = md.convert(str(filepath)).text_content or ""
                except Exception as exc:
                    print(f"  Warning: MarkItDown không convert được ({exc}); dùng fallback.")
            if not text.strip() and filepath.suffix.lower() == ".pdf":
                text = _extract_pdf_text(filepath)
            if not text.strip():
                text = f"Không thể trích xuất nội dung từ {filepath.name}."

            header = (
                f"# {filepath.stem}\n\n"
                f"**Source file:** `{filepath.name}`\n"
                f"**Converted:** Task 3\n\n---\n\n"
            )
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + text.strip() + "\n", encoding="utf-8")
            print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        return
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            title = data.get("title", "Unknown")
            header = (
                f"# {title}\n\n"
                f"**Source:** {data.get('url', 'N/A')}\n"
                f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            )
            content = data.get("content_markdown") or data.get("content") or ""
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + str(content).strip() + "\n", encoding="utf-8")
            print(f"  Saved: {output_path}")


def _extract_pdf_text(filepath: Path) -> str:
    """Extract text operators from simple Flate-compressed PDFs.

    This is intentionally a fallback for the lab image. When MarkItDown/PDF
    extras are installed, they remain the preferred converter.
    """
    raw = filepath.read_bytes()
    cmap = _extract_tounicode_map(raw)
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, flags=re.DOTALL)
    text_parts: list[str] = []
    for stream in streams:
        try:
            decoded = zlib.decompress(stream)
        except zlib.error:
            decoded = stream
        for literal in re.findall(rb"\((?:\\.|[^\\)])*\)", decoded):
            value = literal[1:-1]
            value = re.sub(rb"\\([\\()\\\\])", rb"\1", value)
            decoded_value = _decode_pdf_value(value, cmap)
            if decoded_value.strip():
                text_parts.append(decoded_value)
        for hex_value in re.findall(rb"<([0-9A-Fa-f\s]{4,})>", decoded):
            compact = re.sub(rb"\s+", b"", hex_value)
            try:
                decoded_value = _decode_pdf_value(bytes.fromhex(compact.decode()), cmap)
                if decoded_value.strip():
                    text_parts.append(decoded_value)
            except ValueError:
                continue
    return "\n".join(part.strip() for part in text_parts if part.strip())


def _extract_tounicode_map(raw: bytes) -> dict[int, str]:
    """Read the common PDF ``ToUnicode`` CMap emitted by fpdf2."""
    mapping: dict[int, str] = {}
    for block in re.findall(rb"beginbfchar(.*?)endbfchar", raw, flags=re.DOTALL):
        for code, target in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            try:
                mapping[int(code, 16)] = chr(int(target, 16))
            except (ValueError, OverflowError):
                continue
    return mapping


def _decode_pdf_value(value: bytes, cmap: dict[int, str]) -> str:
    if cmap and len(value) % 2 == 0:
        pairs = [int.from_bytes(value[index:index + 2], "big") for index in range(0, len(value), 2)]
        mapped = "".join(cmap.get(pair, "") for pair in pairs)
        if mapped and len(mapped) >= len(pairs) * 0.5:
            return mapped
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("latin-1", errors="replace")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output tai:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
