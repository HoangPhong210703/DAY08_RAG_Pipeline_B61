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
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            'Thiếu MarkItDown. Cài bằng: pip install "markitdown[pdf]"'
        ) from exc

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            try:
                result = md.convert(str(filepath))
                content = (result.text_content or "").strip()
                if len(content) < 200:
                    raise ValueError(f"Nội dung quá ngắn ({len(content)} ký tự)")
                output_path.write_text(content + "\n", encoding="utf-8")
                print(f"  [OK] Saved: {output_path}")
            except Exception as exc:
                print(f"  [ERROR] Failed {filepath.name}: {exc}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                # `content` là schema chuẩn của dữ liệu landing; hỗ trợ
                # `content_markdown` để tương thích với starter README cũ.
                article_content = str(
                    data.get("content") or data.get("content_markdown", "")
                ).strip()
                if not data.get("url"):
                    raise ValueError("Thiếu metadata 'url'")
                if len(article_content) < 200:
                    raise ValueError(
                        f"Nội dung quá ngắn ({len(article_content)} ký tự)"
                    )

                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data['url']}\n\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
                output_path.write_text(
                    header + article_content + "\n", encoding="utf-8"
                )
                print(f"  [OK] Saved: {output_path}")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(f"  [ERROR] Failed {filepath.name}: {exc}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[OK] Done! Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
