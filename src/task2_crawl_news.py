"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Nguồn công khai chính thức của VinUniversity, bao phủ các chủ đề dịch vụ sinh viên
# được yêu cầu trong README: thư viện, hỗ trợ sinh viên, wellbeing, học bổng và
# thông báo học vụ.
ARTICLE_URLS = [
    "https://experience.vinuni.edu.vn/student-life-support/health-well-being/",
    "https://experience.vinuni.edu.vn/",
    "https://library.vinuni.edu.vn/",
    "https://admissions.vinuni.edu.vn/scholarship-and-financial-aid/undergraduate-programs/scholarships/",
    "https://registrar.vinuni.edu.vn/announcements-decisions/",
]


class _ReadableHTMLParser(HTMLParser):
    """Trích xuất title và phần text có thể đọc từ HTML khi Crawl4AI unavailable."""

    BLOCK_TAGS = {
        "article", "blockquote", "br", "div", "footer", "h1", "h2", "h3",
        "h4", "header", "li", "main", "p", "section", "td", "th", "tr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._title_parts = []
        self._parts = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS and self._ignored_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            self.title = " ".join(self._title_parts).strip()
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag in self.BLOCK_TAGS and self._ignored_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._ignored_depth:
            return
        text = unescape(data).strip()
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        else:
            self._parts.append(text + " ")

    def markdown(self) -> str:
        lines = []
        for raw_line in "".join(self._parts).splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if line and (not lines or line != lines[-1]):
                lines.append(line)
        return "\n\n".join(lines)


def _crawl_with_requests(url: str) -> dict:
    """HTTP fallback để task vẫn chạy khi Playwright/Chromium chưa được cài."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; UniversityServicesRAG/1.0; "
                "+educational-project)"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="replace")
    parser = _ReadableHTMLParser()
    parser.feed(html)
    content = parser.markdown()
    if len(content) < 500:
        raise ValueError(f"Nội dung crawl quá ngắn ({len(content)} ký tự): {url}")
    return {
        "url": url,
        "title": parser.title or urlparse(url).path.strip("/").replace("-", " ").title(),
        "date_crawled": datetime.now().astimezone().isoformat(),
        "content": content,
        "content_markdown": content,
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            content = str(getattr(result, "markdown", "") or "").strip()
            if getattr(result, "success", True) and len(content) >= 500:
                metadata = getattr(result, "metadata", {}) or {}
                return {
                    "url": url,
                    "title": metadata.get("title") or urlparse(url).netloc,
                    "date_crawled": datetime.now().astimezone().isoformat(),
                    "content": content,
                    "content_markdown": content,
                }
    except Exception as exc:
        print(f"  Crawl4AI unavailable/failed ({exc}); using HTTP fallback")

    return await asyncio.to_thread(_crawl_with_requests, url)


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  [ERROR] Failed: {exc}")
            continue

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARN] Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
