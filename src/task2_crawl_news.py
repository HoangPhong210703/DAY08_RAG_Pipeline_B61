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
from html.parser import HTMLParser
import json
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://www.rmit.edu.vn/study-at-rmit/tuition-fees",
    "https://www.rmit.edu.vn/study-at-rmit/scholarships",
    "https://www.rmit.edu.vn/libraryvn/student-support/book-a-study-room",
    "https://www.rmit.edu.vn/students/support/student-academic-success",
    "https://www.rmit.edu.vn/student-life/support-services",
]


class _VisibleTextParser(HTMLParser):
    """Nhặt title và text cơ bản khi Crawl4AI không có trong môi trường."""

    _ignored = {"script", "style", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_title = False
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag in self._ignored:
            self._ignored_depth += 1

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in self._ignored and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data):
        value = " ".join(data.split())
        if not value or self._ignored_depth:
            return
        if self._in_title:
            self.title += value
        elif len(value) > 1:
            self.parts.append(value)


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
            metadata = getattr(result, "metadata", {}) or {}
            markdown = getattr(result, "markdown", "") or ""
            return {
                "url": url,
                "title": metadata.get("title", "Unknown"),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": markdown.strip(),
            }
    except ImportError:
        # Requests/HTMLParser fallback keeps the task runnable without a
        # browser binary; production crawling can still opt into Crawl4AI.
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "RAG-lab-crawler/1.0"},
        )
        response.raise_for_status()
        parser = _VisibleTextParser()
        parser.feed(response.text)
        content = "\n\n".join(parser.parts)
        return {
            "url": url,
            "title": parser.title.strip() or "Unknown",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
