"""HTML → text cleaning for the main corpus' ``content_html`` (Step 1)."""

from __future__ import annotations

from ragvbpl.normalize.text import normalize_lines


def html_to_text(html: str | None) -> str:
    """Extract readable text from raw legal-document HTML, preserving line structure.

    Prefers trafilatura (good at dropping boilerplate); falls back to BeautifulSoup. Line breaks are
    kept (normalize_lines) so the Điều/Khoản parser can detect "Điều N" at line starts. Heavy deps
    are imported lazily so the package stays importable without them.
    """
    if not html:
        return ""

    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, include_tables=True, include_comments=False, favor_recall=True
        )
        if extracted:
            return normalize_lines(extracted)
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return normalize_lines(soup.get_text(separator="\n"))
    except Exception:
        return normalize_lines(html)
