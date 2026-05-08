"""Plain text from HTML-heavy listing fields."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


def strip_html_to_text(html: str | float | None, *, max_len: int | None = 200_000) -> str:
    if html is None or (isinstance(html, float) and html != html):
        return ""
    s = str(html)
    if max_len is not None and len(s) > max_len:
        s = s[:max_len]
    text = BeautifulSoup(s, "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
