"""
Fetch a URL and extract the main article/page text, stripping nav,
ads, and boilerplate. Uses trafilatura, which is purpose-built for
this and handles most real-world sites well; falls back to a plain
BeautifulSoup text dump if trafilatura comes back empty.
"""

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PersonaTwinBot/1.0; "
        "+https://example.com/bot)"
    )
}


def _fallback_extract(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_url(url: str, timeout: int = 20) -> str:
    import trafilatura

    response = requests.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    html = response.text

    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    if extracted and extracted.strip():
        return extracted.strip()

    # trafilatura found nothing useful (e.g. heavily JS-rendered page) - fall back
    fallback = _fallback_extract(html)
    if not fallback.strip():
        raise ValueError(f"Could not extract any readable text from {url}")
    return fallback
