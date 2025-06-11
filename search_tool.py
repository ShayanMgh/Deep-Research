# search_tool.py
from duckduckgo_search import DDGS
import requests, bs4, textwrap

def web_search_digest(query: str, max_results: int = 5) -> str:
    """Return ~1 kB of raw text made of snippets and first paragraphs."""
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    digest_lines = []
    for h in hits:
        digest_lines.append(f"- {h['title']}: {h['body']}")
        try:
            html = requests.get(h["href"], timeout=5).text
            soup = bs4.BeautifulSoup(html, "html.parser")
            first_para = soup.find("p")
            if first_para:
                digest_lines.append(first_para.get_text(" ", strip=True))
        except Exception:
            pass
    digest = "\n".join(digest_lines)
    # trim to 1 kB so it fits easily in context
    return textwrap.shorten(digest, width=1024, placeholder=" […]")
