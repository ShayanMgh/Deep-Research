import asyncio, requests, torch
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS           # <— new API
from transformers import pipeline
from functools import partial

# ---------- 1) WEB SEARCH (new DDGS API) ----------
def search_web(query: str, max_results: int = 5):
    """Return a list of dicts: title, href, body (snippet)."""
    with DDGS() as ddgs:                      # context manager cleans up sessions
        return list(ddgs.text(query,max_results=max_results))
    
# ---------- 2) PAGE SCRAPING ----------
def fetch_page_text(url: str) -> str:
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    except Exception:
        return ""