"""
Multi-Engine Web Search for Люба — DuckDuckGo HTML → DDG lite → Yandex.

Used for web verification of facts/news/prices that Lyuba mentions or is asked about.
Returns lightweight SearchResult list. No API keys required (all free endpoints).
"""

import asyncio
import logging
import re
from typing import List, Dict, Optional
from urllib.parse import quote_plus

import httpx

from bot.config import config

logger = logging.getLogger("luba.web_search")


class SearchResult:
    def __init__(self, title: str, url: str, snippet: str = "", source: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "source": self.source}


DDG_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _clean_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"&quot;", '"', s)
    s = re.sub(r"&#39;|&apos;", "'", s)
    s = re.sub(r"&lt;", "<", s)
    s = re.sub(r"&gt;", ">", s)
    return re.sub(r"\s+", " ", s).strip()


async def search_ddg_html(query: str, max_results: int = 5) -> List[SearchResult]:
    results: List[SearchResult] = []
    try:
        async with httpx.AsyncClient(timeout=config.SEARCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
            params = {"q": query, "kl": "ru-ru", "no_redirect": "1"}
            resp = await client.get("https://html.duckduckgo.com/html/", params=params, headers=DDG_HEADERS)
            if resp.status_code == 202:
                # rate-limited — try lite
                resp = await client.get("https://lite.duckduckgo.com/lite/",
                                        params={"q": query, "kl": "ru-ru"}, headers=DDG_HEADERS)
                if resp.status_code != 200:
                    return results
                urls = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"', resp.text)
                titles = re.findall(r'<a[^>]+class="result-link"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                snippets = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', resp.text, re.DOTALL)
                for i, url in enumerate(urls[:max_results]):
                    title = _clean_html(titles[i]) if i < len(titles) else ""
                    snippet = _clean_html(snippets[i]) if i < len(snippets) else ""
                    if url and title:
                        results.append(SearchResult(title, url, snippet, "duckduckgo_lite"))
                return results
            if resp.status_code != 200:
                return results
            blocks = re.findall(
                r'<a rel="nofollow" class="result__a" href="([^"]+?)".*?>(.*?)</a>.*?'
                r'<a class="result__snippet".*?>(.*?)</a>',
                resp.text, re.DOTALL,
            )
            for url, title, snippet in blocks[:max_results]:
                results.append(SearchResult(_clean_html(title), url, _clean_html(snippet), "duckduckgo"))
    except Exception as e:
        logger.debug(f"DDG search error: {e}")
    return results


async def search_yandex(query: str, max_results: int = 5) -> List[SearchResult]:
    """Yandex search via HTML scraping (best-effort)."""
    results: List[SearchResult] = []
    try:
        async with httpx.AsyncClient(timeout=config.SEARCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.get(
                "https://yandex.ru/search/",
                params={"text": query, "lr": 213},
                headers={**DDG_HEADERS, "Cookie": "yandex_gid=213"},
            )
            if resp.status_code != 200:
                return results
            # Yandex often returns a JS page or captcha; parse what we can
            blocks = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', resp.text)
            for url, title in blocks[:max_results]:
                title = _clean_html(title)
                if title and len(title) > 5 and "yandex" not in url:
                    results.append(SearchResult(title, url, "", "yandex"))
    except Exception as e:
        logger.debug(f"Yandex search error: {e}")
    return results


async def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Try DDG first, then Yandex. Returns combined unique results."""
    results = await search_ddg_html(query, max_results=max_results)
    if not results:
        results = await search_yandex(query, max_results=max_results)
    # Dedup by URL
    seen = set()
    unique = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    return unique[:max_results]


def format_search_results(results: List[SearchResult], max_items: int = 3) -> str:
    if not results:
        return ""
    lines = []
    for r in results[:max_items]:
        snippet = f" — {r.snippet}" if r.snippet else ""
        lines.append(f"• {r.title}{snippet}\n  {r.url}")
    return "\n".join(lines)


async def verify_claim(claim: str, fast: bool = True) -> str:
    """Verify a factual claim via web search. Returns formatted context string.

    Fast mode (default): shorter timeout (5s), fewer results — optimized for
    quick fact-checks during conversation so the user doesn't wait long.
    """
    timeout = 5.0 if fast else config.SEARCH_TIMEOUT_SECONDS
    try:
        import asyncio as _a
        results = await _a.wait_for(web_search(claim, max_results=3), timeout=timeout)
    except _a.TimeoutError:
        return ""
    except Exception:
        return ""
    if not results:
        return ""
    return format_search_results(results, max_items=2)
