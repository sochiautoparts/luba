"""Люба Site Content — fetches products and posts from sochiautoparts.ru."""
import asyncio, logging, random, re, time
from html import unescape
from typing import List, Dict, Optional
import httpx

logger = logging.getLogger("luba.site")

_CACHE_TTL = 3600
_product_cache: List[Dict] = []
_product_cache_time = 0.0
_post_cache: List[Dict] = []
_post_cache_time = 0.0
_lock = asyncio.Lock()

_PRODUCT_CARD_RE = re.compile(r'<article\s+class="shop-product-card">(.*?)</article>', re.DOTALL)
_CARD_LINK_RE = re.compile(r'href="/shop/product/([^"]+)"')
_CARD_IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"[^>]+alt="([^"]*)"')
_CARD_PRICE_RE = re.compile(r'<span\s+class="price">([^<]+)</span>')
_POST_LINK_RE = re.compile(r'href="/post/(\d+)"')

async def _fetch(url):
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
        resp = await c.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; LyubaBot/1.0)", "Accept-Language": "ru-RU,ru;q=0.9"})
    return resp.text if resp.status_code == 200 else ""

async def _refresh_products(force=False):
    global _product_cache, _product_cache_time
    async with _lock:
        if not force and _product_cache and time.time() - _product_cache_time < _CACHE_TTL:
            return
        html = await _fetch("https://sochiautoparts.ru/shop")
        if not html: return
        products = []
        for card_html in _PRODUCT_CARD_RE.findall(html):
            link_m = _CARD_LINK_RE.search(card_html)
            img_m = _CARD_IMG_RE.search(card_html)
            price_m = _CARD_PRICE_RE.search(card_html)
            if link_m:
                products.append({
                    "slug": link_m.group(1),
                    "url": f"https://sochiautoparts.ru/shop/product/{link_m.group(1)}",
                    "name": unescape(img_m.group(2)) if img_m else "",
                    "img": img_m.group(1) if img_m else "",
                    "price": price_m.group(1).strip() if price_m else "",
                })
        _product_cache = products
        _product_cache_time = time.time()
        logger.info(f"Loaded {len(products)} products")

async def _refresh_posts(force=False):
    global _post_cache, _post_cache_time
    async with _lock:
        if not force and _post_cache and time.time() - _post_cache_time < _CACHE_TTL:
            return
        html = await _fetch("https://sochiautoparts.ru")
        if not html: return
        post_ids = list(set(_POST_LINK_RE.findall(html)))[:20]
        _post_cache = [{"id": pid, "url": f"https://sochiautoparts.ru/post/{pid}"} for pid in post_ids]
        _post_cache_time = time.time()
        logger.info(f"Loaded {len(_post_cache)} posts")

async def init_site_content():
    try:
        await _refresh_products()
        await _refresh_posts()
    except Exception as e:
        logger.warning(f"site content init failed: {e}")

async def random_product() -> Optional[Dict]:
    if not _product_cache: await _refresh_products()
    return random.choice(_product_cache) if _product_cache else None

async def random_post() -> Optional[Dict]:
    if not _post_cache: await _refresh_posts()
    return random.choice(_post_cache) if _post_cache else None

async def relevant_product(text) -> Optional[Dict]:
    if not _product_cache: await _refresh_products()
    if not _product_cache: return None
    t = (text or "").lower()
    scored = []
    for p in _product_cache:
        name = p.get("name", "").lower()
        score = sum(1 for w in name.split() if len(w) > 3 and w in t)
        if score > 0: scored.append((score, p))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]
    return random.choice(_product_cache)

def format_product_for_context(p):
    return f"{p.get('name','?')} — {p.get('price','?')} — {p.get('url','')}"

def format_post_for_context(p):
    return f"{p.get('url','')}"
