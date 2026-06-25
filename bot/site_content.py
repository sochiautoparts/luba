"""
Site Content Fetcher for Lyuba — fetches random products and posts from
sochiautoparts.ru to recommend in conversations.

- Products: scraped from https://sochiautoparts.ru/shop (50 product cards per page)
  Each card: <article class="shop-product-card"> with /shop/product/<slug>, img, name, price.
- Posts: links of form /post/<id> found on the homepage.

The fetcher caches results for 1 hour to avoid repeated HTTP requests, and
exposes simple pickers: random_product(), random_post(), relevant_product(text).
"""

import asyncio
import logging
import random
import re
import time
from html import unescape
from typing import List, Dict, Optional

import httpx

from bot.config import config

logger = logging.getLogger("luba.site_content")

_CACHE_TTL = 3600  # 1 hour
_product_cache: List[Dict] = []
_product_cache_time = 0.0
_post_cache: List[Dict] = []
_post_cache_time = 0.0
_lock = asyncio.Lock()

# Regexes (same structure as Asya's shop.py)
_PRODUCT_CARD_RE = re.compile(r'<article\s+class="shop-product-card">(.*?)</article>', re.DOTALL)
_CARD_LINK_RE = re.compile(r'href="/shop/product/([^"]+)"')
_CARD_IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"[^>]+alt="([^"]*)"')
_CARD_PRICE_RE = re.compile(r'<span\s+class="price">([^<]+)</span>')
_CARD_BADGE_RE = re.compile(r'<span\s+class="badge[^"]*"[^>]*>([^<]+)</span>')
_POST_LINK_RE = re.compile(r'href="/post/(\d+)"')


async def _fetch(url: str) -> str:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; LyubaBot/1.0)",
            "Accept-Language": "ru-RU,ru;q=0.9",
        })
    if resp.status_code != 200:
        return ""
    return resp.text


async def refresh_products(force: bool = False) -> int:
    """Fetch the shop page and parse product cards. Returns count cached."""
    global _product_cache, _product_cache_time
    async with _lock:
        now = time.time()
        if not force and _product_cache and (now - _product_cache_time) < _CACHE_TTL:
            return len(_product_cache)
        try:
            html = await _fetch(config.SHOP_URL)
            if not html:
                return len(_product_cache)
            products = []
            seen = set()
            for m in _PRODUCT_CARD_RE.finditer(html):
                block = m.group(1)
                link_m = _CARD_LINK_RE.search(block)
                if not link_m:
                    continue
                slug = unescape(link_m.group(1))
                if slug in seen:
                    continue
                seen.add(slug)
                img_m = _CARD_IMG_RE.search(block)
                image_url = img_m.group(1) if img_m else ""
                name = unescape(img_m.group(2)) if img_m else slug.replace("-", " ")
                price_m = _CARD_PRICE_RE.search(block)
                price = unescape(price_m.group(1)).strip() if price_m else ""
                badge_m = _CARD_BADGE_RE.search(block)
                supplier = unescape(badge_m.group(1)).strip() if badge_m else ""
                products.append({
                    "slug": slug,
                    "name": name,
                    "price": price,
                    "supplier": supplier,
                    "image_url": image_url,
                    "url": f"{config.SITE_URL}/shop/product/{slug}",
                })
            if products:
                _product_cache = products
                _product_cache_time = now
                logger.info(f"Site content: cached {len(products)} products")
        except Exception as e:
            logger.warning(f"refresh_products error: {e}")
        return len(_product_cache)


async def refresh_posts(force: bool = False) -> int:
    """Fetch the homepage and collect /post/<id> links. Returns count cached."""
    global _post_cache, _post_cache_time
    async with _lock:
        now = time.time()
        if not force and _post_cache and (now - _post_cache_time) < _CACHE_TTL:
            return len(_post_cache)
        try:
            html = await _fetch(config.SITE_URL)
            if not html:
                return len(_post_cache)
            ids = set()
            for m in _POST_LINK_RE.finditer(html):
                ids.add(m.group(1))
            posts = [{"id": pid, "url": f"{config.SITE_URL}/post/{pid}"} for pid in ids]
            if posts:
                _post_cache = posts
                _post_cache_time = now
                logger.info(f"Site content: cached {len(posts)} posts")
        except Exception as e:
            logger.warning(f"refresh_posts error: {e}")
        return len(_post_cache)


async def random_product() -> Optional[Dict]:
    """Return a random product dict, or None if none cached/fetched."""
    if not _product_cache:
        await refresh_products()
    if not _product_cache:
        return None
    return random.choice(_product_cache)


async def random_post() -> Optional[Dict]:
    if not _post_cache:
        await refresh_posts()
    if not _post_cache:
        return None
    return random.choice(_post_cache)


async def relevant_product(text: str) -> Optional[Dict]:
    """Find a product whose name matches keywords in text. Falls back to random."""
    if not _product_cache:
        await refresh_products()
    if not _product_cache:
        return None
    t = (text or "").lower()
    scored = []
    for p in _product_cache:
        name = p.get("name", "").lower()
        score = 0
        for word in name.split():
            if len(word) > 3 and word in t:
                score += 1
        if score > 0:
            scored.append((score, p))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]
    return random.choice(_product_cache)


def format_product_for_context(p: Dict) -> str:
    """Format a product as a compact context line for the AI."""
    parts = [p.get("name", "Товар")]
    if p.get("price"):
        parts.append(f"({p['price']})")
    if p.get("supplier"):
        parts.append(f"— {p['supplier']}")
    parts.append(p.get("url", ""))
    return " ".join(parts)


def format_post_for_context(post: Dict) -> str:
    return f"Свежий пост на сайте: {post.get('url', '')}"


async def init_site_content():
    """Background-ish init: pre-fetch products and posts at startup."""
    await refresh_products()
    await refresh_posts()
