"""Люба Partners — affiliate program integration (sochiautoparts.ru/partners.json)."""
import asyncio, json, logging, os, random, time
from typing import List, Dict, Optional
import httpx
from bot.config import config

logger = logging.getLogger("luba.partners")

CATEGORY_MAP = {
    "авто": ("autoparts", "Автозапчасти"),
    "автомобили и мотоциклы": ("autoparts", "Автозапчасти"),
    "товары для авто и мотоциклов": ("autoparts", "Автозапчасти"),
    "аренда машин": ("autorent", "Аренда авто"),
    "туризм, путешествия": ("travel", "Путешествия"),
    "электроника": ("electronics", "Электроника"),
    "одежда, обувь, аксессуары": ("shopping", "Покупки"),
    "товары для дома": ("home", "Для дома"),
}

class PartnerManager:
    def __init__(self):
        self.campaigns: List[Dict] = []
        self._last_load = 0.0

    async def load(self):
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
                resp = await c.get(config.PARTNERS_URL, headers={"User-Agent": "LyubaBot/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                self.campaigns = data.get("campaigns", [])
                self._last_load = time.time()
                logger.info(f"Loaded {len(self.campaigns)} partner campaigns")
        except Exception as e:
            logger.warning(f"partner load failed: {e}")

    def get_all_partner_links_for_dialog(self, text, max_programs=2):
        """Return relevant partner links based on message text."""
        if not self.campaigns: return []
        t = (text or "").lower()
        results = []
        for camp in self.campaigns:
            cats = " ".join(camp.get("categories", []) or []).lower()
            name = (camp.get("name") or "").lower()
            # Check if any category keyword matches the text
            matched = False
            for ru_cat, (key, label) in CATEGORY_MAP.items():
                if ru_cat in cats and any(w in t for w in ru_cat.split(",")[0].split()):
                    matched = True
                    break
            if matched:
                results.append({
                    "name": camp.get("name", ""),
                    "url": camp.get("goto_link", camp.get("site_url", "")),
                    "label": camp.get("categories", [""])[0] if camp.get("categories") else "",
                })
            if len(results) >= max_programs:
                break
        return results

    async def refresh_if_needed(self):
        if time.time() - self._last_load > config.PARTNER_REFRESH_HOURS * 3600:
            await self.load()

partner_manager = PartnerManager()
