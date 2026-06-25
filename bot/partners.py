"""
Partner / Affiliate Program Integration for Люба.

Same source as Asya: https://sochiautoparts.ru/partners.json
Schema:
    {
      "updated": "<iso-ts>",
      "campaigns": [
        { "id", "name", "logo", "goto_link", "site_url", "regions", "categories" },
        ...
      ]
    }

Features:
- Downloads partners.json from sochiautoparts.ru (or uses local cache file).
- Auto-refreshes every PARTNER_REFRESH_HOURS.
- Category mapping (RU strings -> internal keys + human labels).
- Relevance matching: given a message text, returns the most relevant partner
  links (autoparts / tires / tools / travel / electronics / shopping / etc.).
- Uses goto_link EXACTLY as provided — no subid modifications.
- Privacy: only the goto_link is surfaced in chat; no personal data collected.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import List, Dict, Optional

import httpx

from bot.config import config

logger = logging.getLogger("luba.partners")


# ── Category mapping: RU category strings -> internal key + human label ───────

CATEGORY_MAP = {
    # autoparts
    "авто": ("autoparts", "Автозапчасти"),
    "автомобили и мотоциклы": ("autoparts", "Автозапчасти"),
    "товары для авто и мотоциклов": ("autoparts", "Автозапчасти"),
    "аренда машин": ("autorent", "Аренда авто"),
    # tires / tools fall under autoparts umbrella
    # travel
    "туризм, путешествия": ("travel", "Путешествия"),
    "билеты на самолеты": ("travel", "Авиабилеты"),
    "транспортные услуги": ("travel", "Транспорт"),
    # shopping / marketplaces
    "маркетплейсы (включая товары из китая)": ("marketplace", "Маркетплейсы"),
    "интернет-магазины": ("shopping", "Магазины"),
    # electronics
    "электроника и бытовая техника": ("electronics", "Электроника"),
    # home
    "мебель и товары для дома": ("home", "Для дома"),
    "diy и товары для сада": ("home", "Для дома и сада"),
    # kids
    "игрушки и товары для детей": ("kids", "Детские товары"),
    # beauty / health
    "косметика, гигиена, аптеки": ("beauty", "Красота и здоровье"),
    # clothing
    "одежда, обувь, аксессуары": ("fashion", "Одежда и обувь"),
    # telecom
    "телекоммуникационные услуги": ("telecom", "Связь и интернет"),
    "интернет-услуги": ("telecom", "Интернет-сервисы"),
    # edu
    "онлайн-образование": ("education", "Образование"),
    # gifts
    "подарки и цветы": ("gifts", "Подарки и цветы"),
    # sport
    "спортивные товары": ("sport", "Спорт"),
    # hobby
    "хобби и канцтовары": ("hobby", "Хобби и канцелярия"),
    # other
    "прочие услуги": ("other", "Сервисы"),
    "smart tracking programs": ("other", "Сервисы"),
}

# Keyword → category key, for relevance matching against message text
KEYWORD_CATEGORIES = {
    "autoparts": ["запчаст", "детал", "артикул", "vin", "вин", "масло", "фильтр", "колодк",
                  "амортизатор", "свеч", "ремень", "ролик", "подшипник", "тормоз", "кузов",
                  "двигатель", "коробк", "акпп", "мкпп", "авто", "машин", "кузовн"],
    "autorent": ["аренд", "прокат", "каршеринг", "взять машину напрокат"],
    "travel": ["путешеств", "тур", "отпуск", "билет", "самолет", "авиа", "рейс", "перелёт",
               "перелет", "гостиниц", "отель", "бронир"],
    "marketplace": ["маркетплейс", "алиэкспресс", "taobao", "китай", "товары из китая",
                    "ozon", "wildberries", "заказ"],
    "shopping": ["купить", "магазин", "заказать", "цена", "сколько стоит", "где взять",
                 "приобрести"],
    "electronics": ["телефон", "смартфон", "ноутбук", "планшет", "техник", "электроник",
                    "бытовая техника", "гаджет", "наушник"],
    "home": ["мебель", "дом", "сад", "дача", "ремонт квартиры", "инструмент"],
    "kids": ["ребен", "дет", "игрушк", "малыш"],
    "beauty": ["косметик", "витамин", "аптек", "красота", "уход", "гигиен"],
    "fashion": ["одежд", "обувь", "кроссовк", "куртк", "платье", "时尚", "аксессуар"],
    "telecom": ["интернет", "связь", "сим-карта", "мобильн", "оператор", "vpn", "esim"],
    "education": ["курс", "обучение", "образование", "урок", "школа", "универ"],
    "gifts": ["подарок", "подарки", "цветы", "букет"],
    "sport": ["спорт", "фитнес", "тренажёр", "тренажер", "бег", "велосипед"],
    "hobby": ["хобби", "рукоделие", "канцеляр", "творчество"],
}


class PartnerManager:
    def __init__(self):
        self._campaigns: List[Dict] = []
        self._last_load = 0.0
        self._lock = asyncio.Lock()

    @property
    def campaigns(self) -> List[Dict]:
        return self._campaigns

    def _categorize(self, campaign: Dict) -> List[str]:
        cats = [c.lower().strip() for c in campaign.get("categories", [])]
        keys = set()
        for c in cats:
            if c in CATEGORY_MAP:
                keys.add(CATEGORY_MAP[c][0])
        return list(keys)

    def _label(self, campaign: Dict) -> str:
        cats = [c.lower().strip() for c in campaign.get("categories", [])]
        labels = []
        for c in cats:
            if c in CATEGORY_MAP:
                labels.append(CATEGORY_MAP[c][1])
        return labels[0] if labels else "Сервис"

    async def load(self, force: bool = False) -> None:
        async with self._lock:
            now = time.time()
            if not force and self._campaigns and (now - self._last_load) < config.PARTNER_REFRESH_HOURS * 3600:
                return
            data = None
            # 1) Try local cache file
            cache = config.ADMITAD_ADS_FILE
            if cache and os.path.exists(cache):
                try:
                    with open(cache, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"Loaded partners from cache: {len(data.get('campaigns', []))} campaigns")
                except Exception as e:
                    logger.debug(f"Cache load failed: {e}")
            # 2) Try remote refresh
            if not data or force:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        resp = await client.get(config.PARTNERS_URL)
                    if resp.status_code == 200:
                        data = resp.json()
                        # Save cache
                        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
                        with open(cache, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False)
                        logger.info(f"Refreshed partners from {config.PARTNERS_URL}: "
                                    f"{len(data.get('campaigns', []))} campaigns")
                except Exception as e:
                    logger.warning(f"Remote partner refresh failed: {e}")
            if data and data.get("campaigns"):
                self._campaigns = data["campaigns"]
                self._last_load = time.time()

    def _filter_ru_relevant(self, campaign: Dict) -> bool:
        """Keep campaigns available in RU or WW (broad audience)."""
        regions = [r.upper() for r in campaign.get("regions", [])]
        if not regions:
            return True
        return "RU" in regions or "00" in regions or "WW" in regions or len(regions) > 20

    def get_all_partner_links_for_dialog(self, text: str, max_programs: int = 3) -> List[Dict]:
        """Return up to max_programs partner links relevant to the text."""
        if not self._campaigns:
            return []
        t = (text or "").lower()
        scored: List[tuple] = []
        for c in self._campaigns:
            if not self._filter_ru_relevant(c):
                continue
            keys = self._categorize(c)
            score = 0
            matched_key = None
            for key in keys:
                kws = KEYWORD_CATEGORIES.get(key, [])
                for kw in kws:
                    if kw in t:
                        score += 2
                        matched_key = key
                        break
            if score > 0:
                scored.append((score, c, matched_key))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[0])
        result = []
        seen = set()
        for score, c, matched_key in scored:
            name = c.get("name", "")
            url = c.get("goto_link", "")
            if not url or name in seen:
                continue
            seen.add(name)
            # Prefer the label of the matched category
            label = self._label_for_key(matched_key) or self._label(c)
            result.append({
                "name": name,
                "url": url,
                "label": label,
                "site": c.get("site_url", ""),
            })
            if len(result) >= max_programs:
                break
        return result

    def _label_for_key(self, key: Optional[str]) -> str:
        for cat_str, (k, label) in CATEGORY_MAP.items():
            if k == key:
                return label
        return ""

    def format_primary_parts_links(self) -> str:
        """Format the primary autoparts partners (Rossko/Autopiter/AvtoALL-like) for context."""
        parts = [c for c in self._campaigns
                 if "autoparts" in self._categorize(c) and self._filter_ru_relevant(c)]
        if not parts:
            return ""
        lines = ["Партнёрские ссылки (используй КАК ЕСТЬ, если уместно):"]
        for c in parts[:5]:
            lines.append(f"- {c.get('name', '')}: {c.get('goto_link', '')}")
        return "\n".join(lines)

    def random_partner(self) -> Optional[Dict]:
        if not self._campaigns:
            return None
        c = random.choice(self._campaigns)
        return {"name": c.get("name", ""), "url": c.get("goto_link", ""), "label": self._label(c)}


partner_manager = PartnerManager()
