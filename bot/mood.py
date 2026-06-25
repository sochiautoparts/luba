"""
Lyuba Mood System — dynamic, human-like emotional state.

Mood is influenced by:
  - Time of day (morning sleepy, day energetic, evening relaxed, night owl)
  - Slow random drift (so it's not deterministic)
  - Recent conversation sentiment (if someone was warm → Lyuba warms up;
    if rude → mildly put off but stays polite)

The mood is stored in DB (single row) and refreshed periodically. It is injected
into the system prompt so the model's tone reflects it.
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Tuple

from bot import database as db

logger = logging.getLogger("luba.mood")


# Time-of-day base moods (Moscow time)
def _time_base_mood() -> Tuple[str, float]:
    try:
        from zoneinfo import ZoneInfo
        hour = datetime.now(ZoneInfo("Europe/Moscow")).hour
    except Exception:
        hour = datetime.utcnow().hour
    if 5 <= hour < 12:
        return ("сонная, тёплая", 0.45)
    if 12 <= hour < 18:
        return ("бодрая, приветливая", 0.8)
    if 18 <= hour < 23:
        return ("расслабленная, задумчивая", 0.55)
    return ("ночная, философская", 0.4)


# Sentiment modifiers — adjust energy/mood label based on recent messages
WARM_WORDS = ["спасибо", "класс", "супер", "обожаю", "люблю", "здорово", "мило", "добр", "привет"]
RUDE_WORDS = ["тупая", "дура", "идиот", "ненавижу", "заткнись", "отстой", "ужас", "бяка"]


def detect_sentiment(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in WARM_WORDS):
        return "warm"
    if any(w in t for w in RUDE_WORDS):
        return "rude"
    return "neutral"


# Mood label variants for variety
MOOD_VARIANTS = {
    "warm": ["тёплая, радостная", "в приподнятом настроении", "улыбается"],
    "rude": ["немного уставшая от грубости, но вежливая", "спокойная, сдержанная"],
    "neutral": ["спокойная", "ровная", "в хорошем настроении"],
}


async def update_mood_from_message(text: str) -> None:
    """Nudge mood based on a received message sentiment."""
    sentiment = detect_sentiment(text)
    if sentiment == "neutral":
        return
    current = await db.get_mood()
    base_label, base_energy = _time_base_mood()
    if sentiment == "warm":
        energy = min(1.0, current.get("energy", base_energy) + 0.1)
        label = random.choice(MOOD_VARIANTS["warm"])
    else:  # rude
        energy = max(0.2, current.get("energy", base_energy) - 0.1)
        label = random.choice(MOOD_VARIANTS["rude"])
    await db.set_mood(label, energy)


async def refresh_mood() -> str:
    """Periodic drift: blend time-of-day base with a little randomness."""
    base_label, base_energy = _time_base_mood()
    # Small random drift
    energy = max(0.2, min(1.0, base_energy + random.uniform(-0.1, 0.1)))
    # Occasionally pick a variant for variety
    variants = ["спокойная", "в хорошем настроении", "бодрая", "тёплая", "задумчивая"]
    label = random.choice(variants) if random.random() < 0.3 else base_label
    await db.set_mood(label, energy)
    return label


async def current_mood_descriptor() -> str:
    """Return a short mood descriptor for the system prompt."""
    current = await db.get_mood()
    label = current.get("mood", "спокойная")
    energy = current.get("energy", 0.5)
    # Compose
    if energy > 0.7:
        return f"{label}, энергичная"
    if energy < 0.4:
        return f"{label}, тихая"
    return label


async def mood_loop():
    """Background task: refresh mood every ~15 minutes."""
    while True:
        try:
            await refresh_mood()
        except Exception as e:
            logger.debug(f"mood refresh error: {e}")
        await asyncio.sleep(900)
