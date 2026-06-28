"""
Люба Bot Configuration v2.0 — упрощённый.

Убрано:
  - Локальная модель (source of segfaults, 15s timeouts)
  - Optional providers (Groq/Gemini/OpenRouter/Cloudflare) — не настроены,
    только усложняли код. HF Qwen2.5-7B как primary + Pollinations free
    как backup покрывают все потребности.

Оставлено:
  - Telegram (BOT_TOKEN, OWNER_ID)
  - HuggingFace (HF_TOKEN) — primary AI provider
  - Pollinations (free, no auth) — backup
  - GitHub PAT для self-dispatch
  - SQLite DB
  - Partners / site content
  - Group activity tuning
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class BotConfig:
    """Main bot configuration loaded from environment variables."""

    # ── Telegram ──
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
    BOT_USERNAME: str = os.getenv("BOT_USERNAME", "@asluba_bot")
    BOT_ID: int = int(os.getenv("BOT_ID", "8614302697"))
    OWNER_ID: int = int(os.getenv("OWNER_ID", "265070804"))

    # ── HuggingFace (PRIMARY AI provider) ──
    # router.huggingface.co — Qwen2.5-7B-Instruct, ~0.8s, стабильно
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # ── Pollinations (BACKUP, free, no auth — always available) ──
    POLLINATIONS_API_KEY: str = os.getenv("POLLINATIONS_API_KEY", "")
    POLLINATIONS_API_KEY_2: str = os.getenv("POLLINATIONS_API_KEY_2", "")

    # ── GitHub PAT for 24/7 self-dispatch ──
    GH_PAT_TOKEN: str = os.getenv("GH_PAT_TOKEN", "")
    GH_REPO: str = os.getenv("GH_REPO", "sochiautoparts/luba")

    # ── Database ──
    DB_PATH: str = os.getenv("DB_PATH", "luba.db")

    # ── Partners / affiliate ──
    PARTNERS_URL: str = os.getenv("PARTNERS_URL", "https://sochiautoparts.ru/partners.json")
    ADMITAD_ADS_FILE: str = os.getenv("ADMITAD_ADS_FILE", "data/partners.json")
    PARTNER_REFRESH_HOURS: int = int(os.getenv("PARTNER_REFRESH_HOURS", "6"))

    # ── Group activity tuning ──
    # Lyuba VERY ACTIVE in groups: responds to mentions/replies ALWAYS,
    # proactively comments on 60% of other messages.
    # safe_send handles RetryAfter (waits + retries) so we can be very active.
    GROUP_PROACTIVE_PROB: float = float(os.getenv("GROUP_PROACTIVE_PROB", "0.60"))
    GROUP_MAX_PER_MINUTE: int = int(os.getenv("GROUP_MAX_PER_MINUTE", "15"))
    GROUP_MEMORY_SIZE: int = int(os.getenv("GROUP_MEMORY_SIZE", "30"))

    # ── Web verification ──
    SEARCH_TIMEOUT_SECONDS: int = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "6"))
    WEB_VERIFY_PROB: float = float(os.getenv("WEB_VERIFY_PROB", "0.85"))

    # ── Channels Lyuba recommends ──
    RECOMMEND_CHANNELS: List[str] = field(default_factory=lambda: [
        "https://t.me/sochiautoparts",
        "https://t.me/bmw_mpower_club",
    ])
    SHOP_URL: str = os.getenv("SHOP_URL", "https://sochiautoparts.ru/shop")
    SITE_URL: str = os.getenv("SITE_URL", "https://sochiautoparts.ru")

    # ── Reactions (likes) ──
    REACTION_PROB: float = float(os.getenv("REACTION_PROB", "0.45"))
    CHANNEL_REACTION_PROB: float = float(os.getenv("CHANNEL_REACTION_PROB", "0.65"))

    # ── AI tuning ──
    CHAT_TEMPERATURE: float = 0.9
    CHAT_MAX_CHARS: int = 1200
    COMMENT_MAX_CHARS: int = 500
    GROUP_MAX_CHARS: int = 700

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def BOT_HANDLE(self) -> str:
        return self.BOT_USERNAME.lstrip("@")

    def has_pollinations_key(self) -> bool:
        return bool(self.POLLINATIONS_API_KEY or self.POLLINATIONS_API_KEY_2)

    def providers_status(self) -> str:
        """Краткий статус провайдеров для лога."""
        parts = []
        parts.append(f"HF={'ON' if self.HF_TOKEN else 'OFF'}")
        parts.append("Pollinations=ON(free)")
        return ", ".join(parts)


config = BotConfig()
