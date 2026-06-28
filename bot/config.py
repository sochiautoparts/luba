"""
Люба Bot Configuration v2.1 — мульти-провайдерная бесплатная архитектура.

AI-провайдеры (порядок = приоритет, авто-включение при наличии ключа):
  1. Groq        — Llama 3.3 70B, ~0.4с, free tier (30 req/min, 14K req/day)
  2. Gemini      — gemini-2.0-flash, ~1с, free tier (15 req/min, 1500/day)
  3. Pollinations — GPT-OSS 20B, free, no auth (всегда доступен)
  4. OpenRouter  — Llama 3.3 70B free, free tier (50 req/day)
  5. HuggingFace — Qwen2.5-7B, если есть credits (часто 402 depleted)
  6. Cloudflare  — Workers AI, 10K req/day free (если есть ключ)
  Запас: статические фразы

Каждый провайдер имеет свой circuit breaker (3 errors → 120s cooldown).
Линейный failover: первый успех выигрывает, без гонок.

Бесплатные ключи (пользователь добавляет в GitHub Secrets):
  - GROQ_API_KEY       — https://console.groq.com/keys
  - GEMINI_API_KEY     — https://aistudio.google.com/apikey
  - OPENROUTER_API_KEY — https://openrouter.ai/keys
  - HF_TOKEN           — https://huggingface.co/settings/tokens
  - CF_ACCOUNT_ID_1 + CF_API_TOKEN_1 — https://dash.cloudflare.com/

Минимум для работы: НИЧЕГО (Pollinations free всегда доступен).
Рекомендуется: GROQ + GEMINI для скорости и надёжности.
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

    # ── AI Providers (free tiers) ──
    # Groq — Llama 3.3 70B, самый быстрый (~0.4с), free
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # Google Gemini — gemini-2.0-flash, free tier
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # Pollinations — GPT-OSS 20B, free, no auth (всегда работает)
    POLLINATIONS_API_KEY: str = os.getenv("POLLINATIONS_API_KEY", "")
    POLLINATIONS_API_KEY_2: str = os.getenv("POLLINATIONS_API_KEY_2", "")
    # OpenRouter — free models (Llama, Mistral)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    # HuggingFace — Qwen2.5-7B (часто 402 depleted на free tier)
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    # Cloudflare Workers AI — Mistral Small 3.1, 10K req/day free
    CF_ACCOUNT_ID_1: str = os.getenv("CF_ACCOUNT_ID_1", "")
    CF_API_TOKEN_1: str = os.getenv("CF_API_TOKEN_1", "")
    CF_ACCOUNT_ID_2: str = os.getenv("CF_ACCOUNT_ID_2", "")
    CF_API_TOKEN_2: str = os.getenv("CF_API_TOKEN_2", "")

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
    # Lyuba VERY ACTIVE: responds to mentions/replies ALWAYS, proactively
    # comments on 75% of other messages (was 60%).
    GROUP_PROACTIVE_PROB: float = float(os.getenv("GROUP_PROACTIVE_PROB", "0.75"))
    GROUP_MAX_PER_MINUTE: int = int(os.getenv("GROUP_MAX_PER_MINUTE", "15"))
    GROUP_MEMORY_SIZE: int = int(os.getenv("GROUP_MEMORY_SIZE", "30"))

    # ── Web verification ──
    # Always verify events/facts (was 85%) — Lyuba uses web search actively
    SEARCH_TIMEOUT_SECONDS: int = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "8"))
    WEB_VERIFY_PROB: float = float(os.getenv("WEB_VERIFY_PROB", "1.0"))

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

    def active_providers(self) -> List[str]:
        """Список активных провайдеров (с ключами), в порядке приоритета."""
        providers = []
        if self.GROQ_API_KEY:
            providers.append("groq")
        if self.GEMINI_API_KEY:
            providers.append("gemini")
        providers.append("pollinations")  # всегда (free, no auth)
        if self.OPENROUTER_API_KEY:
            providers.append("openrouter")
        if self.HF_TOKEN:
            providers.append("huggingface")
        if self.CF_API_TOKEN_1:
            providers.append("cloudflare")
        return providers

    def providers_status(self) -> str:
        """Краткий статус провайдеров для лога."""
        active = self.active_providers()
        return f"active={active}, total={len(active)}"


config = BotConfig()
