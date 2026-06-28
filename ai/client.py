"""
AI Client v2.1 — мульти-провайдерная бесплатная архитектура.

Стратегия: ЛИНЕЙНЫЙ FAILOVER по приоритету.
  1. Groq (Llama 3.3 70B)     — ~0.4с, free, если GROQ_API_KEY
  2. Gemini (2.0-flash)        — ~1с, free, если GEMINI_API_KEY
  3. Pollinations (GPT-OSS 20B)— ~1с, free, ВСЕГДА (no auth)
  4. OpenRouter (Llama 70B free)— ~1.5с, если OPENROUTER_API_KEY
  5. HuggingFace (Qwen2.5-7B)  — ~0.8с, если HF_TOKEN (часто 402 depleted)
  6. Cloudflare (Mistral 3.1)  — ~1с, если CF_API_TOKEN_1
  Запас: статические фразы

Каждый провайдер имеет circuit breaker (3 errors → 120s cooldown).
Первый успех выигрывает — без гонок, без траты ресурсов всех провайдеров.

Бесплатные ключи (пользователь добавляет в GitHub Secrets):
  - GROQ_API_KEY       — https://console.groq.com/keys       (самый рекомендуемый)
  - GEMINI_API_KEY     — https://aistudio.google.com/apikey
  - OPENROUTER_API_KEY — https://openrouter.ai/keys
  - HF_TOKEN           — https://huggingface.co/settings/tokens
  - CF_ACCOUNT_ID_1 + CF_API_TOKEN_1 — https://dash.cloudflare.com/

Минимум для работы: НИЧЕГО (Pollinations free всегда доступен).
Рекомендуется: GROQ + GEMINI для скорости и надёжности (убирают зависимость
от единственного free провайдера).
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Optional, List, Tuple, Callable

import httpx

from ai.persona import PERSONA_PROMPT, VISION_PROMPT
from ai.clean import clean_response, contains_non_cyrillic_script
from bot.config import config
from bot import database as db

logger = logging.getLogger("luba.ai")


# ── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """Простой circuit breaker: N ошибок → cooldown → один retry."""

    def __init__(self, name: str, threshold: int = 3, cooldown: int = 120):
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self.errors = 0
        self.open_until = 0.0

    def ok(self) -> bool:
        if self.errors >= self.threshold:
            if time.time() < self.open_until:
                return False
            # Cooldown истёк — даём одну попытку
            self.errors = self.threshold - 1
        return True

    def fail(self) -> None:
        self.errors += 1
        if self.errors >= self.threshold:
            self.open_until = time.time() + self.cooldown
            logger.warning(
                f"Circuit breaker '{self.name}' OPEN after {self.errors} "
                f"errors — cooldown {self.cooldown}s"
            )

    def success(self) -> None:
        self.errors = 0

    def status(self) -> dict:
        return {
            "errors": self.errors,
            "open": self.errors >= self.threshold and time.time() < self.open_until,
        }


# ── Static fallbacks ─────────────────────────────────────────────────────────

_CHAT_FALLBACKS = [
    "ой, у меня что-то мысль застряла 🙈 давай ещё раз?",
    "секунду, я немного зависла… повторишь?",
    "блин, связь капризничает. напиши ещё разок?",
]
_COMMENT_FALLBACKS = ["интересно 😊", "согласна", "ого", "хех, жизненно"]


# ── Time context (Moscow) ────────────────────────────────────────────────────

def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "зима (в Сочи мягкая, +10..+15)"
    if month in (3, 4, 5):
        return "весна"
    if month in (6, 7, 8):
        return "лето (жарко, тёплое море)"
    return "осень"


def _time_context() -> str:
    """Контекст времени как обычные предложения (НЕ маркированный блок)."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Europe/Moscow"))
    except Exception:
        now = datetime.utcnow()
    hour = now.hour
    if 5 <= hour < 12:
        tod, mood = "утро", "ты только проснулась, пьёшь кофе, сонная"
    elif 12 <= hour < 18:
        tod, mood = "день", "ты бодрая, в середине рабочего дня"
    elif 18 <= hour < 23:
        tod, mood = "вечер", "ты расслабленная, устала за день"
    else:
        tod, mood = "ночь", "тебе не спится, сова"
    weekdays = ["понедельник", "вторник", "среда", "четверг",
                "пятница", "суббота", "воскресенье"]
    return (
        f"Сейчас {now.strftime('%d.%m.%Y')} {now.strftime('%H:%M')} по Москве. "
        f"День недели: {weekdays[now.weekday()]}. Время суток: {tod}, {mood}. "
        f"Сезон: {_season(now.month)}. Ты в Сочи."
    )


# ── Provider implementations ─────────────────────────────────────────────────
# Каждый провайдер — функция (http, messages, max_tokens) -> (text, http_status)
# Возвращает (None, status) при ошибке, (text, 200) при успехе.

async def _groq_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """Groq — Llama 3.3 70B. Самый быстрый (~0.4с). Free tier 30 req/min."""
    try:
        resp = await http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": config.CHAT_TEMPERATURE,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.warning(f"Groq HTTP {resp.status_code}: {resp.text[:150]}")
            return None, resp.status_code
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None, 200
        text = choices[0].get("message", {}).get("content", "") or ""
        text = text.strip()
        return (text or None), 200
    except Exception as e:
        logger.warning(f"Groq exception: {type(e).__name__}: {e}")
        return None, 0


async def _gemini_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """Google Gemini 2.0-flash. Free tier 15 req/min, 1500/day."""
    try:
        # Gemini REST API: convert OpenAI messages → Gemini format
        contents = []
        sys_text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                sys_text += content + "\n"
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": config.CHAT_TEMPERATURE,
                "maxOutputTokens": max_tokens,
            },
        }
        if sys_text:
            payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
        resp = await http.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash:generateContent",
            json=payload,
            params={"key": config.GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            timeout=20.0,
        )
        if resp.status_code != 200:
            logger.warning(f"Gemini HTTP {resp.status_code}: {resp.text[:150]}")
            return None, resp.status_code
        data = resp.json()
        cands = data.get("candidates", [])
        if not cands:
            return None, 200
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return (text or None), 200
    except Exception as e:
        logger.warning(f"Gemini exception: {type(e).__name__}: {e}")
        return None, 0


async def _pollinations_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """Pollinations GPT-OSS 20B. Free, no auth. Всегда доступен (rate-limited)."""
    try:
        headers = {"Content-Type": "application/json"}
        # Если есть API key — используем для лучших rate limits
        if config.POLLINATIONS_API_KEY:
            headers["Authorization"] = f"Bearer {config.POLLINATIONS_API_KEY}"
        resp = await http.post(
            "https://text.pollinations.ai/openai",
            json={
                "model": "openai",
                "messages": messages,
                "temperature": config.CHAT_TEMPERATURE,
                "max_tokens": max_tokens,
                "private": True,
                "referrer": "asluba_bot",
            },
            headers=headers,
            timeout=20.0,
        )
        if resp.status_code == 429:
            logger.warning("Pollinations 429 (rate-limited on free tier)")
            return None, 429
        if resp.status_code != 200:
            logger.warning(f"Pollinations HTTP {resp.status_code}: {resp.text[:100]}")
            return None, resp.status_code
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None, 200
        msg = choices[0].get("message", {})
        text = (msg.get("content", "") if isinstance(msg, dict) else "") or ""
        text = text.strip()
        return (text or None), 200
    except Exception as e:
        logger.warning(f"Pollinations exception: {type(e).__name__}: {e}")
        return None, 0


async def _openrouter_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """OpenRouter free models: Llama 3.3 70B. Free tier 50 req/day."""
    try:
        resp = await http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": messages,
                "temperature": config.CHAT_TEMPERATURE,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sochiautoparts.ru",
                "X-Title": "Lyuba Bot",
            },
            timeout=25.0,
        )
        if resp.status_code != 200:
            logger.warning(f"OpenRouter HTTP {resp.status_code}: {resp.text[:150]}")
            return None, resp.status_code
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None, 200
        text = choices[0].get("message", {}).get("content", "") or ""
        text = text.strip()
        return (text or None), 200
    except Exception as e:
        logger.warning(f"OpenRouter exception: {type(e).__name__}: {e}")
        return None, 0


async def _hf_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """HuggingFace Qwen2.5-7B. Часто 402 (credits depleted)."""
    try:
        resp = await http.post(
            "https://router.huggingface.co/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": messages,
                "temperature": config.CHAT_TEMPERATURE,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {config.HF_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=25.0,
        )
        if resp.status_code != 200:
            body = resp.text[:200] if resp.status_code in (401, 402, 403, 429) else resp.text[:80]
            logger.warning(f"HF HTTP {resp.status_code}: {body}")
            return None, resp.status_code
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None, 200
        text = choices[0].get("message", {}).get("content", "") or ""
        text = text.strip()
        if not text:
            return None, 200
        if contains_non_cyrillic_script(text):
            logger.warning("HF response rejected: non-Cyrillic script (hallucination)")
            return None, 200
        return text, 200
    except Exception as e:
        logger.warning(f"HF exception: {type(e).__name__}: {e}")
        return None, 0


async def _cloudflare_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """Cloudflare Workers AI — Mistral Small 3.1. 10K req/day free."""
    try:
        account = config.CF_ACCOUNT_ID_1
        token = config.CF_API_TOKEN_1
        resp = await http.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/"
            f"@cf/mistralai/mistral-small-3.1-24b-instruct",
            json={
                "messages": messages,
                "temperature": config.CHAT_TEMPERATURE,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=25.0,
        )
        if resp.status_code != 200:
            logger.warning(f"Cloudflare HTTP {resp.status_code}: {resp.text[:150]}")
            return None, resp.status_code
        data = resp.json()
        if not data.get("success"):
            return None, 200
        res = data.get("result", {})
        text = res.get("response", "") or ""
        if not text and isinstance(res.get("choices"), list) and res["choices"]:
            text = res["choices"][0].get("message", {}).get("content", "")
        text = text.strip()
        return (text or None), 200
    except Exception as e:
        logger.warning(f"Cloudflare exception: {type(e).__name__}: {e}")
        return None, 0


# ── Provider registry ────────────────────────────────────────────────────────
# (name, is_active_fn, chat_fn, circuit_breaker)
# Порядок = приоритет.

def _build_providers() -> List[Tuple[str, Callable, CircuitBreaker]]:
    """Собирает список активных провайдеров в порядке приоритета."""
    providers: List[Tuple[str, Callable, CircuitBreaker]] = []
    if config.GROQ_API_KEY:
        providers.append(("groq", _groq_chat, CircuitBreaker("groq", threshold=3, cooldown=120)))
    if config.GEMINI_API_KEY:
        providers.append(("gemini", _gemini_chat, CircuitBreaker("gemini", threshold=3, cooldown=120)))
    # Pollinations — ВСЕГДА (free, no auth)
    providers.append(("pollinations", _pollinations_chat, CircuitBreaker("pollinations", threshold=5, cooldown=60)))
    if config.OPENROUTER_API_KEY:
        providers.append(("openrouter", _openrouter_chat, CircuitBreaker("openrouter", threshold=3, cooldown=180)))
    if config.HF_TOKEN:
        providers.append(("huggingface", _hf_chat, CircuitBreaker("huggingface", threshold=2, cooldown=600)))
    if config.CF_API_TOKEN_1:
        providers.append(("cloudflare", _cloudflare_chat, CircuitBreaker("cloudflare", threshold=3, cooldown=180)))
    return providers


# ── AI Client ────────────────────────────────────────────────────────────────

class AIClient:
    """Единый AI клиент: мульти-провайдерный linear failover."""

    HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    HF_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
    GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self._providers: List[Tuple[str, Callable, CircuitBreaker]] = []
        # Статистика
        self._total = 0
        self._provider_ok: dict = {}
        self._static = 0

    async def initialize(self) -> None:
        """Создаёт singleton httpx client с connection pooling + строит провайдеров."""
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0, read=25.0),
            limits=httpx.Limits(
                max_connections=30,
                max_keepalive_connections=15,
                keepalive_expiry=30.0,
            ),
            headers={"User-Agent": "LyubaBot/2.1"},
        )
        self._providers = _build_providers()
        names = [p[0] for p in self._providers]
        for name in names:
            self._provider_ok[name] = 0
        logger.info(
            f"AI Client v2.1 ready — providers (priority order): {names}. "
            f"{config.providers_status()}"
        )

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # ── System prompt builder ──

    def _build_system(self, extra_context: str = "", mood: str = "",
                      vision: bool = False) -> str:
        """Собирает system prompt: персона + время + настроение + контекст."""
        parts = [PERSONA_PROMPT]
        if vision:
            parts.append(VISION_PROMPT)
        parts.append(_time_context())
        if mood:
            parts.append(f"Твоё настроение сейчас: {mood}.")
        if extra_context:
            parts.append(extra_context)
        return "\n\n".join(parts)

    # ── Public API ──

    async def chat(self, user_id: int, message: str,
                   extra_context: str = "", mood: str = "",
                   max_chars: int = None) -> str:
        """Личный чат: с историей диалога."""
        self._total += 1
        sys_prompt = self._build_system(extra_context, mood)
        history = await db.get_chat_history(user_id, limit=8)

        messages = [{"role": "system", "content": sys_prompt}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"][:600]})
        messages.append({"role": "user", "content": message[:1500]})

        text = await self._generate(messages, max_tokens=600)
        await db.add_chat_message(user_id, "user", message)
        if text:
            await db.add_chat_message(user_id, "assistant", text)

        cap = max_chars or config.CHAT_MAX_CHARS
        return text[:cap] if text else random.choice(_CHAT_FALLBACKS)

    async def comment(self, prompt: str, extra_context: str = "",
                      mood: str = "", max_chars: int = None) -> str:
        """Комментарий в группе/канале: без истории, короче."""
        self._total += 1
        sys_prompt = self._build_system(extra_context, mood)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt[:2000]},
        ]
        text = await self._generate(messages, max_tokens=300)
        cap = max_chars or config.COMMENT_MAX_CHARS
        return text[:cap] if text else random.choice(_COMMENT_FALLBACKS)

    async def vision(self, image_url: str, prompt: str) -> str:
        """Анализ изображения: Groq vision → HF Qwen3-VL → Pollinations vision."""
        self._total += 1
        sys_prompt = self._build_system(vision=True)

        # 1. Groq vision (Llama 3.2 90B Vision) — fastest, free tier
        if config.GROQ_API_KEY:
            text = await self._groq_vision(image_url, prompt, sys_prompt)
            if text:
                self._provider_ok["groq"] = self._provider_ok.get("groq", 0) + 1
                return clean_response(text)

        # 2. HF Qwen3-VL (если credits есть)
        if config.HF_TOKEN:
            text = await self._hf_vision(image_url, prompt, sys_prompt)
            if text:
                self._provider_ok["huggingface"] = self._provider_ok.get("huggingface", 0) + 1
                return clean_response(text)

        # 3. Pollinations vision (free, no auth)
        text = await self._poll_vision(image_url, prompt, sys_prompt)
        if text:
            self._provider_ok["pollinations"] = self._provider_ok.get("pollinations", 0) + 1
            return clean_response(text)

        return ""

    # ── Core generation: linear failover ──

    async def _generate(self, messages: list, max_tokens: int) -> str:
        """Linear failover по приоритету. Первый успех выигрывает."""
        tried = []
        for name, chat_fn, breaker in self._providers:
            if not breaker.ok():
                tried.append(f"{name}(circuit-open)")
                continue
            tried.append(name)
            text, status = await chat_fn(self._http, messages, max_tokens)
            if text:
                breaker.success()
                self._provider_ok[name] = self._provider_ok.get(name, 0) + 1
                logger.info(f"AI: {name} answered ({len(text)} chars)")
                return clean_response(text)
            breaker.fail()
            logger.warning(f"AI: {name} failed (HTTP {status}) — trying next provider")

        # Все провайдеры упали — static fallback
        self._static += 1
        logger.error(f"AI: ALL providers failed — tried: {tried} — static fallback")
        return ""

    # ── Vision implementations ──

    async def _groq_vision(self, image_url: str, prompt: str,
                           system_prompt: str) -> Optional[str]:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt or "Опиши что на картинке."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ]
            resp = await self._http.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={"model": self.GROQ_VISION_MODEL, "messages": messages,
                      "max_tokens": 500, "temperature": 0.4},
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                timeout=30.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            choices = data.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            return text.strip() or None
        except Exception as e:
            logger.debug(f"Groq vision exception: {e}")
            return None

    async def _hf_vision(self, image_url: str, prompt: str,
                         system_prompt: str) -> Optional[str]:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt or "Опиши что на картинке."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ]
            resp = await self._http.post(
                "https://router.huggingface.co/v1/chat/completions",
                json={"model": self.HF_VISION_MODEL, "messages": messages,
                      "max_tokens": 500, "temperature": 0.4},
                headers={"Authorization": f"Bearer {config.HF_TOKEN}"},
                timeout=30.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            choices = data.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            return text.strip() or None
        except Exception as e:
            logger.debug(f"HF vision exception: {e}")
            return None

    async def _poll_vision(self, image_url: str, prompt: str,
                           system_prompt: str) -> Optional[str]:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt or "Опиши что на картинке."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ]
            resp = await self._http.post(
                "https://text.pollinations.ai/openai",
                json={"model": "openai", "messages": messages,
                      "max_tokens": 500, "temperature": 0.5,
                      "referrer": "asluba_bot"},
                timeout=30.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            choices = data.get("choices", [])
            msg = choices[0].get("message", {}) if choices else {}
            text = (msg.get("content", "") if isinstance(msg, dict) else "") or ""
            text = text.strip()
            if not text:
                return None
            low = text.lower()
            if any(k in low for k in ["не вижу", "не могу увидеть", "пришлите её",
                                      "i can't see", "i cannot see", "no image"]):
                return None
            return text
        except Exception as e:
            logger.debug(f"Pollinations vision exception: {e}")
            return None

    # ── Stats ──

    def stats(self) -> dict:
        result = {
            "total": self._total,
            "static": self._static,
            "providers": {},
        }
        for name, _, breaker in self._providers:
            result["providers"][name] = {
                "ok": self._provider_ok.get(name, 0),
                "errors": breaker.errors,
                "circuit_open": breaker.status()["open"],
            }
        return result


# ── Singleton ────────────────────────────────────────────────────────────────

ai = AIClient()
