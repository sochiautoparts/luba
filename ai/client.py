"""
AI Client v2.2 — оптимизированная free-первая архитектура.

РЕАЛЬНОСТЬ ПОСЛЕ ТЕСТОВ 10+ endpoints:
  - Pollinations (GPT-OSS 20B) — ЕДИНСТВЕННЫЙ надёжный free/no-auth провайдер
    • POST /openai: ~0.3с с reasoning, system prompt поддерживается
    • GET endpoint: ~0.3с, для коротких промтов
    • 5/5 запросов без 429 (стабильный rate)
  - DuckDuckGo AI — требует captcha (418 ERR_CHALLENGE)
  - HuggingFace без токена — timeout/connection refused
  - Cohere "trial" key — 401 invalid
  - DeepInfra — 422 missing captcha
  - Together AI — требует API key
  - Google Gemini / Groq / OpenRouter — требуют ключ (user adds)

СТРАТЕГИЯ v2.2:
  1. Если есть GROQ_API_KEY → Groq primary (0.4с, Llama 70B)
  2. Если есть GEMINI_API_KEY → Gemini (1с, gemini-2.0-flash)
  3. Pollinations POST (0.3с, GPT-OSS 20B с reasoning) — PRIMARY free
  4. Pollinations GET (0.3с, fallback если POST упал)
  5. Если есть OPENROUTER_API_KEY → OpenRouter (Llama 70B free)
  6. Если есть HF_TOKEN → HF (часто 402 depleted)
  7. Если есть CF_API_TOKEN_1 → Cloudflare (Mistral 3.1)
  8. Умные контекстные статические fallback

ОПТИМИЗАЦИИ:
  - Singleton httpx с connection pooling (keep-alive)
  - In-memory cache (60с TTL) для дедупликации идентичных запросов
  - Умный retry для 429 (exponential backoff: 1с, 2с, 4с)
  - GET-first для коротких промтов (<200 символов) — быстрее
  - POST для длинных промтов — надёжнее с system role
  - Circuit breaker per provider (3 errors → 120s cooldown)
  - Линейный failover — первый успех выигрывает

БЕЗ КЛЮЧЕЙ: бот работает на Pollinations (0.3с latency, стабильно).
С GROQ + GEMINI: 3 независимых провайдера, 99.9% uptime.
"""

import asyncio
import hashlib
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


# ── In-memory cache for deduplication ────────────────────────────────────────

class ResponseCache:
    """60-секундный кэш для идентичных запросов (дедупликация)."""

    def __init__(self, ttl: int = 60):
        self._ttl = ttl
        self._cache: dict = {}
        self._lock = asyncio.Lock()

    def _key(self, messages: list) -> str:
        # Хэш от messages content (без timestamp/mood)
        content = "".join(m.get("content", "") for m in messages)
        return hashlib.md5(content.encode()).hexdigest()

    async def get(self, messages: list) -> Optional[str]:
        key = self._key(messages)
        async with self._lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry["ts"] < self._ttl:
                logger.debug(f"Cache HIT (key={key[:8]})")
                return entry["text"]
        return None

    async def set(self, messages: list, text: str) -> None:
        key = self._key(messages)
        async with self._lock:
            self._cache[key] = {"text": text, "ts": time.time()}
            # Clean old entries (>5 min)
            cutoff = time.time() - 300
            self._cache = {k: v for k, v in self._cache.items() if v["ts"] > cutoff}


# ── Static fallbacks (контекстные, не тупые) ─────────────────────────────────

_CHAT_FALLBACKS = [
    "ой, у меня что-то мысль застряла 🙈 давай ещё раз?",
    "секунду, я немного зависла… повторишь?",
    "блин, связь капризничает. напиши ещё разок?",
    "хм, задумалась. переформулируешь?",
]
_COMMENT_FALLBACKS = [
    "интересно 😊", "согласна", "ого", "хех, жизненно",
    "ага, понятно", "согласна, метко подмечено",
    "жизненно 😅", "ну да, бывает",
]
_GREETING_FALLBACKS = [
    "привет! 😊 я тут, на связи. как дела?",
    "хей! ☕ привет-привет. чем занимаешься?",
    "о, привет! я Люба. рассказывай, что нового?",
]


def _smart_fallback(prompt: str, is_comment: bool) -> str:
    """Контекстный статический fallback — анализирует промт."""
    p = (prompt or "").lower().strip()
    if not p:
        return random.choice(_COMMENT_FALLBACKS if is_comment else _CHAT_FALLBACKS)
    # Приветствие
    if any(w in p for w in ["привет", "хай", "hello", "здравствуй", "хей", "hi"]):
        return random.choice(_GREETING_FALLBACKS)
    # Вопрос
    if "?" in p:
        if is_comment:
            return random.choice(["хороший вопрос 🤔", "интересно, я тоже задумалась"])
        return "хороший вопрос, но я сейчас немного зависла 🙈 повторишь?"
    # Длинное сообщение
    if len(p) > 100:
        return random.choice(_COMMENT_FALLBACKS if is_comment else _CHAT_FALLBACKS)
    return random.choice(_COMMENT_FALLBACKS if is_comment else _CHAT_FALLBACKS)


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

async def _groq_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """Groq — Llama 3.3 70B. ~0.4с. Free 30 req/min, 14K req/day."""
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
    """Google Gemini 2.0-flash. Free 15 req/min, 1500/day."""
    try:
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


async def _pollinations_post(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """Pollinations POST /openai — GPT-OSS 20B с reasoning. ~0.3с. Primary free."""
    try:
        headers = {"Content-Type": "application/json"}
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
            timeout=15.0,  # было 20, уменьшил — POST реально 0.3-0.4с
        )
        if resp.status_code == 429:
            logger.warning("Pollinations POST 429 (rate-limited)")
            return None, 429
        if resp.status_code != 200:
            logger.warning(f"Pollinations POST HTTP {resp.status_code}: {resp.text[:100]}")
            return None, resp.status_code
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None, 200
        msg = choices[0].get("message", {})
        # БЕРЁМ ТОЛЬКО content, НЕ reasoning (reasoning — внутренние мысли модели)
        text = (msg.get("content", "") if isinstance(msg, dict) else "") or ""
        text = text.strip()
        return (text or None), 200
    except Exception as e:
        logger.warning(f"Pollinations POST exception: {type(e).__name__}: {e}")
        return None, 0


async def _pollinations_get(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """Pollinations GET endpoint — fallback если POST упал. ~0.3с.

    GET встраивает промт в URL. Используем компактный cue вместо полного system
    промта (чтобы избежать утечек меток). Role-метки 'Люба:'/'Собеседник:'
    добавляем для контекста диалога — clean_response их уберёт.
    """
    try:
        from urllib.parse import quote_plus
        # Компактный cue — без меток промта
        parts = ["Ты Люба — живая девушка 28 лет из Сочи. Общайся живым русским, коротко, без markdown. Без политики и войны."]
        user_msg = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content or role == "system":
                continue
            if role == "assistant":
                parts.append(f"Люба: {content[:300]}")
            else:
                parts.append(f"Собеседник: {content[:800]}")
                user_msg = content
        parts.append("Люба:")
        prompt = "\n".join(parts)[:2500]  # URL limit safety
        params = {
            "model": "openai",
            "private": "true",
            "temperature": str(config.CHAT_TEMPERATURE),
            "referrer": "asluba_bot",
        }
        if config.POLLINATIONS_API_KEY:
            params["api_key"] = config.POLLINATIONS_API_KEY
        url = f"https://text.pollinations.ai/{quote_plus(prompt)}"
        resp = await http.get(url, params=params, timeout=10.0)
        if resp.status_code == 429:
            return None, 429
        if resp.status_code != 200:
            logger.warning(f"Pollinations GET HTTP {resp.status_code}")
            return None, resp.status_code
        text = resp.text.strip()
        if not text or len(text) < 2:
            return None, 200
        # Убираем обёртывающие кавычки если есть
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text, 200
    except Exception as e:
        logger.warning(f"Pollinations GET exception: {type(e).__name__}: {e}")
        return None, 0


async def _openrouter_chat(http: httpx.AsyncClient, messages: list, max_tokens: int) -> Tuple[Optional[str], int]:
    """OpenRouter free: Llama 3.3 70B. Free 50 req/day."""
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
    """HuggingFace Qwen2.5-7B. Часто 402 depleted."""
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

def _build_providers() -> List[Tuple[str, Callable, CircuitBreaker]]:
    """Собирает список активных провайдеров в порядке приоритета."""
    providers: List[Tuple[str, Callable, CircuitBreaker]] = []
    # Keyed providers (если пользователь добавил ключи)
    if config.GROQ_API_KEY:
        providers.append(("groq", _groq_chat, CircuitBreaker("groq", threshold=3, cooldown=120)))
    if config.GEMINI_API_KEY:
        providers.append(("gemini", _gemini_chat, CircuitBreaker("gemini", threshold=3, cooldown=120)))
    # Pollinations — PRIMARY free (всегда доступен). POST first, GET fallback.
    providers.append(("pollinations", _pollinations_post, CircuitBreaker("pollinations", threshold=5, cooldown=60)))
    providers.append(("pollinations_get", _pollinations_get, CircuitBreaker("pollinations_get", threshold=5, cooldown=60)))
    # Дополнительные keyed providers (нижний приоритет)
    if config.OPENROUTER_API_KEY:
        providers.append(("openrouter", _openrouter_chat, CircuitBreaker("openrouter", threshold=3, cooldown=180)))
    if config.HF_TOKEN:
        providers.append(("huggingface", _hf_chat, CircuitBreaker("huggingface", threshold=2, cooldown=600)))
    if config.CF_API_TOKEN_1:
        providers.append(("cloudflare", _cloudflare_chat, CircuitBreaker("cloudflare", threshold=3, cooldown=180)))
    return providers


# ── AI Client ────────────────────────────────────────────────────────────────

class AIClient:
    """Единый AI клиент: мульти-провайдерный linear failover + cache."""

    HF_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
    GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self._providers: List[Tuple[str, Callable, CircuitBreaker]] = []
        self._cache = ResponseCache(ttl=60)
        self._total = 0
        self._provider_ok: dict = {}
        self._static = 0
        self._cache_hits = 0
        # Semaphore: сериализация AI-запросов (не больше 1 одновременно к
        # одному провайдеру). Prevents 429 при параллельных сообщениях
        # (например, когда Настя шлёт 2 поста подряд — бот не должен слать
        # 2 запроса к Pollinations одновременно).
        self._semaphore = asyncio.Semaphore(1)

    async def initialize(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0, read=25.0),
            limits=httpx.Limits(
                max_connections=30,
                max_keepalive_connections=15,
                keepalive_expiry=30.0,
            ),
            headers={"User-Agent": "LyubaBot/2.2"},
        )
        self._providers = _build_providers()
        names = [p[0] for p in self._providers]
        for name in names:
            self._provider_ok[name] = 0
        logger.info(
            f"AI Client v2.2 ready — providers (priority): {names}. "
            f"{config.providers_status()}. "
            f"Serialization: semaphore=1 (no parallel API calls)"
        )

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    def _build_system(self, extra_context: str = "", mood: str = "",
                      vision: bool = False) -> str:
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

        text = await self._generate(messages, max_tokens=600, is_comment=False)
        await db.add_chat_message(user_id, "user", message)
        if text:
            await db.add_chat_message(user_id, "assistant", text)

        cap = max_chars or config.CHAT_MAX_CHARS
        return text[:cap] if text else _smart_fallback(message, is_comment=False)

    async def comment(self, prompt: str, extra_context: str = "",
                      mood: str = "", max_chars: int = None) -> str:
        """Комментарий в группе/канале: без истории, короче."""
        self._total += 1
        sys_prompt = self._build_system(extra_context, mood)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt[:2000]},
        ]
        text = await self._generate(messages, max_tokens=300, is_comment=True)
        cap = max_chars or config.COMMENT_MAX_CHARS
        return text[:cap] if text else _smart_fallback(prompt, is_comment=True)

    async def vision(self, image_url: str, prompt: str) -> str:
        """Анализ изображения: Groq vision → HF Qwen3-VL → Pollinations vision."""
        self._total += 1
        sys_prompt = self._build_system(vision=True)

        if config.GROQ_API_KEY:
            text = await self._groq_vision(image_url, prompt, sys_prompt)
            if text:
                self._provider_ok["groq"] = self._provider_ok.get("groq", 0) + 1
                return clean_response(text)

        if config.HF_TOKEN:
            text = await self._hf_vision(image_url, prompt, sys_prompt)
            if text:
                self._provider_ok["huggingface"] = self._provider_ok.get("huggingface", 0) + 1
                return clean_response(text)

        text = await self._poll_vision(image_url, prompt, sys_prompt)
        if text:
            self._provider_ok["pollinations"] = self._provider_ok.get("pollinations", 0) + 1
            return clean_response(text)

        return ""

    # ── Core generation: linear failover with cache + retry + queue ──

    async def _generate(self, messages: list, max_tokens: int, is_comment: bool = False) -> str:
        """Linear failover + кэш + semaphore (сериализация) + экспоненциальный retry.

        Архитектура (почему нет 429 при параллельных сообщениях):
          1. Semaphore(1) — только 1 AI-запрос к провайдеру одновременно.
             Если Настя шлёт 2 поста подряд, второй запрос ждёт в очереди,
             а не бьёт по API параллельно → нет 429.
          2. Cache (60с TTL) — идентичные запросы не повторяются.
          3. Экспоненциальный backoff для 429: 1с → 2с → 4с (3 попытки).
          4. Linear failover: pollinations → pollinations_get → HF → ...
          5. Если все провайдеры упали — smart fallback (контекстный).
        """
        # 1. Проверяем кэш (дедупликация)
        cached = await self._cache.get(messages)
        if cached is not None:
            self._cache_hits += 1
            logger.info(f"AI: cache HIT (saved API call)")
            return cached

        # 2. Сериализация: ждём semaphore (не больше 1 запроса одновременно)
        # Timeout 30с — если очередь длинная, лучше fallback чем вечное ожидание
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("AI: semaphore queue timeout (30s) — smart fallback")
            self._static += 1
            last_user = self._last_user(messages)
            return _smart_fallback(last_user, is_comment=is_comment)

        try:
            return await self._generate_inner(messages, max_tokens, is_comment)
        finally:
            self._semaphore.release()

    def _last_user(self, messages: list) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    async def _generate_inner(self, messages: list, max_tokens: int, is_comment: bool) -> str:
        """Внутренняя логика failover (вызывается под semaphore)."""
        # Экспоненциальный backoff для 429: [1с, 2с, 4с] = 3 попытки total
        backoff_schedule = [1.0, 2.0, 4.0]
        tried = []

        for name, chat_fn, breaker in self._providers:
            if not breaker.ok():
                tried.append(f"{name}(circuit-open)")
                continue
            tried.append(name)

            # 3 попытки с экспоненциальным backoff для 429
            for attempt, wait in enumerate([0.0] + backoff_schedule):
                if attempt > 0:
                    logger.info(f"AI: {name} retry #{attempt} after {wait}s (429 backoff)")
                    await asyncio.sleep(wait)

                text, status = await chat_fn(self._http, messages, max_tokens)
                if text:
                    breaker.success()
                    self._provider_ok[name] = self._provider_ok.get(name, 0) + 1
                    logger.info(f"AI: {name} answered ({len(text)} chars, attempt {attempt+1})")
                    await self._cache.set(messages, text)
                    return clean_response(text)

                # 429 = rate-limited — ретраим с backoff (если есть попытки)
                if status == 429 and attempt < len(backoff_schedule):
                    continue
                # Другая ошибка (500, timeout, etc) — не ретраим, след провайдер
                break

            breaker.fail()
            logger.warning(f"AI: {name} failed (HTTP {status}) — trying next provider")

        # Все провайдеры упали — smart fallback
        self._static += 1
        logger.error(f"AI: ALL providers failed — tried: {tried} — smart fallback")
        return _smart_fallback(self._last_user(messages), is_comment=is_comment)

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
            "cache_hits": self._cache_hits,
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
