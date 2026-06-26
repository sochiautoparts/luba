"""AI Router for Люба — LOCAL-FIRST multi-provider failover.

FAILOVER CHAIN:
  Level 0: Local Model (RuadaptQwen3-4B GGUF, CPU) — chat & comment routes
           (privacy-safe: group messages never leave the machine)
  Level 1: Optional keyed providers (Groq → OpenRouter → Gemini → Cloudflare → HF)
           (only if their API keys are configured — best quality free cloud models)
  Level 2: Pollinations FREE API (no auth required — always available)
  Last resort: Static fallback phrases

Route strategy:
  CHAT (private + group-direct)   → Local → optional → Pollinations-free → static
  COMMENT (group proactive + channel) → Local → Pollinations-free → optional → static
  VISION (photos)                 → Pollinations vision (free) → optional(Groq/OpenRouter vision) → static
"""

import asyncio
import hashlib
import logging
import random
import re
import time
from datetime import datetime
from typing import Optional, List, Dict

from ai.providers.base import BaseAIProvider, AIResponse
from ai.providers.local_provider import LocalProvider
from ai.providers.pollinations_provider import PollinationsProvider
from ai.providers.optional_providers import build_optional_providers
from bot.config import config, persona
from bot import database as db

logger = logging.getLogger("luba.ai.router")


# ── Moscow time context (like Asya) ───────────────────────────────────────────

def _moscow_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Moscow"))
    except Exception:
        return datetime.utcnow()


def get_time_context() -> str:
    """Current Moscow time + time-of-day mood descriptor."""
    now = _moscow_now()
    hour = now.hour
    if 5 <= hour < 12:
        tod, mood = "утро", "ты только проснулась, пьёшь кофе, сонная"
    elif 12 <= hour < 18:
        tod, mood = "день", "ты бодрая, в середине рабочего дня"
    elif 18 <= hour < 23:
        tod, mood = "вечер", "ты расслабленная, устала за день"
    else:
        tod, mood = "ночь", "тебе не спится, сова"
    weekday_names = ["понедельник", "вторник", "среда", "четверг",
                     "пятница", "суббота", "воскресенье"]
    wd = weekday_names[now.weekday()]
    season = _season(now.month)
    return (
        f"Сейчас {now.strftime('%d.%m.%Y')} {now.strftime('%H:%M')} по Москве (Europe/Moscow, UTC+3). "
        f"День недели: {wd}. Время суток: {tod}. Сезон: {season}. Настроение: {mood}. "
        f"Ты в Сочи."
    )


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "зима (в Сочи мягкая, +10..+15)"
    if month in (3, 4, 5):
        return "весна"
    if month in (6, 7, 8):
        return "лето (жарко, тёплое море)"
    return "осень"


# ── Response cleaner ──────────────────────────────────────────────────────────

def clean_ai_response(text: str) -> str:
    if not text:
        return ""
    # Strip think tags
    text = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think[^>]*>', '', text, flags=re.IGNORECASE)
    # Strip prefixes
    for prefix in ["Люба:", "Lyuba:", "ЛЮБА:", "Assistant:", "Ответ:", "Ассистент:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Strip quotes
    if len(text) > 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1]
    # Strip markdown
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', '', text)
    # Whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _adaptive_length_hint(prompt: str) -> str:
    """Decide a natural response length band from the prompt.

    Returns a short instruction telling the model how long to be:
      - "короткая реакция"   (1 sentence, ~50-120 chars) — quick comments, emoji-like
      - "средний ответ"      (2-4 sentences, ~150-400 chars) — normal chat
      - "развёрнутый ответ"  (3-6 sentences, ~400-700 chars) — questions, discussions
    The model still chooses; this is a nudge, not a hard cap.
    """
    p = (prompt or "").strip()
    # Question to Lyuba → she can be a bit more elaborate
    if "?" in p and len(p) > 40:
        return "ДЛИНА: развёрнутый ответ (3-6 предложений), задай встречный вопрос если уместно."
    # Very short prompt (greeting, exclamation) → short reaction
    if len(p) < 30 or p.endswith("!") or p.endswith("😄") or p.endswith("👍"):
        return "ДЛИНА: короткая живая реакция (1 предложение)."
    # Long thoughtful message → medium discussion reply
    if len(p) > 200:
        return "ДЛИНА: развёрнутый ответ (3-5 предложений), подхвали тему и задай вопрос."
    # Default
    return "ДЛИНА: средний ответ (2-4 предложения, по ситуации). Часто задай встречный вопрос."


# ── Static fallbacks (when all providers fail) ────────────────────────────────

_CHAT_FALLBACKS = [
    "ой, у меня что-то мысль застряла 🙈 давай ещё раз?",
    "секунду, я немного зависла… повторишь?",
    "блин, связь капризничает. напиши ещё разок?",
]
_COMMENT_FALLBACKS = [
    "интересно 😊",
    "согласна",
    "ого",
    "хех, жизненно",
]


# ── Router ────────────────────────────────────────────────────────────────────

class AIRouter:
    def __init__(self):
        self._local = LocalProvider()
        self._pollinations_key = PollinationsProvider(use_key=True) if config.has_pollinations_key() else None
        self._pollinations_free = PollinationsProvider(use_key=False)
        self._optional: List[BaseAIProvider] = build_optional_providers()
        self._total = 0
        self._level0 = 0
        self._level_cloud = 0
        self._level_static = 0

    async def initialize(self) -> None:
        await self._local.initialize()
        logger.info(
            f"AI router ready — local={'ON' if self._local._model_loaded else 'OFF'}, "
            f"optional={[p.name for p in self._optional]}, pollinations_free=ON"
        )

    async def close(self) -> None:
        await self._local.close()

    def _build_system(self, base: str = "", extra_context: str = "", mood: str = "") -> str:
        sys_prompt = base or persona.system_prompt
        sys_prompt += "\n\n" + get_time_context()
        if mood:
            sys_prompt += f"\nТекущее настроение Любы: {mood}."
        if extra_context:
            sys_prompt += "\n\n" + extra_context
        return sys_prompt

    # ── Core chat ──

    async def chat(
        self,
        user_id: int,
        message: str,
        system_prompt: str = "",
        extra_context: str = "",
        route_type: str = "chat",
        mood: str = "",
        max_chars: int = None,
    ) -> AIResponse:
        sys_prompt = self._build_system(system_prompt, extra_context, mood)
        # Adaptive length hint for private chats too
        sys_prompt += "\n\n" + _adaptive_length_hint(message)
        history = await db.get_chat_history(user_id, limit=8)

        messages = [{"role": "system", "content": sys_prompt}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"][:600]})
        messages.append({"role": "user", "content": message[:1500]})

        response = await self._route(messages, route_type)

        # Save history
        await db.add_chat_message(user_id, "user", message)
        if response.ok:
            await db.add_chat_message(user_id, "assistant", response.text)

        # Truncate to char limit
        if response.ok and max_chars:
            response.text = response.text[:max_chars]
        response.text = clean_ai_response(response.text or "")
        return response

    async def comment(
        self,
        prompt: str,
        extra_context: str = "",
        mood: str = "",
        route_type: str = "comment",
        max_chars: int = None,
        max_tokens: int = None,
    ) -> AIResponse:
        """Generate a short comment (group proactive / channel post comment).

        Response length is ADAPTIVE: the system prompt asks the model to vary
        length by context (1 sentence for quick reactions, 2-3 for discussions).
        We pass a length-band hint derived from the prompt, and only hard-truncate
        at the end as a safety cap.
        """
        base = persona.local_system_prompt if route_type == "comment" else persona.system_prompt
        sys_prompt = self._build_system(base, extra_context, mood)

        # Adaptive length hint: short prompts → short replies; questions/long
        # messages → can be a bit longer. This makes Lyuba feel natural, not
        # uniformly terse.
        adaptive_hint = _adaptive_length_hint(prompt)

        messages = [
            {"role": "system", "content": sys_prompt + "\n\n" + adaptive_hint},
            {"role": "user", "content": prompt[:2000]},
        ]
        tokens = max_tokens or config.COMMENT_MAX_TOKENS
        response = await self._route(messages, route_type, max_tokens=tokens)
        response.text = clean_ai_response(response.text or "")
        cap = max_chars or config.COMMENT_MAX_CHARS
        if response.ok:
            response.text = response.text[:cap]
        return response

    async def vision(
        self,
        image_url: str,
        prompt: str,
        system_prompt: str = "",
    ) -> AIResponse:
        """Analyze an image.

        Tries vision-capable providers in priority order:
          1. Gemini (free, native vision — most reliable)
          2. Groq (free, llama-vision)
          3. Pollinations free (no key, but often rate-limited for vision)
        Detects "can't see the image" refusals and falls through to the next.
        """
        sys_prompt = self._build_system(system_prompt or (persona.system_prompt + persona.vision_prompt_suffix))

        # Build ordered list of vision-capable providers
        vision_providers = []
        for p in self._optional:
            if p.name in ("gemini", "groq", "huggingface"):
                vision_providers.append(p)
        # Pollinations free (always available, but vision often rate-limited)
        vision_providers.append(self._pollinations_free)
        if self._pollinations_key:
            vision_providers.append(self._pollinations_key)

        for p in vision_providers:
            try:
                resp = await p.analyze_image(image_url, prompt, sys_prompt)
            except Exception as e:
                logger.debug(f"{p.name} vision exception: {e}")
                continue
            if not resp.ok:
                logger.debug(f"{p.name} vision failed: {resp.error_message}")
                continue
            text = resp.text.strip()
            # Detect "can't see" refusals (free Pollinations often does this)
            low = text.lower()
            if any(k in low for k in ["не вижу", "не вижу картинку", "не могу увидеть",
                                      "пришлите её", "не вижу прикреплён",
                                      "i can't see", "i cannot see", "no image"]):
                logger.debug(f"{p.name} vision returned a 'can't see' refusal — trying next")
                continue
            resp.text = clean_ai_response(text)
            return resp

        return AIResponse(error=True, error_message="vision failed on all providers", provider="vision")

    async def _route(self, messages: List[Dict[str, str]], route_type: str,
                     max_tokens: int = None) -> AIResponse:
        """Run the failover chain. For chat/comment, LOCAL-FIRST then concurrent cloud."""
        self._total += 1
        max_tokens = max_tokens or config.CHAT_MAX_TOKENS
        temp = config.CHAT_TEMPERATURE

        # ── Level 0: Local model (chat & comment) — with timeout ──
        # Local model on CPU can be slow (30-60s for long responses).
        # We give it a short timeout (15s) and limited tokens (256) to ensure
        # fast responses. If it doesn't finish, we immediately fall through
        # to cloud providers. This prevents the "Local generation cancelled"
        # cascade that blocks subsequent requests.
        if route_type in ("chat", "comment") and config.ENABLE_LOCAL_MODEL:
            if await self._local.is_available():
                # Use fewer tokens for local model (faster generation on CPU)
                local_max_tokens = min(max_tokens, 300)
                try:
                    resp = await asyncio.wait_for(
                        self._local.chat(messages, temperature=temp, max_tokens=local_max_tokens),
                        timeout=15.0  # 15s max for local model
                    )
                    if resp.ok:
                        self._level0 += 1
                        return resp
                    logger.debug(f"Local failed ({route_type}): {resp.error_message}")
                except asyncio.TimeoutError:
                    logger.warning(f"Local model timed out (15s) for route={route_type} — falling through to cloud")
                    # Don't wait for the cancelled C-thread — let it finish in background
                    # while we use cloud providers for this request.

        # ── Level 1+2: Concurrent cloud providers ──
        # Wait for the first SUCCESSFUL response (not just first completion,
        # since the first to finish might be an error). Loop until a success
        # arrives or all tasks are exhausted / timeout.
        cloud_providers = self._cloud_chain(route_type)
        if cloud_providers:
            tasks = [asyncio.create_task(self._safe_call(p, messages, temp, max_tokens))
                     for p in cloud_providers]
            deadline = asyncio.get_running_loop().time() + 30.0
            done = set()
            pending = set(tasks)
            best = None
            while pending:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    finished, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED, timeout=remaining
                    )
                except Exception as e:
                    logger.error(f"asyncio.wait error: {e}")
                    break
                done |= finished
                for t in finished:
                    try:
                        r = t.result()
                    except Exception:
                        r = None
                    if r and r.ok:
                        best = r
                        break
                if best:
                    break
            # Cancel any still-pending tasks
            for t in pending:
                t.cancel()
            if best:
                self._level_cloud += 1
                return best
            # No success — but maybe a non-error-but-empty? Return any finished as last resort
            for t in done:
                try:
                    r = t.result()
                    if r and r.text:
                        self._level_cloud += 1
                        return r
                except Exception:
                    pass

        # ── Last resort: static fallback ──
        self._level_static += 1
        fb = random.choice(_CHAT_FALLBACKS if route_type == "chat" else _COMMENT_FALLBACKS)
        return AIResponse(text=fb, provider="static", model="fallback")

    def _cloud_chain(self, route_type: str) -> List[BaseAIProvider]:
        """Order of cloud providers to try concurrently.

        Both routes now use the same priority: optional keyed providers first
        (HuggingFace Qwen2.5-7B is fast + reliable), then Pollinations free.
        The concurrent fan-out means the first SUCCESS wins.
        """
        chain: List[BaseAIProvider] = []
        # Optional keyed providers (HuggingFace, Groq, etc.) — better quality + faster
        chain.extend(self._optional)
        # Pollinations as fallback (always available, no key needed)
        if self._pollinations_key:
            chain.append(self._pollinations_key)
        chain.append(self._pollinations_free)
        return chain

    async def _safe_call(self, provider: BaseAIProvider, messages: List[Dict[str, str]],
                         temp: float, max_tokens: int) -> AIResponse:
        try:
            return await provider.chat(messages, temperature=temp, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"{provider.name} exception: {e}")
            return AIResponse(error=True, error_message=str(e), provider=provider.name)

    def stats(self) -> Dict[str, int]:
        return {
            "total": self._total,
            "local": self._level0,
            "cloud": self._level_cloud,
            "static": self._level_static,
        }


ai_router = AIRouter()
