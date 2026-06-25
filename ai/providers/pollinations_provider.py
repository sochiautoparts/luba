"""Pollinations AI provider — FREE (no auth required), text + vision.

Pollinations.ai offers (verified June 2025):
  - ONLY valid chat model: `openai` (= `openai-fast`, GPT-OSS 20B by OVH)
    Other names (mistral, deepseek, qwen-coder, searchgpt, etc.) were REMOVED
    from the legacy text API and now return 404 "Model not found".
  - Two endpoints:
      GET  https://text.pollinations.ai/{prompt}?model=openai  ← MORE RELIABLE
      POST https://text.pollinations.ai/openai  (OpenAI-compatible JSON)
    The GET endpoint is simpler and succeeds more often when POST is rate-limited.
  - Vision: same openai model accepts image_url in POST messages.

When API keys are configured, they're sent as Bearer token for better rate limits.
This is the always-available cloud fallback that needs NO credentials.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional
from urllib.parse import quote_plus

import httpx

from ai.providers.base import BaseAIProvider, AIResponse
from bot.config import config

logger = logging.getLogger("luba.ai.pollinations")

# Pollinations models — verified available (June 2025).
# `openai` and `openai-fast` are the same GPT-OSS 20B model.
# Other model names (mistral, deepseek, etc.) MAY return 404 OR 429:
#   - 404 = permanently removed (skip)
#   - 429 = temporarily overloaded (retry later, do NOT disable)
# We keep a broader candidate list and try each; 429s trigger a short sleep+retry.
CHAT_MODELS = ["openai", "openai-fast", "mistral", "deepseek"]
# Models confirmed to return 404 "Model not found" — skip these permanently
_DISABLED_MODELS = {"searchgpt", "roblox", "unity", "evil", "nova", "midijourney"}
VISION_MODEL = "openai"


class PollinationsProvider(BaseAIProvider):
    """Pollinations free API — GET-first, POST-fallback."""

    name = "pollinations"

    def __init__(self, use_key: bool = False):
        super().__init__()
        self.use_key = use_key
        self._base_url = config.POLLINATIONS_BASE_URL
        self._free_url = config.POLLINATIONS_FREE_TEXT_URL

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.use_key:
            key = config.POLLINATIONS_API_KEY or config.POLLINATIONS_API_KEY_2
            if key:
                h["Authorization"] = f"Bearer {key}"
        return h

    async def is_available(self) -> bool:
        return True  # always considered available; failures handled in chat()

    def _build_get_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Flatten messages into a single prompt for the GET endpoint.

        The GET endpoint takes a raw prompt string, so we embed a COMPACT role
        cue + conversation. We do NOT send the full system prompt (it's ~5KB
        and would bloat the URL + slow down generation). Instead, a one-line
        role reminder is enough — the model already knows Lyuba from training
        cues in the conversation.
        """
        parts = []
        sys_added = False
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            if role == "system":
                # Use only the first 400 chars of system prompt as a cue
                # (the full prompt is too long for a GET URL)
                cue = content[:400]
                parts.append(f"Ты Люба — живая девушка из Сочи, общайся живо и коротко. {cue}")
                sys_added = True
            elif role == "assistant":
                parts.append(f"Люба: {content}")
            else:
                parts.append(f"Собеседник: {content}")
        if not sys_added:
            parts.insert(0, "Ты Люба — живая девушка из Сочи, общайся живо и коротко.")
        parts.append("Люба:")  # cue the model to answer as Lyuba
        return "\n".join(parts)

    async def _try_get(self, messages: List[Dict[str, str]], model: str,
                       temperature: float) -> Optional[str]:
        """GET endpoint — returns plain text. More reliable than POST.

        Uses a SHORT timeout (12s) so if it's slow, we fail fast and try POST.
        """
        prompt = self._build_get_prompt(messages)
        params = {"model": model, "private": "true", "temperature": str(temperature)}
        params["referrer"] = config.BOT_USERNAME.lstrip("@")
        url = f"{self._free_url}/{quote_plus(prompt[:3000])}"
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code == 200:
                text = resp.text.strip()
                if text and len(text) > 2:
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    return text
            else:
                logger.debug(f"Pollinations GET {model} -> {resp.status_code}")
        except Exception as e:
            logger.debug(f"Pollinations GET {model} exception: {e}")
        return None

    async def _try_post(self, messages: List[Dict[str, str]], model: str,
                        temperature: float, max_tokens: int) -> Optional[str]:
        """POST endpoint — OpenAI-compatible JSON.

        Handles 429 (queue full / rate-limited) with a short retry, since
        Pollinations models are often temporarily unavailable (not broken).
        404 "Model not found" = permanently removed → no retry.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "private": True,
            "referrer": config.BOT_USERNAME.lstrip("@"),
        }
        urls = []
        if self.use_key:
            urls.append(f"{self._base_url}/openai")
        urls.append(f"{self._free_url}/openai")
        last_err = ""
        for url in urls:
            for attempt in range(2):  # one retry on 429
                try:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        resp = await client.post(url, json=payload, headers=self._headers())
                    if resp.status_code == 200:
                        data = resp.json()
                        text = ""
                        if isinstance(data, dict):
                            choices = data.get("choices", [])
                            if choices:
                                msg = choices[0].get("message", {})
                                text = msg.get("content", "") or choices[0].get("text", "")
                        text = (text or "").strip()
                        if text:
                            return text
                        last_err = "empty response"
                    elif resp.status_code == 429:
                        # Temporarily overloaded — wait and retry once
                        last_err = "429 queue full"
                        if attempt == 0:
                            await asyncio.sleep(2)
                            continue
                    elif resp.status_code == 404:
                        # Model permanently removed — stop trying this model
                        last_err = f"404 model not found: {model}"
                        break
                    else:
                        last_err = f"HTTP {resp.status_code}"
                        logger.debug(f"Pollinations POST {url} {model} -> {resp.status_code}")
                except Exception as e:
                    last_err = str(e)
                    logger.debug(f"Pollinations POST {url} {model} exception: {e}")
        return None

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512,
                   model: str = "") -> AIResponse:
        # Strategy: GET first (more reliable), then POST for each valid model.
        models_to_try = [model] if model else CHAT_MODELS
        for mdl in models_to_try:
            text = await self._try_get(messages, mdl, temperature)
            if text:
                return self._ok(text, model=f"{mdl}-get")
            text = await self._try_post(messages, mdl, temperature, max_tokens)
            if text:
                return self._ok(text, model=f"{mdl}-post")
        return self._fail("pollinations all endpoints failed")

    async def analyze_image(self, image_url: str, prompt: str,
                            system_prompt: str = "") -> AIResponse:
        """Vision via Pollinations POST openai endpoint (model supports image_url)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or "Опиши что на картинке."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        })
        payload = {
            "model": VISION_MODEL,
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.6,
            "referrer": config.BOT_USERNAME.lstrip("@"),
        }
        urls = []
        if self.use_key:
            urls.append(f"{self._base_url}/openai")
        urls.append(f"{self._free_url}/openai")
        last_err = ""
        for url in urls:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, json=payload, headers=self._headers())
                if resp.status_code == 200:
                    data = resp.json()
                    text = ""
                    if isinstance(data, dict):
                        choices = data.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    text = (text or "").strip()
                    if text:
                        return self._ok(text, model=f"{VISION_MODEL}-vision")
                    last_err = "empty vision response"
                else:
                    last_err = f"HTTP {resp.status_code}"
            except Exception as e:
                last_err = str(e)
        return self._fail(last_err or "pollinations vision failed")
