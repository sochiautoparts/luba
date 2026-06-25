"""Pollinations AI provider — FREE (no auth required), text + vision.

Pollinations.ai offers:
  - Text generation: https://text.pollinations.ai/openai (OpenAI-compatible, no key needed)
  - Image understanding (vision): send image_url in messages
  - Image generation: https://image.pollinations.ai/prompt/<text>

When API keys are configured, they're sent as Bearer token for better rate limits.
This is the always-available cloud fallback that needs NO credentials.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional

import httpx

from ai.providers.base import BaseAIProvider, AIResponse
from bot.config import config

logger = logging.getLogger("luba.ai.pollinations")

# Models — Pollinations exposes many via the openai-compatible endpoint.
# We prefer strong free chat models. Order = preference.
CHAT_MODELS = ["openai", "mistral", "qwen-coder", "deepseek"]
VISION_MODEL = "openai"  # vision-capable


class PollinationsProvider(BaseAIProvider):
    """Pollinations free API (OpenAI-compatible endpoint)."""

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
        # Always considered available (free endpoint). Actual failures handled in chat().
        return True

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512,
                   model: str = "") -> AIResponse:
        model = model or CHAT_MODELS[0]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "private": True,
        }
        # Try authenticated endpoint first if key, else free endpoint
        urls = []
        if self.use_key:
            urls.append(f"{self._base_url}/openai")
        urls.append(f"{self._free_url}/openai")

        last_err = ""
        for url in urls:
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
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
                        return self._ok(text, model=model)
                    last_err = "empty response"
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
                    logger.debug(f"Pollinations {url} -> {resp.status_code}")
            except Exception as e:
                last_err = str(e)
                logger.debug(f"Pollinations {url} exception: {e}")

        return self._fail(last_err or "pollinations failed")

    async def analyze_image(self, image_url: str, prompt: str,
                            system_prompt: str = "") -> AIResponse:
        """Vision via Pollinations openai-compatible endpoint (model supports image_url)."""
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
        }
        urls = []
        if self.use_key:
            urls.append(f"{self._base_url}/openai")
        urls.append(f"{self._free_url}/openai")

        last_err = ""
        for url in urls:
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
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
