"""Optional free cloud providers — auto-enabled when API keys are configured.

Each provider is a small adapter. The router only instantiates the ones whose
credentials are present in the environment, so out-of-the-box (no keys) the
bot runs on Local + Pollinations-free. Adding any key below immediately
upgrades routing to include that provider.

Providers:
  - GitHub Models (gpt-4o, gpt-4o-mini, Llama-3.1-405B/8B) — FREE, needs PAT with 'models' scope
  - Groq        (free tier, very fast, Llama 3.3 70B / Llama 3.1 8B)  — OpenAI-compatible
  - OpenRouter  (free models: Llama, Mistral, Gemini-flash, Qwen)     — OpenAI-compatible
  - Gemini      (Google Generative AI free tier)                       — Google REST
  - Cloudflare  (Workers AI, 10K req/day free per account)            — CF REST
  - HuggingFace (Inference API, free tier)                            — HF REST
"""

import asyncio
import base64
import logging
from typing import List, Dict

import httpx

from ai.providers.base import BaseAIProvider, AIResponse
from bot.config import config

logger = logging.getLogger("luba.ai.optional")


# ── OpenAI-compatible base (Groq, OpenRouter) ─────────────────────────────────

class _OpenAICompatProvider(BaseAIProvider):
    """Shared logic for OpenAI-compatible chat completion endpoints."""

    base_url: str = ""
    default_model: str = ""
    extra_headers: Dict[str, str] = {}

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        h.update(self.extra_headers)
        return h

    async def is_available(self) -> bool:
        return True  # instantiated only when key present

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512,
                   model: str = "") -> AIResponse:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(f"{self.base_url}/chat/completions",
                                         json=payload, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                text = choices[0]["message"]["content"] if choices else ""
                text = (text or "").strip()
                if text:
                    return self._ok(text, model=payload["model"])
                return self._fail("empty response")
            return self._fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            return self._fail(str(e))


class GroqProvider(_OpenAICompatProvider):
    """Groq — free tier, very fast Llama models. Also supports vision."""
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
    default_model = "llama-3.3-70b-versatile"
    VISION_MODEL = "llama-3.2-90b-vision-preview"

    def __init__(self):
        super().__init__()
        self._api_key = config.GROQ_API_KEY

    async def analyze_image(self, image_url: str, prompt: str,
                            system_prompt: str = "") -> AIResponse:
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
        payload = {"model": self.VISION_MODEL, "messages": messages,
                   "max_tokens": 600, "temperature": 0.6}
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(f"{self.base_url}/chat/completions",
                                         json=payload, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                text = choices[0]["message"]["content"] if choices else ""
                text = (text or "").strip()
                if text:
                    return self._ok(text, model=self.VISION_MODEL)
            return self._fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            return self._fail(str(e))


class OpenRouterProvider(_OpenAICompatProvider):
    """OpenRouter — access free models (Llama, Mistral, Qwen, Gemini-flash)."""
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    default_model = "meta-llama/llama-3.3-70b-instruct:free"
    extra_headers = {
        "HTTP-Referer": "https://sochiautoparts.ru",
        "X-Title": "Lyuba Bot",
    }

    def __init__(self):
        super().__init__()
        self._api_key = config.OPENROUTER_API_KEY


# ── Google Gemini (free tier) ─────────────────────────────────────────────────

class GeminiProvider(BaseAIProvider):
    """Google Generative AI — free tier via REST. Supports vision natively."""
    name = "gemini"
    MODEL = "gemini-1.5-flash"

    def __init__(self):
        super().__init__()
        self._api_key = config.GEMINI_API_KEY

    async def is_available(self) -> bool:
        return True

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512,
                   model: str = "") -> AIResponse:
        # Convert OpenAI-style messages to Gemini contents
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
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if sys_text:
            payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model or self.MODEL}:generateContent?key={self._api_key}")
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(url, json=payload,
                                         headers={"Content-Type": "application/json"})
            if resp.status_code == 200:
                data = resp.json()
                cands = data.get("candidates", [])
                text = ""
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                text = (text or "").strip()
                if text:
                    return self._ok(text, model=model or self.MODEL)
                return self._fail("empty response")
            return self._fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            return self._fail(str(e))

    async def analyze_image(self, image_url: str, prompt: str,
                            system_prompt: str = "") -> AIResponse:
        """Gemini vision via inline_data (accepts data URIs and http URLs)."""
        import re as _re
        # Parse data URI or use URL directly
        m = _re.match(r"data:([^;]+);base64,(.+)", image_url, _re.DOTALL)
        if m:
            mime, b64 = m.group(1), m.group(2)
            inline = {"inlineData": {"mimeType": mime, "data": b64}}
        else:
            # Remote URL — Gemini file API needs upload; fallback to rejecting
            return self._fail("gemini vision requires data URI (got URL)")
        parts = [{"text": prompt or "Опиши что на картинке."}, inline]
        payload = {"contents": [{"role": "user", "parts": parts}]}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.MODEL}:generateContent?key={self._api_key}")
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(url, json=payload,
                                         headers={"Content-Type": "application/json"})
            if resp.status_code == 200:
                data = resp.json()
                cands = data.get("candidates", [])
                text = ""
                if cands:
                    p = cands[0].get("content", {}).get("parts", [])
                    text = "".join(x.get("text", "") for x in p)
                text = (text or "").strip()
                if text:
                    return self._ok(text, model=self.MODEL)
                return self._fail("empty vision response")
            return self._fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            return self._fail(str(e))


# ── Cloudflare Workers AI ─────────────────────────────────────────────────────

class CloudflareProvider(BaseAIProvider):
    """Cloudflare Workers AI — Mistral Small 3.1 (free 10K req/day per account)."""
    name = "cloudflare"
    MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct"

    def __init__(self, account_id: str = "", api_token: str = ""):
        super().__init__()
        self._account_id = account_id or config.CF_ACCOUNT_ID_1
        self._api_token = api_token or config.CF_API_TOKEN_1

    async def is_available(self) -> bool:
        return True

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512,
                   model: str = "") -> AIResponse:
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{self._account_id}/ai/run/{model or self.MODEL}")
        payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(url, json=payload,
                                         headers={"Authorization": f"Bearer {self._api_token}"})
            if resp.status_code == 200:
                data = resp.json()
                text = ""
                if data.get("success"):
                    res = data.get("result", {})
                    text = res.get("response", "") or ""
                    if not text and isinstance(res.get("choices"), list) and res["choices"]:
                        text = res["choices"][0].get("message", {}).get("content", "")
                text = (text or "").strip()
                if text:
                    return self._ok(text, model=model or self.MODEL)
                return self._fail(f"cf no text: {str(data)[:200]}")
            return self._fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            return self._fail(str(e))


# ── HuggingFace Inference API ─────────────────────────────────────────────────

class HuggingFaceProvider(BaseAIProvider):
    """HuggingFace Inference API — chat completion (free tier)."""
    name = "huggingface"
    MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self):
        super().__init__()
        self._api_key = config.HF_TOKEN

    async def is_available(self) -> bool:
        return True

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512,
                   model: str = "") -> AIResponse:
        # Use the OpenAI-compatible router on HF
        url = "https://router.huggingface.co/v1/chat/completions"
        payload = {
            "model": model or self.MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload,
                                         headers={"Authorization": f"Bearer {self._api_key}"})
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                text = choices[0]["message"]["content"] if choices else ""
                text = (text or "").strip()
                if text:
                    return self._ok(text, model=model or self.MODEL)
                return self._fail("empty response")
            return self._fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            return self._fail(str(e))


# ── Factory ───────────────────────────────────────────────────────────────────

def build_optional_providers() -> List[BaseAIProvider]:
    """Instantiate all optional providers whose credentials are configured.

    Order = priority: HuggingFace first (works, 0.8s, Qwen2.5-7B), then Groq
    (fastest), then the rest.
    """
    providers: List[BaseAIProvider] = []
    if config.HF_TOKEN:
        providers.append(HuggingFaceProvider())
        logger.info("Optional provider enabled: HuggingFace (Qwen2.5-7B)")
    if config.GROQ_API_KEY:
        providers.append(GroqProvider())
        logger.info("Optional provider enabled: Groq")
    if config.OPENROUTER_API_KEY:
        providers.append(OpenRouterProvider())
        logger.info("Optional provider enabled: OpenRouter")
    if config.GEMINI_API_KEY:
        providers.append(GeminiProvider())
        logger.info("Optional provider enabled: Gemini")
    if config.CF_API_TOKEN_1:
        providers.append(CloudflareProvider(config.CF_ACCOUNT_ID_1, config.CF_API_TOKEN_1))
        logger.info("Optional provider enabled: Cloudflare #1")
    if config.CF_API_TOKEN_2:
        providers.append(CloudflareProvider(config.CF_ACCOUNT_ID_2, config.CF_API_TOKEN_2))
        logger.info("Optional provider enabled: Cloudflare #2")
    return providers
