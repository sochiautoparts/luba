"""Base AI provider interface and response model."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger("luba.ai.base")


@dataclass
class AIResponse:
    text: str = ""
    model: str = ""
    provider: str = ""
    cached: bool = False
    error: bool = False
    error_message: str = ""
    usage_tokens: int = 0

    @property
    def ok(self) -> bool:
        return (not self.error) and bool(self.text and self.text.strip())


class BaseAIProvider:
    """Abstract AI provider."""

    name: str = "base"

    def __init__(self):
        self._consecutive_errors = 0
        self._circuit_open_until = 0.0  # timestamp when circuit breaker expires

    async def is_available(self) -> bool:
        import time
        # Circuit breaker: if too many errors, skip provider for 30s
        if self._consecutive_errors >= 5:
            if time.time() < self._circuit_open_until:
                return False
            # Reset after cooldown
            self._consecutive_errors = 0
        return True

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.8, max_tokens: int = 512) -> AIResponse:
        raise NotImplementedError

    async def analyze_image(self, image_url: str, prompt: str,
                            system_prompt: str = "") -> AIResponse:
        """Vision: describe/analyze an image. Default: not supported."""
        return AIResponse(error=True, error_message="vision not supported", provider=self.name)

    def _fail(self, msg: str) -> AIResponse:
        import time
        self._consecutive_errors += 1
        if self._consecutive_errors >= 5:
            self._circuit_open_until = time.time() + 30  # 30s cooldown
        return AIResponse(error=True, error_message=msg, provider=self.name)

    def _ok(self, text: str, model: str = "") -> AIResponse:
        self._consecutive_errors = 0
        return AIResponse(text=text, model=model, provider=self.name)
