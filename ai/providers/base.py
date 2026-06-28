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

    # Circuit-breaker tuning. The local CPU model is the main beneficiary:
    # when it starts timing out under load, we want to back OFF for a long
    # time (not 30s) so the OS can reclaim the cancelled C-threads and the
    # process doesn't get segfaulted by GGML_ASSERT failures.
    CIRCUIT_ERROR_THRESHOLD: int = 3      # trip after 3 consecutive errors
    CIRCUIT_COOLDOWN_SECONDS: int = 300   # stay open for 5 minutes once tripped

    def __init__(self):
        self._consecutive_errors = 0
        self._circuit_open_until = 0.0  # timestamp when circuit breaker expires

    async def is_available(self) -> bool:
        import time
        # Circuit breaker: if too many consecutive errors, skip this provider
        # for the cooldown window. This prevents the local model from being
        # re-queried in a tight loop while it's still grinding through cancelled
        # C-threads (which is what segfaults the process).
        if self._consecutive_errors >= self.CIRCUIT_ERROR_THRESHOLD:
            if time.time() < self._circuit_open_until:
                return False
            # Cooldown elapsed — give the provider one more chance
            self._consecutive_errors = self.CIRCUIT_ERROR_THRESHOLD - 1
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
        if self._consecutive_errors >= self.CIRCUIT_ERROR_THRESHOLD:
            self._circuit_open_until = time.time() + self.CIRCUIT_COOLDOWN_SECONDS
        return AIResponse(error=True, error_message=msg, provider=self.name)

    def _ok(self, text: str, model: str = "") -> AIResponse:
        self._consecutive_errors = 0
        return AIResponse(text=text, model=model, provider=self.name)
