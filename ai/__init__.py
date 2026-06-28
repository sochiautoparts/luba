"""AI Client v2.0 for Люба Bot.

Модули:
  - client.py  — AIClient (singleton httpx, HF primary, Pollinations backup)
  - persona.py — PERSONA_PROMPT, VISION_PROMPT, THINKING/GREETING/DEFLECTION фразы
  - clean.py   — clean_response (единая точка очистки от утечек промтов)

Использование:
    from ai import ai, persona, clean_response
    text = await ai.chat(user_id, message, extra_context=ctx, mood=mood)
"""

from ai.client import ai, AIClient, CircuitBreaker
from ai.persona import (
    PERSONA_PROMPT, VISION_PROMPT,
    THINKING_PHRASES, GREETING_PHRASES, DEFLECTION_PHRASES,
)
from ai.clean import clean_response, contains_non_cyrillic_script

__all__ = [
    "ai", "AIClient", "CircuitBreaker",
    "PERSONA_PROMPT", "VISION_PROMPT",
    "THINKING_PHRASES", "GREETING_PHRASES", "DEFLECTION_PHRASES",
    "clean_response", "contains_non_cyrillic_script",
]
