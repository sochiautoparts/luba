"""
Telegram Safe Send — flood-control-aware message sending.

Handles:
  - TelegramRetryAfter: waits the requested seconds, then retries.
  - Per-chat rate limiting: max N messages per minute per chat.
  - Graceful fallback: if a reply fails, try answer() instead of reply().

Used by all handlers so Lyuba never triggers Telegram flood bans.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Optional

from aiogram import Bot
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest

logger = logging.getLogger("luba.safe_send")

# Per-chat message timestamps (for rate limiting)
_chat_send_times: dict = defaultdict(deque)
_rate_lock = asyncio.Lock()


async def _check_rate(chat_id: int, max_per_minute: int) -> bool:
    """Returns True if we can send now (under the rate limit)."""
    async with _rate_lock:
        now = time.time()
        times = _chat_send_times[chat_id]
        # drop entries older than 60s
        while times and (now - times[0]) > 60:
            times.popleft()
        if len(times) >= max_per_minute:
            return False
        times.append(now)
        return True


async def safe_send(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_to_message_id: Optional[int] = None,
    max_retries: int = 2,
) -> bool:
    """Send a message with flood-control handling.

    - If Telegram says RetryAfter X, waits X seconds and retries.
    - Respects per-chat rate limit (GROUP_MAX_PER_MINUTE).
    - Falls back from reply → answer on bad-request errors.

    Returns True if sent, False otherwise.
    """
    from bot.config import config

    # Per-chat rate limit (only for group chats; private always allowed)
    if str(chat_id).startswith("-"):
        if not await _check_rate(chat_id, config.GROUP_MAX_PER_MINUTE):
            logger.warning(f"Rate limit hit for chat {chat_id} — skipping send (flood safety)")
            return False

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            if reply_to_message_id:
                await bot.send_message(chat_id, text, reply_to_message_id=reply_to_message_id)
            else:
                await bot.send_message(chat_id, text)
            return True
        except TelegramRetryAfter as e:
            wait = getattr(e, "retry_after", 5) or 5
            logger.warning(f"Flood control: waiting {wait}s (chat {chat_id})")
            if attempt < max_retries:
                await asyncio.sleep(wait + 1)
                continue
            last_err = e
        except TelegramBadRequest as e:
            msg = str(e)
            # "message to reply not found" → fall back to plain answer
            if "reply" in msg.lower() and reply_to_message_id:
                logger.debug(f"reply failed ({msg[:80]}) — falling back to answer")
                reply_to_message_id = None
                continue
            # "chat not found" / "bot was blocked" → give up
            if "blocked" in msg.lower() or "not found" in msg.lower():
                return False
            last_err = e
        except Exception as e:
            last_err = e
            logger.debug(f"send error (attempt {attempt}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(1)

    if last_err:
        logger.warning(f"safe_send failed after {max_retries+1} attempts: {last_err}")
    return False


async def safe_reply(
    bot: Bot,
    target_message,
    text: str,
    always_reply: bool = True,
) -> bool:
    """Reply to a specific message (threaded). Falls back to answer.

    If always_reply=True (default), Lyuba's response is threaded under the
    user's message — so it's clear WHO she's replying to. This is the
    recommended Telegram pattern for group conversations.
    """
    reply_to = target_message.message_id if always_reply else None
    return await safe_send(bot, target_message.chat.id, text, reply_to_message_id=reply_to)
