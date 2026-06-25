"""
Telegram Safe Send — flood-control + length-limit-aware message sending.

Handles:
  - TelegramRetryAfter: waits the requested seconds, then retries.
  - Per-chat rate limiting: max N messages per minute per chat.
  - **4096-char limit**: auto-splits long text into multiple messages,
    sending them sequentially (first as reply, rest as continuation).
  - Graceful fallback: if a reply fails, try answer() instead of reply().

Used by all handlers so Lyuba never triggers Telegram flood bans and never
silently loses a long AI response.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Optional, List

from aiogram import Bot
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest

logger = logging.getLogger("luba.safe_send")

# Telegram's hard limit per message is 4096 chars. We use 3900 as a safe
# margin (accounting for entity markup overhead).
MAX_MESSAGE_CHARS = 3900

# Per-chat message timestamps (for rate limiting)
_chat_send_times: dict = defaultdict(deque)
_rate_lock = asyncio.Lock()


async def _check_rate(chat_id: int, max_per_minute: int) -> bool:
    """Returns True if we can send now (under the rate limit)."""
    async with _rate_lock:
        now = time.time()
        times = _chat_send_times[chat_id]
        while times and (now - times[0]) > 60:
            times.popleft()
        if len(times) >= max_per_minute:
            return False
        times.append(now)
        return True


def _split_text(text: str, max_len: int = MAX_MESSAGE_CHARS) -> List[str]:
    """Split long text into chunks <= max_len, preferring newline boundaries.

    Strategy:
      1. Split on double-newlines (paragraph breaks) first.
      2. If a paragraph is still too long, split on single newlines.
      3. If a line is still too long, hard-split at max_len.
    Returns a list of chunks.
    """
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    # Try paragraph splits first
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_len:
            current = (current + "\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)
                current = ""
            # If the paragraph itself is too long, split by lines
            if len(para) > max_len:
                lines = para.split("\n")
                for line in lines:
                    if len(current) + len(line) + 1 <= max_len:
                        current = (current + "\n" + line) if current else line
                    else:
                        if current:
                            chunks.append(current)
                            current = ""
                        # Hard-split very long single lines
                        while len(line) > max_len:
                            chunks.append(line[:max_len])
                            line = line[max_len:]
                        current = line
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


async def _send_single(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_to_message_id: Optional[int] = None,
    max_retries: int = 2,
) -> bool:
    """Send a single message (must be <= 4096 chars). With retry."""
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
            # "message is too long" → split (shouldn't happen if _split_text works)
            if "too long" in msg.lower():
                logger.warning(f"Message too long even after split — truncating")
                await bot.send_message(chat_id, text[:MAX_MESSAGE_CHARS])
                return True
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


async def safe_send(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_to_message_id: Optional[int] = None,
    max_retries: int = 2,
    priority: bool = False,
) -> bool:
    """Send a message with flood-control + length-limit handling.

    - If text > 3900 chars, auto-splits into multiple messages (sent sequentially).
      First chunk is sent as reply (if reply_to given), rest as continuation.
    - If Telegram says RetryAfter X, waits X seconds and retries.
    - Respects per-chat rate limit (GROUP_MAX_PER_MINUTE).
    - priority=True (directed messages) uses a HIGHER rate-limit cap.

    Returns True if ALL chunks sent successfully, False otherwise.
    """
    from bot.config import config

    if not text or not text.strip():
        return False

    # Per-chat rate limit (only for group chats; private always allowed)
    if str(chat_id).startswith("-"):
        cap = config.GROUP_MAX_PER_MINUTE * 3 if priority else config.GROUP_MAX_PER_MINUTE
        if not await _check_rate(chat_id, cap):
            logger.warning(f"Rate limit hit for chat {chat_id} (cap={cap}) — skipping send")
            return False

    # Split long text into chunks <= 3900 chars
    chunks = _split_text(text)
    if len(chunks) == 1:
        return await _send_single(bot, chat_id, chunks[0], reply_to_message_id, max_retries)

    # Multi-chunk: send first as reply, rest as continuation
    logger.info(f"Message too long ({len(text)} chars) — splitting into {len(chunks)} parts")
    all_sent = True
    first_reply = reply_to_message_id
    for i, chunk in enumerate(chunks):
        sent = await _send_single(bot, chat_id, chunk, first_reply, max_retries)
        if not sent:
            all_sent = False
            logger.warning(f"Chunk {i+1}/{len(chunks)} failed to send")
            break
        first_reply = None  # only first chunk replies to original
        # Small delay between chunks to avoid flood
        if i < len(chunks) - 1:
            await asyncio.sleep(0.3)
    return all_sent


async def safe_reply(
    bot: Bot,
    target_message,
    text: str,
    always_reply: bool = True,
    priority: bool = False,
) -> bool:
    """Reply to a specific message (threaded). Auto-splits if too long.

    If always_reply=True (default), Lyuba's response is threaded under the
    user's message. priority=True for directed messages (higher rate-limit cap).
    """
    reply_to = target_message.message_id if always_reply else None
    return await safe_send(bot, target_message.chat.id, text,
                           reply_to_message_id=reply_to, priority=priority)
