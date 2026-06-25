"""
Channel handler for Lyuba — ONLY sets reactions on channel posts.

Per user requirement: when Lyuba is added to a channel, she should ONLY put
emoji reactions (likes) on posts, NOT write comments directly in the channel.
This keeps channels clean — the bot is a silent engaged subscriber that
reacts but doesn't flood with comments.

Reactions use Telegram setMessageReaction (👍❤️🔥😄😮🙏 etc.).
The bot must be added as a channel admin (with at least basic rights) for
reactions to work on posts.
"""

import asyncio
import logging
import random
import time

from aiogram import Router, F, types
from aiogram.types import Message, Chat

from bot.config import config
from bot import database as db

logger = logging.getLogger("luba.channels")

channel_router = Router()

# Probability Lyuba reacts to a channel post (0.0-1.0)
# Uses config.CHANNEL_REACTION_PROB (default 0.5) — single source of truth.


def _is_politics_or_war(text: str) -> bool:
    t = (text or "").lower()
    triggers = ["путин", "кремль", "госдума", "санкци", "сво", "мобилиз", "война",
                "зеленск", "байден", "трамп", "выборы", "парламент", "ракетн", "обстрел"]
    return any(w in t for w in triggers)


@channel_router.channel_post(F.text | F.photo | F.video | F.animation)
async def handle_channel_post(message: Message):
    """React to channel posts with emoji — NO comments, NO replies.

    Lyuba is a silent engaged subscriber in channels: she only puts likes
    (reactions), never writes comments. This keeps the channel clean.
    """
    chat: Chat = message.chat
    await db.upsert_channel(chat.id, username=chat.username or "", title=chat.title or "")

    # Respect enabled flag
    if not await db.is_channel_enabled(chat.id):
        return

    # Probabilistic reaction (don't react to EVERY post — feels more natural)
    if random.random() > config.CHANNEL_REACTION_PROB:
        return

    post_text = (message.caption or message.text or "").strip()
    if _is_politics_or_war(post_text):
        return  # skip politics/war posts entirely

    # React to the channel post (like) — this is the ONLY action in channels
    try:
        from bot.reactions import maybe_react
        await maybe_react(
            message.bot, chat.id, message.message_id, post_text,
            prob=1.0,  # we already checked probability above
        )
    except Exception as e:
        logger.debug(f"channel reaction failed: {e}")

    # NO comment reply — channels are reaction-only per design.
    # (Comments happen in the linked discussion group, not the channel itself.)
