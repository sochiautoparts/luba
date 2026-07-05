"""
Channel handler — sets 3 POSITIVE emoji reactions on ALL channel posts.

Per user request: bot puts 3 positive reactions (👍❤️🔥 etc.) on every post.
NO comments — channels stay clean, bot is a silent engaged subscriber.
"""

import logging
import random

from aiogram import Router, F
from aiogram.types import Message, Chat

from bot.config import config
from bot import database as db
from bot.reactions import maybe_react

logger = logging.getLogger("luba.channels")

channel_router = Router()


def _is_politics_or_war(text: str) -> bool:
    t = (text or "").lower()
    triggers = ["путин", "кремль", "госдума", "санкци", "сво", "мобилиз", "война",
                "зеленск", "байден", "трамп", "выборы", "парламент", "ракетн", "обстрел"]
    return any(w in t for w in triggers)


@channel_router.channel_post(F.text | F.photo | F.video | F.animation | F.sticker | F.voice | F.document | F.video_note)
async def handle_channel_post(message: Message):
    """React to channel posts with 3 POSITIVE emojis — NO comments.

    3 reactions per post (👍❤️🔥 / 😄🎉👏 / etc.) — visually engaging.
    Handles all common post types: text, photo, video, animation, sticker,
    voice, document, video_note.
    """
    chat: Chat = message.chat
    await db.upsert_channel(chat.id, username=chat.username or "", title=chat.title or "")
    logger.info(f"CHANNEL POST received: chat={chat.id} (@{chat.username or ''} / '{chat.title or ''}') msg={message.message_id}")

    if not await db.is_channel_enabled(chat.id):
        logger.warning(f"  channel {chat.id} DISABLED in DB — skip reactions")
        return

    # Always react (probability check removed — user wants 3 reactions on EVERY post)
    post_text = (message.caption or message.text or "").strip()
    if _is_politics_or_war(post_text):
        logger.info(f"  skip: politics/war detected")
        return  # skip politics/war posts

    already = await db.already_reacted(chat.id, message.message_id)
    if already:
        logger.info(f"  already reacted to msg {message.message_id} — skip")
        return

    try:
        ok = await maybe_react(
            message.bot, chat.id, message.message_id, post_text,
            prob=1.0, force=True,
            count=3,  # 3 positive reactions per post
        )
        logger.info(f"  maybe_react result: {'OK (3 reactions set)' if ok else 'FAILED (see warnings above)'}")
    except Exception as e:
        logger.warning(f"channel reaction failed: {e}")


@channel_router.channel_post()
async def handle_channel_post_catchall(message: Message):
    """Catch-all for any other channel post type (polls, dice, etc.)."""
    chat: Chat = message.chat
    await db.upsert_channel(chat.id, username=chat.username or "", title=chat.title or "")
    logger.info(f"CHANNEL POST (catch-all) received: chat={chat.id} (@{chat.username or ''}) msg={message.message_id}")

    if not await db.is_channel_enabled(chat.id):
        logger.warning(f"  channel {chat.id} DISABLED in DB — skip reactions")
        return

    already = await db.already_reacted(chat.id, message.message_id)
    if already:
        logger.info(f"  already reacted to msg {message.message_id} — skip")
        return

    try:
        ok = await maybe_react(
            message.bot, chat.id, message.message_id, "",
            prob=1.0, force=True,
            count=3,
        )
        logger.info(f"  maybe_react result (catch-all): {'OK' if ok else 'FAILED'}")
    except Exception as e:
        logger.warning(f"channel catch-all reaction failed: {e}")
