"""
Channel handler for Lyuba — comments on channel posts where she's an admin.

When Lyuba is added as an admin (with post/comment rights) to a channel,
she receives channel_post updates. With some probability she writes a short
comment (reply) under the post — based on the post TEXT (vision disabled for
resource efficiency). She also sets emoji reactions (likes) on posts.

Unlike Asya, Lyuba does NOT manage/publish to channels — she only comments
and reacts as an active subscriber.
"""

import asyncio
import logging
import random
import time

from aiogram import Router, F, types
from aiogram.types import Message, Chat
from aiogram.enums import ChatAction

from bot.config import config, persona
from bot import database as db
from bot.context import build_channel_context
from bot.mood import current_mood_descriptor
from ai.router import ai_router

logger = logging.getLogger("luba.channels")

channel_router = Router()

# Probability Lyuba comments on a given channel post
CHANNEL_COMMENT_PROB = 0.4
# Min seconds between comments on the same channel
CHANNEL_MIN_INTERVAL = 60


def _is_politics_or_war(text: str) -> bool:
    t = (text or "").lower()
    triggers = ["путин", "кремль", "госдума", "санкци", "сво", "мобилиз", "война",
                "зеленск", "байден", "трамп", "выборы", "парламент", "ракетн", "обстрел"]
    return any(w in t for w in triggers)


@channel_router.channel_post(F.text | F.photo)
async def handle_channel_post(message: Message):
    chat: Chat = message.chat
    await db.upsert_channel(chat.id, username=chat.username or "", title=chat.title or "")

    # Respect enabled flag
    if not await db.is_channel_enabled(chat.id):
        return

    # Min interval between comments on the same channel
    last = await db.get_channel_last_commented(chat.id)
    if (time.time() - last) < CHANNEL_MIN_INTERVAL:
        return

    # Probabilistic comment
    if random.random() > CHANNEL_COMMENT_PROB:
        return

    post_text = (message.caption or message.text or "").strip()
    if _is_politics_or_war(post_text):
        return

    # Skip albums (media groups) to avoid commenting on every photo
    if message.media_group_id:
        return

    await message.bot.send_chat_action(chat.id, ChatAction.TYPING)
    mood = await current_mood_descriptor()

    # React to the channel post (like) — makes Lyuba feel like an engaged subscriber
    try:
        from bot.reactions import react_to_channel_post
        await react_to_channel_post(message.bot, chat.id, message.message_id, post_text)
    except Exception as e:
        logger.debug(f"channel reaction failed: {e}")

    # Vision DISABLED for channel posts too (resource saving, same as groups).
    # Lyuba comments based on the post TEXT only. If a post is photo-only with
    # no caption, she makes a light generic comment or skips.

    extra_ctx = build_channel_context(chat, post_text, message.from_user)
    # Add channel + site recommendation context
    extra_ctx += (
        "\n\nРЕКОМЕНДАЦИИ (только если к месту):\n"
        "- Каналы: https://t.me/sochiautoparts, https://t.me/bmw_mpower_club\n"
        "- Магазин: https://sochiautoparts.ru/shop | Статьи: https://sochiautoparts.ru"
    )
    # Occasionally include a real product/post from the site
    try:
        if random.random() < 0.25:
            from bot import site_content as sc
            prod = await sc.random_product()
            if prod:
                extra_ctx += "\n\nТОВАР ИЗ МАГАЗИНА (упомяни если к месту): " + sc.format_product_for_context(prod)
    except Exception:
        pass

    try:
        resp = await asyncio.wait_for(
            ai_router.comment(
                prompt="Напиши короткий живой комментарий к этому посту канала.",
                extra_context=extra_ctx,
                mood=mood,
                route_type="comment",
            ),
            timeout=40.0,
        )
    except asyncio.TimeoutError:
        return

    if not resp.ok or not resp.text:
        return

    text = resp.text.strip()
    if not text:
        return
    try:
        await message.reply(text)
        await db.touch_channel_comment(chat.id)
    except Exception as e:
        logger.debug(f"channel comment reply failed: {e}")
