"""
Group / Supergroup handler for Lyuba — ACTIVE participation.

Behavior (requires Privacy Mode DISABLED via @BotFather so the bot receives
ALL group messages, not just mentions):

For every group message:
  1. Log it to group_messages (context window, last N per group).
  2. Update mood from sentiment.
  3. Decide whether to respond:
     - ALWAYS if directed at Lyuba (mention / reply to her / her name).
     - With probability GROUP_PROACTIVE_PROB otherwise (be "very active").
  4. Respect GROUP_MIN_INTERVAL (anti-flood) and skip other bots (avoid loops).
  5. Build rich context: who, where, recent messages, long-term memory facts.
  6. Collect relevant partner links contextually.
  7. Optionally verify factual claims via web (concurrent).
  8. Generate a SHORT comment (route_type=comment) and reply.

Lyuba remembers per-(chat,user) facts and recent topics per chat.
"""

import asyncio
import logging
import random
import time
from typing import List

from aiogram import Router, F, types
from aiogram.types import Message
from aiogram.enums import ChatAction

from bot.config import config, persona
from bot import database as db
from bot.context import (
    user_descriptor, chat_descriptor, is_directed_at_lyuba,
    strip_mention, recent_messages_to_text, build_group_context,
)
from bot.mood import update_mood_from_message, current_mood_descriptor
from bot.media_handler import extract_caption
from bot.partners import partner_manager
from bot.web_search import verify_claim
from ai.router import ai_router

logger = logging.getLogger("luba.groups")

group_router = Router()

_VERIFY_HINTS = ["новост", "правда ли", "это правда", "сколько стоит", "цена",
                 "когда выйдет", "что случилось", "говорят что", "по данным"]


def _needs_verification(text: str) -> bool:
    t = (text or "").lower()
    if len(t) < 20:
        return False
    return any(h in t for h in _VERIFY_HINTS)


def _is_politics_or_war(text: str) -> bool:
    """Detect topics Lyuba must avoid. She'll stay quiet on these."""
    t = (text or "").lower()
    triggers = ["путин", "кремль", "госдума", "санкци", "сво", "мобилиз",
                "война", "зеленск", "байден", "трамп", "выборы", "парламент",
                "оранжев", "наци", "террор", "обеднен", "ввс", "удар",
                "ракетн", "обстрел"]
    return any(w in t for w in triggers)


async def _log_group_message(message: Message, content: str = "", is_media: bool = False,
                              media_caption: str = "", is_bot: bool = False):
    u = message.from_user
    await db.add_group_message(
        chat_id=message.chat.id,
        user_id=u.id if u else 0,
        username=(u.username or "") if u else "",
        first_name=(u.first_name or "") if u else "",
        content=content or (message.text or ""),
        is_media=is_media,
        media_caption=media_caption,
        is_bot=is_bot,
    )


async def _should_respond(message: Message) -> bool:
    """Decide if Lyuba responds to this group message."""
    # Skip if from a bot (avoid loops) — unless it's a reply to Lyuba
    u = message.from_user
    if u and u.is_bot and u.id != config.BOT_ID:
        # Only respond if someone replied to Lyuba and the replier is a bot? No — skip bots.
        return False

    # Skip own messages
    if u and u.id == config.BOT_ID:
        return False

    # Skip channel-forwarded posts in groups (handled by channel logic if it's a channel)
    if message.sender_chat and message.sender_chat.type == "channel":
        # This is a channel post forwarded into a discussion group — treat as comment target
        return random.random() < config.GROUP_PROACTIVE_PROB

    directed = is_directed_at_lyuba(message)
    if directed:
        return True

    # Proactive: with some probability, but respect min interval
    last_bot = await db.last_bot_message_time(message.chat.id)
    if (time.time() - last_bot) < config.GROUP_MIN_INTERVAL:
        return False
    return random.random() < config.GROUP_PROACTIVE_PROB


async def _generate_group_response(message: Message, text: str, directed: bool) -> str:
    """Generate Lyuba's group response. Returns text or empty string.

    Vision is DISABLED in groups (resource saving) — group photos are handled
    by their caption text only.
    """
    # Load recent context + memory
    recent = await db.get_recent_group_messages(message.chat.id, limit=10)
    recent_text = recent_messages_to_text(recent, limit=8)
    memory_facts_rows = await db.get_group_memory(message.chat.id, limit=6)
    memory_facts = [r["fact"] for r in memory_facts_rows]

    mood = await current_mood_descriptor()
    extra_ctx = build_group_context(message, recent_text, memory_facts)

    # Partner links
    try:
        links = partner_manager.get_all_partner_links_for_dialog(text, max_programs=2)
        if links:
            extra_ctx += "\n\nПартнёрские ссылки (ОДНА если к месту, КАК ЕСТЬ):\n"
            for pl in links:
                extra_ctx += f"- {pl['name']}: {pl['url']}\n"
    except Exception as e:
        logger.debug(f"partner links error: {e}")

    # Channel recommendations context (so Lyuba can suggest subscribing)
    extra_ctx += (
        "\n\nРЕКОМЕНДАЦИИ (только если к месту, ~1 из 8 сообщений):\n"
        "- Каналы: https://t.me/sochiautoparts (авто-новости), https://t.me/bmw_mpower_club (BMW клуб)\n"
        "- Магазин: https://sochiautoparts.ru/shop | Статьи: https://sochiautoparts.ru"
    )

    # Occasionally include a real product / post from the site for context
    try:
        if random.random() < 0.3:
            from bot import site_content as sc
            prod = await sc.relevant_product(text) if text else await sc.random_product()
            if prod:
                extra_ctx += "\n\nСЛУЧАЙНЫЙ ТОВАР ИЗ МАГАЗИНА (можешь упомянуть если к месту):\n" + sc.format_product_for_context(prod)
        if random.random() < 0.15:
            from bot import site_content as sc
            post = await sc.random_post()
            if post:
                extra_ctx += "\n\nСВЕЖИЙ ПОСТ НА САЙТЕ: " + sc.format_post_for_context(post)
    except Exception as e:
        logger.debug(f"site content error: {e}")

    # Vision DISABLED in groups per design (resource saving).
    # Group photos are handled by their caption text only — Lyuba reacts to
    # the textual context of events, not the image bytes. This keeps groups
    # fast and cheap. (Vision stays enabled for private 1-on-1 chats.)
    # If a photo has no caption, we note "(фото без подписи)" in the prompt.

    # Web verification (concurrent, best-effort)
    verify_task = None
    if _needs_verification(text) and random.random() < config.WEB_VERIFY_PROB:
        verify_task = asyncio.create_task(verify_claim(text[:200]))

    # Always use the comment() path for groups — it does NOT touch the user's
    # private chat_history (group context comes from extra_context above).
    # The extra_context already tells Lyuba whether the message is directed at her.
    prompt = strip_mention(text) if directed else text
    if not prompt:
        prompt = "(сообщение без текста — прокомментируй контекст чата)"
    if directed:
        prompt = "Тебе пишут напрямую (адресовано тебе). Ответь живо, можно чуть подробнее.\n" + prompt

    try:
        resp = await asyncio.wait_for(
            ai_router.comment(prompt, extra_context=extra_ctx, mood=mood, route_type="comment"),
            timeout=45.0,
        )
    except asyncio.TimeoutError:
        return ""

    out = resp.text or ""

    # Allow slightly longer replies for directed group messages
    limit = config.GROUP_MAX_CHARS if directed else config.COMMENT_MAX_CHARS
    if out:
        out = out[:limit]

    # Append verification if needed
    if verify_task is not None:
        try:
            vctx = await asyncio.wait_for(verify_task, timeout=2.0)
            if vctx:
                import re as _re
                m = _re.search(r"https?://\S+", vctx)
                if m and ("не уверена" in out.lower() or "не знаю" in out.lower()):
                    out += f" вот: {m.group(0)}"
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    return out.strip()


@group_router.message(F.photo)
async def handle_group_photo(message: Message):
    if message.chat.type not in ("group", "supergroup"):
        return
    if message.from_user is None:
        return
    caption = extract_caption(message)
    await _log_group_message(message, content=caption, is_media=True, media_caption=caption,
                             is_bot=False)
    await update_mood_from_message(caption)

    if _is_politics_or_war(caption):
        return
    if not await _should_respond(message):
        return

    directed = is_directed_at_lyuba(message)
    # Vision disabled in groups — don't download the photo (saves bandwidth +
    # CPU). Lyuba reacts to the caption text only.
    # If no caption, use a placeholder so the model knows it's a photo.
    photo_note = ""
    if not caption:
        photo_note = "(прислали фото без подписи — прокомментируй коротко, предположи что там может быть по контексту чата)"

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        out = await _generate_group_response(message, caption or photo_note, directed)
    except Exception as e:
        logger.error(f"group photo response error: {e}")
        return
    if not out:
        return
    try:
        if directed and message.reply_to_message:
            await message.reply_to_message.reply(out)
        else:
            await message.answer(out)
    except Exception as e:
        logger.debug(f"send group reply failed: {e}")
    # Log Lyuba's own message
    await _log_group_message(message, content=out, is_media=False, is_bot=True)


@group_router.message(F.text)
async def handle_group_text(message: Message):
    if message.chat.type not in ("group", "supergroup"):
        return
    if message.from_user is None:
        return
    text = (message.text or "").strip()
    if not text:
        return

    # Log the message
    await _log_group_message(message, content=text, is_media=False, is_bot=False)
    await update_mood_from_message(text)

    # Set a reaction (like) on some messages Lyuba reads — feels alive.
    # This runs even when she doesn't reply (proactive engagement).
    try:
        from bot.reactions import maybe_react
        asyncio.create_task(maybe_react(message.bot, message.chat.id, message.message_id, text))
    except Exception:
        pass

    # Skip commands (unless directed at Lyuba)
    if text.startswith("/") and not is_directed_at_lyuba(message):
        return

    # Avoid politics/war — stay quiet
    if _is_politics_or_war(text) and not is_directed_at_lyuba(message):
        return

    if not await _should_respond(message):
        return

    directed = is_directed_at_lyuba(message)
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        out = await _generate_group_response(message, text, directed)
    except Exception as e:
        logger.error(f"group text response error: {e}")
        return
    if not out:
        return
    try:
        if directed and message.reply_to_message:
            await message.reply_to_message.reply(out)
        else:
            await message.answer(out)
    except Exception as e:
        logger.debug(f"send group reply failed: {e}")
    await _log_group_message(message, content=out, is_media=False, is_bot=True)
