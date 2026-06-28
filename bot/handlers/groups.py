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

from bot.config import config
from bot import database as db
from bot.context import (
    user_descriptor, chat_descriptor, is_directed_at_lyuba,
    strip_mention, recent_messages_to_text, build_group_context,
)
from bot.mood import update_mood_from_message, current_mood_descriptor
from bot.media_handler import extract_caption
from bot.partners import partner_manager
from bot.web_search import verify_claim
from ai import ai as ai_client

logger = logging.getLogger("luba.groups")

group_router = Router()

_VERIFY_HINTS = [
    # News/events
    "новост", "правда ли", "это правда", "что случилось", "говорят что", "по данным",
    "сегодня", "вчера", "слышал", "прочитал", "вот пишут", "источник", "статья",
    "появился", "вышла", "анонс", "запустили", "анонсировал", "выпустил",
    # Facts/claims
    "сколько стоит", "цена", "когда выйдет", "узнал что", "оказывается",
    # Events
    "прошёл", "прошла", "состоялся", "состоялась", "открыли", "закрыли",
    "обновил", "обновление", "патч", "версия", "релиз",
    # Trending
    "тренд", "вирусный", "популярн", "обсуждают", "хайп",
]


def _needs_verification(text: str) -> bool:
    """Check if the message contains factual claims, news, or events worth verifying."""
    t = (text or "").lower()
    if len(t) < 15:
        return False
    return any(h in t for h in _VERIFY_HINTS)


def _is_event_or_news(text: str) -> bool:
    """Check if the message is about an event, news, or happening worth reacting to."""
    t = (text or "").lower()
    if len(t) < 10:
        return False
    event_hints = [
        "новост", "событие", "случил", "произош", "прошёл", "прошла",
        "состоялся", "открыли", "закрыли", "запустили", "анонс", "вышла",
        "выпустил", "обновлен", "релиз", "появился", "анонсировал",
        "сегодня", "вчера", "только что", "прямо сейчас",
    ]
    return any(h in t for h in event_hints)


def _is_politics_or_war(text: str) -> bool:
    """Detect topics Lyuba must avoid. She'll stay quiet on these.
    Only triggers on EXPLICIT political/military context — not generic words."""
    t = (text or "").lower()
    # Only block explicit political figures + military terms
    triggers = ["путин", "кремль", "госдума", "санкци", "сво", "мобилиз",
                "война", "зеленск", "байден", "трамп", "выборы", "парламент",
                "оранжев", "наци", "террор", "обеднен", "обстрел"]
    # Removed: "ввс" (too generic), "удар" (too generic — удар по мячу etc.)
    return any(w in t for w in triggers)


async def _log_group_message(message: Message, content: str = "", is_media: bool = False,
                              media_caption: str = "", is_bot: bool = False):
    """Log a group message to the DB for context.

    is_bot: True if this is Lyuba's own message OR any bot's message.
    For incoming messages, is_bot is auto-detected from from_user.
    """
    u = message.from_user
    # Auto-detect: mark as bot if author is Lyuba OR any bot
    if not is_bot and u and (u.id == config.BOT_ID or u.is_bot):
        is_bot = True
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
    """Decide if Lyuba responds to this group message.

    Lyuba is VERY ACTIVE: responds to direct mentions/replies ALWAYS,
    and proactively comments on most other messages (high probability).
    Only skips: OWN messages (anti-self-reply), politics/war.

    ANTI-SELF-REPLY (critical): Never respond to own messages (by user_id).
    Other bots ARE allowed — Lyuba can chat with them.

    NOTE: GROUP_MIN_INTERVAL is NOT used here anymore — safe_send's
    per-chat rate limiter (GROUP_MAX_PER_MINUTE) handles flood control.
    This ensures Lyuba is equally active in ALL groups, not just quiet ones.
    """
    u = message.from_user
    # CRITICAL: Skip OWN messages — by user_id match.
    if u and u.id == config.BOT_ID:
        return False

    directed = is_directed_at_lyuba(message)
    if directed:
        return True

    # Channel-forwarded posts in discussion groups — high proactive chance
    if message.sender_chat and message.sender_chat.type == "channel":
        return random.random() < config.GROUP_PROACTIVE_PROB

    # If this message is a REPLY to another user (a discussion thread),
    # Lyuba is MORE likely to join the conversation.
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id != config.BOT_ID:
            # Reply to another user/bot = active discussion → higher chance to join
            return random.random() < (config.GROUP_PROACTIVE_PROB + 0.2)

    # Other bots' messages (not directed at Lyuba): high proactive chance
    if u and u.is_bot:
        return random.random() < 0.65  # 65% for bots — very active interaction

    # Proactive: high probability (safe_send handles rate limiting)
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

    # Partner links — context-aware: only when message text matches partner categories
    # (auto, travel, shopping, electronics, etc.). Lyuba weaves ONE link naturally.
    try:
        links = partner_manager.get_all_partner_links_for_dialog(text, max_programs=2)
        if links:
            extra_ctx += "\n\nПартнёрские ссылки (вставь ОДНУ если к месту, естественно, не в каждом ответе):\n"
            for pl in links:
                extra_ctx += f"- {pl['name']} ({pl.get('label','')}): {pl['url']}\n"
    except Exception as e:
        logger.debug(f"partner links error: {e}")

    # Рекомендации обычным текстом, без меток — ничего не утекает
    extra_ctx += (
        "\n\nМожешь иногда (1 из 6 сообщений, если к месту) посоветовать каналы "
        "@sochiautoparts (авто-новости) или @bmw_mpower_club (BMW клуб), "
        "или магазин sochiautoparts.ru/shop."
    )

    # Include a real product / post from the site for context — higher probability
    # when the message is about shopping/parts/cars/products
    try:
        # Check if message is shopping/product-related → higher chance to include product
        t_lower = (text or "").lower()
        is_shopping = any(k in t_lower for k in [
            "купить", "магазин", "цена", "стоим", "заказ", "товар", "запчаст",
            "детал", "артикул", "подобрать", "найти", "выбор", "рекоменд"
        ])
        product_prob = 0.5 if is_shopping else 0.25
        post_prob = 0.25 if is_shopping else 0.12

        if random.random() < product_prob:
            from bot import site_content as sc
            prod = await sc.relevant_product(text) if text else await sc.random_product()
            if prod:
                extra_ctx += "\n\nТовар из магазина sochiautoparts.ru/shop (упомяни если к месту):\n" + sc.format_product_for_context(prod)
        if random.random() < post_prob:
            from bot import site_content as sc
            post = await sc.random_post()
            if post:
                extra_ctx += "\n\nСвежий пост на сайте (можешь поделиться): " + sc.format_post_for_context(post)
    except Exception as e:
        logger.debug(f"site content error: {e}")

    # Vision DISABLED in groups per design (resource saving).
    # Group photos are handled by their caption text only — Lyuba reacts to
    # the textual context of events, not the image bytes. This keeps groups
    # fast and cheap. (Vision stays enabled for private 1-on-1 chats.)
    # If a photo has no caption, we note "(фото без подписи)" in the prompt.

    # Web verification — CONCURRENT with AI (non-blocking!)
    # Previous version blocked 5s before AI call → in active groups messages
    # piled up → bot seemed inactive. Now runs in parallel.
    is_event = _is_event_or_news(text)
    needs_verify = _needs_verification(text)
    verify_task = None
    if is_event:
        verify_task = asyncio.create_task(verify_claim(text[:400]))
    elif needs_verify and random.random() < config.WEB_VERIFY_PROB:
        verify_task = asyncio.create_task(verify_claim(text[:400]))

    # Build the prompt for the AI
    prompt = strip_mention(text) if directed else text
    if not prompt:
        prompt = "(сообщение без текста — прокомментируй контекст чата, вступи в беседу)"

    # For events/news: instruct Lyuba to react, supplement, and share opinion
    if is_event:
        prompt = (
            "В группе поделились событием/новостью. Отреагируй живо — "
            "прокомментируй событие, дополни информацией если знаешь, "
            "поделись своим мнением. Обратись к автору по имени если уместно.\n" + prompt
        )
    elif directed:
        prompt = (
            "Тебе пишут напрямую (адресовано тебе). Ответь живо, можно чуть подробнее. "
            "Обратись по имени если уместно.\n" + prompt
        )
    else:
        prompt = (
            "Вступи в беседу — прокомментируй это сообщение живо. "
            "Ответь участнику, задай вопрос или поделись мнением. "
            "Обратись по имени если уместно.\n" + prompt
        )

    try:
        out = await asyncio.wait_for(
            ai_client.comment(prompt, extra_context=extra_ctx, mood=mood),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        return ""

    out = out or ""

    # Allow slightly longer replies for directed group messages
    limit = config.GROUP_MAX_CHARS if directed else config.COMMENT_MAX_CHARS
    if out:
        out = out[:limit]

    # Append web search results as source link (non-blocking — 2s max wait)
    if verify_task is not None:
        try:
            vctx = await asyncio.wait_for(verify_task, timeout=2.0)
            if vctx:
                import re as _re
                m = _re.search(r"https?://\S+", vctx)
                if m and m.group(0) not in out:
                    out += f"\n\nИсточник: {m.group(0)}"
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    return out.strip()


@group_router.message(F.photo)
async def handle_group_photo(message: Message):
    """Handle photos in groups.

    Lyuba reacts to photos and responds when directed at her.
    - Directed (mention/reply to Lyuba): ALWAYS responds, even for album photos
      and even without caption.
    - Non-directed: sets a reaction with moderate probability (25%), no text.
    - Albums: only process the FIRST photo (skip subsequent ones with same
      media_group_id) to avoid duplicate responses — BUT still respond if
      that first photo is directed at Lyuba.
    - CRITICAL: Never react/respond to own photos (anti-self-reply).
      Other bots' photos ARE allowed (Lyuba can interact with bots).
    """
    if message.chat.type not in ("group", "supergroup"):
        return
    if message.from_user is None:
        return
    # CRITICAL: Skip OWN messages — prevents self-reply loops.
    # Other bots ARE allowed (Lyuba can chat with them).
    u = message.from_user
    if u.id == config.BOT_ID:
        return
    caption = extract_caption(message)
    await _log_group_message(message, content=caption, is_media=True, media_caption=caption,
                             is_bot=False)
    await update_mood_from_message(caption)

    if _is_politics_or_war(caption):
        return

    directed = is_directed_at_lyuba(message)

    # Albums: Telegram sends each photo separately with same media_group_id.
    # To avoid duplicate responses, track which albums we've already handled.
    # BUT if directed at Lyuba, always respond (even in album).
    if message.media_group_id:
        if not directed:
            # Non-directed album photo — just set a reaction, no text
            if caption and random.random() < 0.15:
                try:
                    from bot.reactions import maybe_react
                    asyncio.create_task(maybe_react(
                        message.bot, message.chat.id, message.message_id, caption, prob=1.0))
                except Exception:
                    pass
            return
        # Directed album photo — respond to this one (skip subsequent photos in same album)
        # Use a simple in-memory set to track handled albums per chat
        if not hasattr(handle_group_photo, '_seen_albums'):
            handle_group_photo._seen_albums = {}
        album_key = f"{message.chat.id}:{message.media_group_id}"
        now = time.time()
        # Clean old entries (>5 min)
        handle_group_photo._seen_albums = {k: v for k, v in handle_group_photo._seen_albums.items() if now - v < 300}
        if album_key in handle_group_photo._seen_albums:
            return  # already responded to this album
        handle_group_photo._seen_albums[album_key] = now

    # Non-directed single photo: set reaction with moderate probability, no text
    if not directed:
        if caption and random.random() < 0.25:
            try:
                from bot.reactions import maybe_react
                asyncio.create_task(maybe_react(
                    message.bot, message.chat.id, message.message_id, caption, prob=1.0))
            except Exception:
                pass
        # Also react to photos without caption sometimes (visual engagement)
        elif not caption and random.random() < 0.10:
            try:
                from bot.reactions import maybe_react
                asyncio.create_task(maybe_react(
                    message.bot, message.chat.id, message.message_id, "", prob=1.0))
            except Exception:
                pass
        return  # no text response for non-directed photos

    # Directed photo: ALWAYS respond — even without caption.
    if caption:
        photo_prompt = caption
    else:
        photo_prompt = "(тебе прислали фото — коротко отреагируй живо, предположи что там может быть по контексту)"

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        out = await _generate_group_response(message, photo_prompt, directed)
    except Exception as e:
        logger.error(f"group photo response error: {e}")
        return
    if not out:
        return
    try:
        from bot.safe_send import safe_reply
        await safe_reply(message.bot, message, out, always_reply=True, priority=directed)
    except Exception as e:
        logger.debug(f"safe_reply (photo) failed: {e}")
    await _log_group_message(message, content=out, is_media=False, is_bot=True)


@group_router.message(F.text)
async def handle_group_text(message: Message):
    if message.chat.type not in ("group", "supergroup"):
        return
    if message.from_user is None:
        return
    # CRITICAL: Skip OWN messages — prevents self-reply loops.
    # Other bots ARE allowed (Lyuba can chat with them).
    u = message.from_user
    if u.id == config.BOT_ID:
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
    # ALWAYS reply (thread) to the user's message — so it's clear WHO Lyuba
    # is answering. priority=True for directed messages (higher rate-limit cap,
    # never silently dropped).
    try:
        from bot.safe_send import safe_reply
        await safe_reply(message.bot, message, out, always_reply=True, priority=directed)
    except Exception as e:
        logger.debug(f"safe_reply failed: {e}")
    await _log_group_message(message, content=out, is_media=False, is_bot=True)

    # Extract and store long-term facts about users (memory)
    try:
        await _extract_and_store_memory(message, text)
    except Exception as e:
        logger.debug(f"memory extraction error: {e}")


async def _extract_and_store_memory(message: Message, text: str):
    """Extract personal facts from user messages and store in group_memory."""
    if not text or not message.from_user:
        return
    t = text.lower().strip()
    user_id = message.from_user.id
    chat_id = message.chat.id
    name = message.from_user.first_name or ""
    patterns = [
        ("я живу в ", "живёт в"), ("я из ", "из"), ("я работаю ", "работает"),
        ("у меня собака", "есть собака"), ("у меня кот", "есть кот"),
        ("я люблю ", "любит"), ("мне нравится ", "нравится"),
        ("я обожаю ", "обожает"), ("я фрилансер", "фрилансер"),
    ]
    for pattern, label in patterns:
        if pattern in t:
            idx = t.index(pattern) + len(pattern)
            rest = text[idx:idx+80].split(".")[0].split("!")[0].split("?")[0].strip()
            if rest and 2 < len(rest) < 80:
                fact = f"{name} {label} {rest}"
                await db.add_group_memory(chat_id, user_id, fact)
                break
