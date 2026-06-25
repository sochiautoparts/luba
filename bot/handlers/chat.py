"""Private chat handler for Lyuba — 1-on-1 conversations with memory."""

import asyncio
import logging
import random
from typing import List

from aiogram import Router, F, types
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.enums import ChatAction

from bot.config import config, persona
from bot import database as db
from bot.context import build_private_context, user_descriptor
from bot.mood import update_mood_from_message, current_mood_descriptor
from bot.media_handler import get_photo_data_uri, extract_caption
from bot.partners import partner_manager
from bot.web_search import web_search, format_search_results, verify_claim
from ai.router import ai_router

logger = logging.getLogger("luba.chat")

chat_router = Router()

# Keywords that suggest a factual/news claim worth verifying
_VERIFY_HINTS = ["новост", "правда ли", "это правда", "сколько стоит", "цена", "когда выйдет",
                 "что случилось", "узнать", "проверь", "по данным", "говорят что"]


def _needs_verification(text: str) -> bool:
    t = (text or "").lower()
    if not t or len(t) < 15:
        return False
    return any(h in t for h in _VERIFY_HINTS)


async def _check_user(message: Message) -> bool:
    if message.from_user is None:
        return False
    if message.from_user.is_bot and message.from_user.id != config.BOT_ID:
        return False
    user = await db.get_or_create_user(
        user_id=message.from_user.id,
        username=message.from_user.username or "",
        first_name=message.from_user.first_name or "",
        last_name=message.from_user.last_name or "",
        language_code=message.from_user.language_code or "ru",
    )
    if user.get("is_blocked"):
        return False
    return True


@chat_router.message(CommandStart())
async def cmd_start(message: Message):
    if not await _check_user(message):
        return
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    greeting = random.choice(persona.greeting_phrases)
    await message.answer(greeting)


@chat_router.message(Command("help"))
async def cmd_help(message: Message):
    if not await _check_user(message):
        return
    await message.answer(
        "я Люба — просто поболтать могу обо всём ☕\n\n"
        "/clear — забыть что обсуждали\n"
        "/mood — покажу своё настроение\n"
        "ещё я вижу фото и могу проверить что-нибудь в интернете. "
        "просто пиши как знакомой 😊"
    )


@chat_router.message(Command("clear"))
async def cmd_clear(message: Message):
    if not await _check_user(message):
        return
    await db.clear_chat_history(message.from_user.id)
    await message.answer("всё, чистый лист 🙈 о чём поговорим?")


@chat_router.message(Command("mood"))
async def cmd_mood(message: Message):
    if not await _check_user(message):
        return
    mood = await current_mood_descriptor()
    m = await db.get_mood()
    await message.answer(f"сейчас я {mood}. энергии примерно {int(m.get('energy',0.5)*100)}% ☕")


@chat_router.message(F.photo)
async def handle_photo(message: Message):
    if not await _check_user(message):
        return
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    status = await message.answer(random.choice(persona.thinking_phrases))

    caption = extract_caption(message)
    data_uri = await get_photo_data_uri(message.bot, message.photo)
    if not data_uri:
        await status.delete()
        await message.answer("ой, не могу открыть фото 🙈 попробуй ещё раз?")
        return

    mood = await current_mood_descriptor()
    await update_mood_from_message(caption or "фото")

    prompt = "Опиши что на фото и прокомментируй живо, как подруга."
    if caption:
        prompt = f"Пользователь прислал фото с подписью: «{caption}». Опиши и прокомментируй."
    extra_ctx = build_private_context(message)

    # Vision (Pollinations free)
    vision_resp = await ai_router.vision(data_uri, prompt, system_prompt="")
    if vision_resp.ok:
        text = vision_resp.text[:config.CHAT_MAX_CHARS]
        # Maybe add partner link if photo is car/product related
        partner_hint = ""
        try:
            links = partner_manager.get_all_partner_links_for_dialog(caption or text, max_programs=1)
            if links:
                partner_hint = f"\n\nесли надо — глянь тут: {links[0]['url']}"
        except Exception:
            pass
        await status.delete()
        await message.answer(text + partner_hint)
    else:
        await status.delete()
        await message.answer("блин, не смогла разглядеть фото 🙈 попробуй ещё разок?")


@chat_router.message(F.text)
async def handle_text(message: Message):
    if not await _check_user(message):
        return
    text = (message.text or "").strip()
    if not text:
        return
    # Unknown command guard
    if text.startswith("/"):
        cmd = text.split()[0].lower().split("@")[0]
        known = {"/start", "/help", "/clear", "/mood"}
        if cmd not in known:
            await message.answer(f"не знаю команду {cmd} 🤔 напиши /help")
            return

    await _process_private(message, text)


async def _process_private(message: Message, text: str):
    """Core private chat processing."""
    user_id = message.from_user.id
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    status = await message.answer(random.choice(persona.thinking_phrases))

    await update_mood_from_message(text)
    mood = await current_mood_descriptor()
    extra_ctx = build_private_context(message)

    # Channel + site recommendations context
    extra_ctx += (
        "\n\nРЕКОМЕНДАЦИИ (только если к месту, ~1 из 8 сообщений):\n"
        "- Каналы: https://t.me/sochiautoparts (авто-новости), https://t.me/bmw_mpower_club (BMW клуб)\n"
        "- Магазин: https://sochiautoparts.ru/shop | Статьи: https://sochiautoparts.ru"
    )

    # Occasionally include a real product / post from the site
    try:
        if random.random() < 0.25:
            from bot import site_content as sc
            prod = await sc.relevant_product(text)
            if prod:
                extra_ctx += "\n\nСЛУЧАЙНЫЙ ТОВАР ИЗ МАГАЗИНА (упомяни если к месту):\n" + sc.format_product_for_context(prod)
        if random.random() < 0.12:
            from bot import site_content as sc
            post = await sc.random_post()
            if post:
                extra_ctx += "\n\nСВЕЖИЙ ПОСТ НА САЙТЕ: " + sc.format_post_for_context(post)
    except Exception as e:
        logger.debug(f"site content error: {e}")

    # Collect partner links relevant to the message
    partner_links = []
    try:
        partner_links = partner_manager.get_all_partner_links_for_dialog(text, max_programs=2)
        if partner_links:
            extra_ctx += "\n\nПартнёрские ссылки (используй ОДНУ если к месту, КАК ЕСТЬ):\n"
            for pl in partner_links:
                extra_ctx += f"- {pl['name']} ({pl['label']}): {pl['url']}\n"
            extra_ctx += "Вставь ссылку естественно, не в каждом ответе."
    except Exception as e:
        logger.debug(f"partner links error: {e}")

    # Web verification (concurrent with AI)
    verify_ctx = ""
    need_verify = _needs_verification(text) and random.random() < config.WEB_VERIFY_PROB
    verify_task = None
    if need_verify:
        verify_task = asyncio.create_task(verify_claim(text))

    try:
        response = await asyncio.wait_for(
            ai_router.chat(
                user_id=user_id,
                message=text,
                extra_context=extra_ctx,
                route_type="chat",
                mood=mood,
                max_chars=config.CHAT_MAX_CHARS,
            ),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        await status.delete()
        await message.answer("ой, застряла немного 🙈 давай ещё раз?")
        return

    # Attach verification result if available
    if verify_task is not None:
        try:
            verify_ctx = await asyncio.wait_for(verify_task, timeout=3.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            verify_ctx = ""
        except Exception:
            verify_ctx = ""

    await status.delete()
    out = response.text or random.choice(persona.thinking_phrases)
    if verify_ctx and ("не уверена" in out.lower() or "не знаю" in out.lower() or need_verify):
        # Append a compact verification note
        import re as _re
        first_url = _re.search(r"https?://\S+", verify_ctx)
        if first_url:
            out += f"\n\nвот, нашла кое-что: {first_url.group(0)}"
    await message.answer(out)
