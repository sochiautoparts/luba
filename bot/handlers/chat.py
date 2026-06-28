"""Private chat handler for Lyuba — 1-on-1 conversations with memory."""

import asyncio
import logging
import random

from aiogram import Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.enums import ChatAction

from bot.config import config
from bot import database as db
from bot.context import build_private_context
from bot.mood import update_mood_from_message, current_mood_descriptor
from bot.media_handler import get_photo_data_uri, extract_caption
from bot.partners import partner_manager
from bot.web_search import verify_claim
from ai import ai as ai_client, THINKING_PHRASES, GREETING_PHRASES

logger = logging.getLogger("luba.chat")

chat_router = Router()

_VERIFY_HINTS = ["новост", "правда ли", "это правда", "сколько стоит", "цена",
                 "когда выйдет", "что случилось", "узнать", "проверь",
                 "по данным", "говорят что"]


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


@chat_router.message(CommandStart(), F.chat.type == "private")
async def cmd_start(message: Message):
    if not await _check_user(message):
        return
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    await message.answer(random.choice(GREETING_PHRASES))


@chat_router.message(Command("help"), F.chat.type == "private")
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


@chat_router.message(Command("clear"), F.chat.type == "private")
async def cmd_clear(message: Message):
    if not await _check_user(message):
        return
    await db.clear_chat_history(message.from_user.id)
    await message.answer("всё, чистый лист 🙈 о чём поговорим?")


@chat_router.message(Command("mood"), F.chat.type == "private")
async def cmd_mood(message: Message):
    if not await _check_user(message):
        return
    mood = await current_mood_descriptor()
    m = await db.get_mood()
    await message.answer(
        f"сейчас я {mood}. энергии примерно {int(m.get('energy', 0.5) * 100)}% ☕"
    )


@chat_router.message(F.photo, F.chat.type == "private")
async def handle_photo(message: Message):
    if not await _check_user(message):
        return
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    status = await message.answer(random.choice(THINKING_PHRASES))

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

    # Vision (HF Qwen3-VL → Pollinations)
    text = await ai_client.vision(data_uri, prompt)
    if text:
        text = text[:config.CHAT_MAX_CHARS]
        # Maybe add partner link if photo is car/product related
        partner_hint = ""
        try:
            links = partner_manager.get_all_partner_links_for_dialog(
                caption or text, max_programs=1
            )
            if links:
                partner_hint = f"\n\nесли надо — глянь тут: {links[0]['url']}"
        except Exception:
            pass
        await status.delete()
        await message.answer(text + partner_hint)
    else:
        await status.delete()
        await message.answer("блин, не смогла разглядеть фото 🙈 попробуй ещё разок?")


@chat_router.message(F.text, F.chat.type == "private")
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
    status = await message.answer(random.choice(THINKING_PHRASES))

    await update_mood_from_message(text)
    mood = await current_mood_descriptor()
    extra_ctx = build_private_context(message)

    # Channel + site recommendations context (always present, used sparingly)
    # Обычным текстом, без меток — ничего не утекает
    extra_ctx += (
        "\n\nМожешь иногда (1 из 6 сообщений, если к месту) посоветовать каналы "
        "@sochiautoparts (авто-новости) или @bmw_mpower_club (BMW клуб), "
        "или магазин sochiautoparts.ru/shop."
    )

    # Include real product / post from the site — higher probability for shopping topics
    try:
        t_lower = (text or "").lower()
        is_shopping = any(k in t_lower for k in [
            "купить", "магазин", "цена", "стоим", "заказ", "товар", "запчаст",
            "детал", "артикул", "подобрать", "найти", "выбор", "рекоменд"
        ])
        product_prob = 0.5 if is_shopping else 0.25
        post_prob = 0.25 if is_shopping else 0.12

        if random.random() < product_prob:
            from bot import site_content as sc
            prod = await sc.relevant_product(text)
            if prod:
                extra_ctx += "\n\n" + sc.format_product_for_context(prod)
        if random.random() < post_prob:
            from bot import site_content as sc
            post = await sc.random_post()
            if post:
                extra_ctx += "\n\nСвежий пост на сайте (можешь поделиться): " + sc.format_post_for_context(post)
    except Exception as e:
        logger.debug(f"site content error: {e}")

    # Collect partner links relevant to the message
    try:
        partner_links = partner_manager.get_all_partner_links_for_dialog(text, max_programs=2)
        if partner_links:
            extra_ctx += "\n\nПартнёрские ссылки (используй ОДНУ если к месту, естественно, не в каждом ответе):\n"
            for pl in partner_links:
                extra_ctx += f"- {pl['name']} ({pl['label']}): {pl['url']}\n"
    except Exception as e:
        logger.debug(f"partner links error: {e}")

    # Web verification — BEFORE AI call so results are in context
    web_context = ""
    if _needs_verification(text):
        try:
            web_context = await asyncio.wait_for(verify_claim(text), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            web_context = ""

    if web_context:
        extra_ctx += f"\n\nРезультаты веб-поиска (используй для дополнения ответа):\n{web_context}"

    try:
        out = await asyncio.wait_for(
            ai_client.chat(
                user_id=user_id,
                message=text,
                extra_context=extra_ctx,
                mood=mood,
                max_chars=config.CHAT_MAX_CHARS,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        await status.delete()
        await message.answer("ой, застряла немного 🙈 давай ещё раз?")
        return

    await status.delete()
    out = out or random.choice(THINKING_PHRASES)

    # If AI didn't include a source URL but we have web results, append one
    if web_context:
        try:
            import re as _re
            first_url = _re.search(r"https?://\S+", web_context)
            if first_url and first_url.group(0) not in out:
                out += f"\n\nвот, нашла: {first_url.group(0)}"
        except Exception:
            pass

    # safe_send handles RetryAfter + length limits
    try:
        from bot.safe_send import safe_send
        await safe_send(message.bot, message.chat.id, out)
    except Exception:
        try:
            await message.answer(out)
        except Exception:
            pass
