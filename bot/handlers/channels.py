"""Люба Channel handler — ONLY emoji reactions on channel posts."""
import logging, random
from aiogram import Router, F
from aiogram.types import Message, Chat
from bot.config import config
from bot import database as db
from bot.reactions import maybe_react

logger = logging.getLogger("luba.channels")
channel_router = Router()

def _is_politics_or_war(text):
    t = (text or "").lower()
    return any(w in t for w in ["путин", "кремль", "госдума", "санкци", "сво", "мобилиз", "война", "зеленск", "байден", "трамп", "выборы", "парламент", "ракетн", "обстрел"])

@channel_router.channel_post(F.text | F.photo | F.video | F.animation | F.sticker | F.voice | F.document | F.video_note)
async def handle_channel_post(message):
    chat = message.chat
    await db.upsert_channel(chat.id, chat.username or "", chat.title or "")
    if not await db.is_channel_enabled(chat.id): return
    if random.random() > config.CHANNEL_REACTION_PROB: return
    post_text = (message.caption or message.text or "").strip()
    if _is_politics_or_war(post_text): return
    try:
        await maybe_react(message.bot, chat.id, message.message_id, post_text, prob=1.0, force=True)
    except Exception as e:
        logger.debug(f"channel reaction failed: {e}")

@channel_router.channel_post()
async def handle_channel_post_catchall(message):
    chat = message.chat
    await db.upsert_channel(chat.id, chat.username or "", chat.title or "")
    if not await db.is_channel_enabled(chat.id): return
    if random.random() > config.CHANNEL_REACTION_PROB: return
    try:
        await maybe_react(message.bot, chat.id, message.message_id, "", prob=1.0, force=True)
    except: pass
