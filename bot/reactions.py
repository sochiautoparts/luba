"""Люба Reactions — emoji reactions on messages."""
import asyncio, logging, random
from typing import Optional
from aiogram import Bot
from aiogram.types import ReactionTypeEmoji
from bot.config import config
from bot import database as db

logger = logging.getLogger("luba.reactions")

_POSITIVE = ["👍", "❤️", "🔥", "😄", "👏", "🎉", "💪", "✨"]
_LOVE = ["❤️", "😍", "🥰", "💙", "💜"]
_FUN = ["😄", "😂", "🤣", "😆", "😎"]
_WOW = ["😮", "😱", "🤯", "👀", "🔥"]
_SAD = ["😢", "😔", "🙏", "💔"]
_THINK = ["🤔", "👀", "🧐", "💡"]
_NEUTRAL = ["👍", "👌", "🙌", "✨"]

def _pick_emoji(text):
    t = (text or "").lower()
    if any(w in t for w in ["люблю", "обожаю", "супер", "огонь", "класс", "топ", "🔥", "❤"]): return random.choice(_LOVE + ["🔥"])
    if any(w in t for w in ["смешн", "лол", "ха", "ржу", "😂", "🤣", "шутк"]): return random.choice(_FUN)
    if any(w in t for w in ["ого", "вау", "шок", "жесть", "😱", "невероятн", "удивил"]): return random.choice(_WOW)
    if any(w in t for w in ["грустн", "печаль", "жаль", "соболезн", " умер", "погиб"]): return random.choice(_SAD)
    if any(w in t for w in ["почему", "как так", "интересн", "думаю", "вопрос", "?"]): return random.choice(_THINK)
    if any(w in t for w in ["спасибо", "благодар", "спс"]): return random.choice(["🙏", "👍", "❤️"])
    return random.choice(_POSITIVE)

async def maybe_react(bot, chat_id, message_id, text="", prob=None, force=False):
    if not force:
        p = prob if prob is not None else config.REACTION_PROB
        if random.random() > p: return False
    if await db.already_reacted(chat_id, message_id): return False
    emoji = _pick_emoji(text)
    try:
        await bot.set_message_reaction(chat_id, message_id, [ReactionTypeEmoji(type="emoji", emoji=emoji)])
        await db.mark_reacted(chat_id, message_id)
        return True
    except Exception as e:
        logger.debug(f"reaction failed ({chat_id}/{message_id}): {e}")
        return False
