"""
Reactions / Likes for Lyuba — sets emoji reactions on messages in groups
and channel posts, to feel alive and engaged.

Uses Telegram Bot API setMessageReaction (available since Bot API 7.2).
Bots can set reactions on messages in groups/supergroups/channels where
they are members (channels require admin rights).

Strategy:
- In groups: with probability REACTION_PROB, react to a message Lyuba reads
  (even if she doesn't reply). Picks an emoji fitting the sentiment.
- On channel posts: with probability CHANNEL_REACTION_PROB, react before/after
  commenting.
- Reaction emoji must be from Telegram's allowed list (see ReactionTypeEmoji).
"""

import asyncio
import logging
import random
from typing import Optional

from aiogram import Bot
from aiogram.types import ReactionTypeEmoji

from bot.config import config

logger = logging.getLogger("luba.reactions")

# Allowed Telegram reaction emoji (subset — popular, always available).
# Full list: https://core.telegram.org/bots/api#reactiontypeemoji
REACTION_EMOJIS = ["👍", "❤️", "🔥", "😄", "🎉", "👀", "👌", "💯", "😮", "😢", "🙏"]

# Sentiment → emoji mapping (for context-aware reactions)
POSITIVE = ["❤️", "🔥", "🎉", "😄", "👍", "💯"]
NEUTRAL = ["👍", "👌", "👀"]
SURPRISE = ["😮", "👀"]
SAD = ["😢", "🙏"]


def _pick_emoji(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["поздрав", "ура", "класс", "супер", "обожаю", "люблю", "рад", "счасть"]):
        return random.choice(POSITIVE)
    if any(w in t for w in ["грустно", "печаль", "жаль", "больно", "умер", "соболезн"]):
        return random.choice(SAD)
    if any(w in t for w in ["ого", "вау", "шок", "не верю", "серьёзно", "что"]):
        return random.choice(SURPRISE)
    return random.choice(NEUTRAL)


async def maybe_react(bot: Bot, chat_id: int, message_id: int, text: str = "",
                      prob: Optional[float] = None) -> bool:
    """With probability `prob` (default config.REACTION_PROB), set a reaction.

    Returns True if a reaction was set.
    """
    p = prob if prob is not None else config.REACTION_PROB
    if random.random() > p:
        return False
    emoji = _pick_emoji(text)
    try:
        await bot.set_message_reaction(
            chat_id=chat_id,
            message_id=message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)],
        )
        logger.debug(f"Reacted {emoji} to msg {message_id} in chat {chat_id}")
        return True
    except Exception as e:
        # Common: bot not admin in channel, or reaction forbidden in this chat.
        logger.debug(f"Reaction failed (chat={chat_id}, msg={message_id}): {e}")
        return False


async def react_to_channel_post(bot: Bot, chat_id: int, message_id: int,
                                 text: str = "") -> bool:
    """React to a channel post (higher probability)."""
    return await maybe_react(bot, chat_id, message_id, text, prob=config.CHANNEL_REACTION_PROB)
