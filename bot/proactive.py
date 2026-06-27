"""
Proactive Topic Starter — Lyuba periodically initiates conversations in groups
where there's been silence for a while. This boosts group activity.

How it works:
- Background task runs every ~10 minutes.
- For each group Lyuba is in (from group_messages DB), checks last activity.
- If silence > SILENCE_THRESHOLD (e.g. 2 hours) AND Lyuba hasn't spoken recently,
  Lyuba sends a proactive topic-starter message (news, question, fact, thought).
- Rate-limited: max 1 proactive topic per group per SILENCE_THRESHOLD.
- Uses AI to generate a contextually relevant opener based on recent topics.

This makes Lyuba an ACTIVE community member who doesn't just respond but
also STARTS conversations — exactly as a real engaged group member would.
"""

import asyncio
_bot_ref = None

def set_bot(bot):
    global _bot_ref
    _bot_ref = bot
import logging
import random
import time
from typing import List, Dict

from bot import database as db
from bot.config import config
from bot.mood import current_mood_descriptor
from bot.context import recent_messages_to_text
from ai.router import ai_router
from bot.safe_send import safe_send

logger = logging.getLogger("luba.proactive")

# How long of silence before Lyuba initiates a topic (seconds)
SILENCE_THRESHOLD = 2 * 3600  # 2 hours
# How often to check groups (seconds)
CHECK_INTERVAL = 10 * 60  # 10 minutes
# Min time between Lyuba's proactive topics in the same group
MIN_TOPIC_INTERVAL = 3 * 3600  # 3 hours


# Topic starter templates — Lyuba uses these to spark conversations
TOPIC_STARTERS = [
    "народ, что думаете про",
    "кто-нибудь следил за",
    "а вы как считаете насчёт",
    "кстати, слышали что",
    "интересно ваше мнение —",
    "недавно думала про",
    "народ, а кто-нибудь",
    "блин, только что вспомнила —",
    "кстати о чём говорили,",
    "а что нового у всех? давно не виделись",
]

# General topics Lyuba can bring up (broad, non-political)
GENERAL_TOPICS = [
    "новые гаджеты и технологии",
    "какие фильмы/сериалы сейчас стоит посмотреть",
    "путешествия и куда бы хотелось поехать",
    "любимая еда и рецепты",
    "как проходит неделя",
    "что нового в мире авто",
    "интересные факты о которых мало кто знает",
    "хобби и увлечения",
    "какой кофе/чай любите",
    "что читаете сейчас",
    "планы на выходные",
    "любимые места в вашем городе",
]


async def _get_active_groups() -> List[Dict]:
    """Get groups where Lyuba has participated (has messages in DB)."""
    try:
        async with db._connect_db() as conn:
            async with conn.execute(
                """SELECT DISTINCT chat_id FROM group_messages
                   WHERE timestamp > ?
                   ORDER BY (SELECT MAX(timestamp) FROM group_messages g2 WHERE g2.chat_id = group_messages.chat_id) DESC
                   LIMIT 20""",
                (time.time() - 24 * 3600,)  # groups active in last 24h
            ) as cur:
                rows = await cur.fetchall()
        return [{"chat_id": r["chat_id"]} for r in rows]
    except Exception as e:
        logger.debug(f"get_active_groups error: {e}")
        return []


async def _check_and_start_topic(chat_id: int) -> None:
    """Check if group is silent and Lyuba should start a topic."""
    try:
        # Get last message time (any user) in this group
        recent = await db.get_recent_group_messages(chat_id, limit=5)
        if not recent:
            return

        last_msg_time = recent[-1].get("timestamp", 0) if recent else 0
        now = time.time()
        silence = now - last_msg_time

        # Only start topic if silence > threshold
        if silence < SILENCE_THRESHOLD:
            return

        # Check if Lyuba already started a topic recently
        last_bot = await db.last_bot_message_time(chat_id)
        if (now - last_bot) < MIN_TOPIC_INTERVAL:
            return

        # Rate limit check
        if str(chat_id).startswith("-"):
            from bot.safe_send import _check_rate as _safe_check_rate
            if not await _safe_check_rate(chat_id, config.GROUP_MAX_PER_MINUTE):
                return

        # Generate a topic-starter message
        recent_text = recent_messages_to_text(recent, limit=4)
        mood = await current_mood_descriptor()

        # Pick a random topic + starter
        topic = random.choice(GENERAL_TOPICS)
        starter = random.choice(TOPIC_STARTERS)

        prompt = (
            f"В группе «{chat_id}» давно тишина ({silence/3600:.0f}ч). "
            f"Начни беседу — поделись мыслью/новостью/вопросом чтобы оживить чат. "
            f"Тема для старта: {topic}. Используй оборот вроде «{starter}». "
            f"Коротко, живо, 1-2 предложения. Задай вопрос группе."
        )

        extra_ctx = (
            f"ГДЕ: группа.\n"
            f"ЗАДАЧА: инициировать беседу после тишины.\n"
            f"Настроение: {mood}.\n"
            f"НЕДАВНИЙ КОНТЕКСТ (если есть):\n{recent_text}\n"
            f"Будь естественной, не формальной. Цель — вовлечь людей в разговор."
        )

        try:
            resp = await asyncio.wait_for(
                ai_router.comment(prompt, extra_context=extra_ctx, mood=mood, route_type="comment"),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return

        if not resp.ok or not resp.text:
            return

        text = resp.text.strip()[:config.GROUP_MAX_CHARS]
        if not text:
            return

        # Send the topic-starter
        _bot = _bot_ref
        if _bot is None:
            return
        sent = await safe_send(_bot, chat_id, text)
        if sent:
            logger.info(f"Started proactive topic in group {chat_id} after {silence/3600:.1f}h silence")
            # Log as Lyuba's message
            await db.add_group_message(
                chat_id=chat_id,
                user_id=config.BOT_ID,
                username=config.BOT_USERNAME.lstrip("@"),
                first_name="Люба",
                content=text,
                is_media=False,
                is_bot=True,
            )
    except Exception as e:
        logger.debug(f"start_topic error for {chat_id}: {e}")


async def proactive_loop():
    """Background task: periodically check groups for silence and start topics."""
    from aiogram import Bot
    logger.info("Proactive topic starter loop started")
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL)
            groups = await _get_active_groups()
            logger.debug(f"Checking {len(groups)} groups for silence...")
            for g in groups:
                await _check_and_start_topic(g["chat_id"])
                await asyncio.sleep(2)  # small delay between groups
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"proactive_loop error: {e}")
            await asyncio.sleep(60)
