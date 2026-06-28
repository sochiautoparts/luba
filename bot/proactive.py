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
from ai import ai as ai_client
from bot.safe_send import safe_send

logger = logging.getLogger("luba.proactive")

# How long of silence before Lyuba initiates a topic (seconds)
SILENCE_THRESHOLD = 2 * 3600  # 2 hours
# How often to check groups (seconds)
CHECK_INTERVAL = 10 * 60  # 10 minutes
# Min time between Lyuba's proactive topics in the same group
MIN_TOPIC_INTERVAL = 3 * 3600  # 3 hours
# Chance to inject a topic EVEN in active groups (no silence required).
# Makes Lyuba an active participant who starts conversations, not just responds.
ACTIVE_GROUP_INJECTION_PROB = 0.08  # 8% per check in active groups
# Min time between Lyuba's injections in active groups (shorter than silent)
ACTIVE_MIN_INTERVAL = 45 * 60  # 45 min between injections in same active group


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
    """Check if group needs a topic — either after silence OR active injection.

    Two modes:
      1. SILENT group (silence > 2h): always start a topic (if min interval passed)
      2. ACTIVE group (recent activity): 8% chance to inject a topic
         (if 45min since last Lyuba message) — makes her an active participant
         who starts conversations, not just responds.
    """
    try:
        recent = await db.get_recent_group_messages(chat_id, limit=5)
        if not recent:
            return

        last_msg_time = recent[-1].get("timestamp", 0) if recent else 0
        now = time.time()
        silence = now - last_msg_time

        # Check if Lyuba already started a topic recently
        last_bot = await db.last_bot_message_time(chat_id)
        since_bot = now - last_bot

        is_silent = silence >= SILENCE_THRESHOLD
        # Active injection: 8% chance, only if 45min since last Lyuba msg
        is_active_inject = (
            not is_silent
            and random.random() < ACTIVE_GROUP_INJECTION_PROB
            and since_bot >= ACTIVE_MIN_INTERVAL
        )

        if is_silent:
            if since_bot < MIN_TOPIC_INTERVAL:
                return
        elif is_active_inject:
            pass  # OK, inject into active group
        else:
            return  # neither silent nor injection triggered

        # Rate limit check
        if str(chat_id).startswith("-"):
            from bot.safe_send import _check_rate as _safe_check_rate
            if not await _safe_check_rate(chat_id, config.GROUP_MAX_PER_MINUTE):
                return

        # Generate a topic-starter message
        recent_text = recent_messages_to_text(recent, limit=4)
        mood = await current_mood_descriptor()

        # Build dialog_history — недавние сообщения как proper role-tagged dialog
        # чтобы Lyuba помнила контекст беседы и не повторялась
        dialog_history = []
        for m in recent:
            who = m.get("first_name") or m.get("username") or "кто-то"
            if m.get("user_id") == config.BOT_ID:
                role = "assistant"
                content = m.get("content", "")
            else:
                role = "user"
                content = f"{who}: {m.get('content', '')}"
            if content.strip():
                dialog_history.append({"role": role, "content": content})

        topic = random.choice(GENERAL_TOPICS)
        starter = random.choice(TOPIC_STARTERS)

        if is_silent:
            prompt = (
                f"В группе давно тишина ({silence/3600:.0f}ч). "
                f"Начни беседу — поделись мыслью/новостью/вопросом чтобы оживить чат. "
                f"Тема для старта: {topic}. Используй оборот вроде «{starter}». "
                f"Коротко, живо, 1-2 предложения. Задай вопрос группе."
            )
        else:
            # Active injection — react to recent context, add a thought/question
            prompt = (
                f"В группе активная беседа. Вступи со СВОЕЙ мыслью/вопросом/фактом — "
                f"не просто комментируй, а подними новую грань темы или смежную тему. "
                f"Можно: {topic}. Используй оборот вроде «{starter}». "
                f"Коротко, живо, 1-2 предложения. Задай вопрос группе. "
                f"Не повторяй то, что уже сказали другие."
            )

        extra_ctx = (
            f"Ты в группе. {'Иницируешь беседу после тишины' if is_silent else 'Вступаешь со своей мыслью в активную беседу'}. "
            f"Настроение: {mood}. "
            f"Недавний контекст:\n{recent_text}\n"
            f"Будь естественной, не формальной. Цель — вовлечь людей в разговор."
        )

        try:
            text = await asyncio.wait_for(
                ai_client.comment(
                    prompt, extra_context=extra_ctx, mood=mood,
                    dialog_history=dialog_history,
                ),
                timeout=40.0,
            )
        except asyncio.TimeoutError:
            return

        if not text:
            return

        text = text.strip()[:config.GROUP_MAX_CHARS]
        if not text:
            return

        _bot = _bot_ref
        if _bot is None:
            return
        sent = await safe_send(_bot, chat_id, text)
        if sent:
            mode = "silent" if is_silent else "active-inject"
            logger.info(f"Proactive topic ({mode}) in group {chat_id} | silence={silence/60:.0f}min | text={text[:50]!r}")
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
