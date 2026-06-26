"""
Context Builder for Lyuba — assembles the "who / what / where" context
that is injected into the AI system prompt before generating a response.

Understands:
  - WHO is writing (name, username, whether it's the owner, whether it's a bot)
  - WHERE (private chat, group, supergroup, channel — and which one)
  - WHAT (the message, who it replies to, what post is being commented)
  - Recent group memory (recent messages + long-term facts)
  - Whether the message is directed at Lyuba (mention / reply / command)
"""

import logging
from typing import Optional, Dict, Any, List

from aiogram.types import Message, Chat, User
from bot.config import config

logger = logging.getLogger("luba.context")


def user_descriptor(user: Optional[User]) -> str:
    """Human-readable descriptor of who is speaking."""
    if not user:
        return "кто-то (не видно кто)"
    parts = []
    name = (user.first_name or "").strip()
    if user.last_name:
        name += " " + user.last_name.strip()
    if not name and user.username:
        name = "@" + user.username
    if not name:
        name = "аноним"
    parts.append(name)
    if user.username:
        parts.append(f"(@{user.username})")
    if user.id == config.OWNER_ID:
        parts.append("[владелец бота]")
    if user.is_bot:
        parts.append("[бот]")
    return " ".join(parts)


def chat_descriptor(chat: Chat) -> str:
    """Where the conversation is happening."""
    ctype = chat.type
    title = chat.title or ""
    uname = chat.username or ""
    if ctype == "private":
        return "личный чат (один на один с тобой)"
    if ctype in ("group", "supergroup"):
        where = title or uname or "группа"
        return f"группа/супергруппа «{where}»"
    if ctype == "channel":
        return f"канал «{title or uname}»"
    return ctype


def is_directed_at_lyuba(message: Message) -> bool:
    """Is this message directed at Lyuba (mention, reply to her, or private)?

    CRITICAL: This function is called AFTER _should_respond already filters
    out Lyuba's own messages (message.from_user.id == BOT_ID). So we never
    check directed on Lyuba's own messages here. But we still guard against
    self-reference: don't treat "Люба" in Lyuba's OWN text as a directed
    mention (she often says "я Люба" in her responses).
    """
    if message.chat.type == "private":
        return True
    text = message.text or message.caption or ""
    handle = config.BOT_HANDLE.lower()
    # Only check for @mention — NOT the bare word "Люба" (which appears in
    # Lyuba's own responses like "я Люба из Сочи" and causes self-reply loops).
    if f"@{handle}" in text.lower():
        return True
    # Reply to Lyuba's message (by user ID, not is_bot flag — is_bot would
    # match ANY bot, causing false positives when other bots are in the chat)
    if message.reply_to_message:
        rep = message.reply_to_message
        if rep.from_user and rep.from_user.id == config.BOT_ID:
            return True
    return False


def strip_mention(text: str) -> str:
    """Remove the @asluba_bot mention from text for cleaner AI input."""
    if not text:
        return text
    handle = config.BOT_HANDLE
    s = text.replace(f"@{handle}", "").replace(f"@{handle.lower()}", "")
    return s.strip()


def recent_messages_to_text(messages: List[Dict[str, Any]], limit: int = 10) -> str:
    """Format recent group messages into a readable context block."""
    if not messages:
        return ""
    lines = []
    for m in messages[-limit:]:
        who = m.get("first_name") or m.get("username") or "кто-то"
        if m.get("is_bot"):
            who = "Люба" if m.get("user_id") == config.BOT_ID else f"{who} (бот)"
        content = m.get("content", "")
        if m.get("is_media"):
            cap = m.get("media_caption", "")
            content = f"[фото/медиа{': ' + cap if cap else ''}]"
        if content:
            lines.append(f"{who}: {content}")
    return "\n".join(lines)


def build_group_context(message: Message, recent_text: str, memory_facts: List[str]) -> str:
    """Build the extra_context string for a group interaction.

    Includes WHO Lyuba is replying to (the reply target), so she can address
    them by name and continue a threaded conversation naturally.
    Lyuba is taught to actively respond to OTHER participants — not just when
    addressed directly, but joining discussions and addressing people by name.
    """
    who = user_descriptor(message.from_user)
    where = chat_descriptor(message.chat)
    directed = is_directed_at_lyuba(message)

    # Extract the speaker's first name for natural addressing
    speaker_name = ""
    if message.from_user:
        speaker_name = (message.from_user.first_name or "").strip()
        if not speaker_name and message.from_user.username:
            speaker_name = message.from_user.username

    parts = [
        f"ГДЕ: {where}.",
        f"КТО ПИШЕТ: {who}.",
    ]

    # Detect if this message is a REPLY to another user — Lyuba should
    # understand the conversation thread and address the right person.
    reply_target = None
    replied_to_name = ""
    if message.reply_to_message:
        rep = message.reply_to_message
        rep_author = rep.from_user
        if rep_author:
            if rep_author.id == config.BOT_ID:
                reply_target = "ответ на твоё (Любы) предыдущее сообщение"
                replied_to_name = "Люба"
            else:
                rn = (rep_author.first_name or "").strip()
                if rep_author.last_name:
                    rn += " " + rep_author.last_name.strip()
                if rep_author.username:
                    rn += f" (@{rep_author.username})"
                replied_to_name = (rep_author.first_name or "").strip() or rep_author.username or ""
                reply_target = f"ответ на сообщение пользователя {rn}"
        else:
            reply_target = "ответ на сообщение от анонима/канала"
        # Include the replied-to text so Lyuba understands the thread
        rep_text = (rep.text or rep.caption or "").strip()
        if rep_text:
            parts.append(f"НА ЧТО ОТВЕЧАЮТ (цитата): {rep_text[:400]}")
    if reply_target:
        parts.append(f"ПРОТЕЖКА: это {reply_target}. Пойми контекст треда и ответь уместно.")

    if directed:
        parts.append(
            "ОБРАЩЕНИЕ: это сообщение адресовано тебе. Отвечай адресно — "
            f"обратись к {speaker_name} по имени если уместно. Веди живой диалог."
        )
    else:
        # Proactive: Lyuba joins the discussion and addresses the speaker
        addr_hint = f" Можешь обратиться к {speaker_name} по имени." if speaker_name else ""
        parts.append(
            "ОБРАЩЕНИЕ: ты вступаешь в беседу АКТИВНО. Не жди пока обратятся — "
            "отвечай на сообщения участников, комментируй что они написали, "
            "задавай вопросы, соглашайся или спорь (доброжелательно)." + addr_hint +
            " Это живой чат — общайся как активная участница."
        )
    if recent_text:
        parts.append(f"НЕДАВНИЕ СООБЩЕНИЯ В ЭТОМ ЧАТЕ (контекст беседы, видишь кто что сказал):\n{recent_text}")
    if memory_facts:
        parts.append("ЧТО ТЫ ПОМНИШЬ ОБ ЭТОМ ЧАТЕ/ЛЮДЯХ:\n" + "\n".join(f"- {f}" for f in memory_facts[:6]))
    return "\n\n".join(parts)


def build_channel_context(channel_chat: Chat, post_text: str, post_author: Optional[User]) -> str:
    where = chat_descriptor(channel_chat)
    author = user_descriptor(post_author) if post_author else "автор поста (канал)"
    parts = [
        f"ГДЕ: {where} — ты комментируешь пост канала.",
        f"АВТОР ПОСТА: {author}.",
    ]
    if post_text:
        parts.append(f"ТЕКСТ ПОСТА:\n{post_text[:1500]}")
    parts.append("ЗАДАЧА: напиши короткий живой комментарий к этому посту (1-3 предложения). "
                 "Как живой подписчик, которому интересно. Без политики и войны.")
    return "\n\n".join(parts)


def build_private_context(message: Message) -> str:
    who = user_descriptor(message.from_user)
    return f"ГДЕ: личный чат. КТО ПИШЕТ: {who}."
