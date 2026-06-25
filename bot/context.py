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
    """Is this message directed at Lyuba (mention, reply to her, or private)?"""
    if message.chat.type == "private":
        return True
    text = message.text or message.caption or ""
    handle = config.BOT_HANDLE.lower()
    if f"@{handle}" in text.lower():
        return True
    if f"люба" in text.lower() or "любаша" in text.lower():
        return True
    # Reply to Lyuba's message
    if message.reply_to_message:
        rep = message.reply_to_message
        if rep.from_user and rep.from_user.id == config.BOT_ID:
            return True
        if rep.from_user and rep.from_user.is_bot:
            # reply to a bot message that might be Lyuba's
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
    """Build the extra_context string for a group interaction."""
    who = user_descriptor(message.from_user)
    where = chat_descriptor(message.chat)
    directed = is_directed_at_lyuba(message)

    parts = [
        f"ГДЕ: {where}.",
        f"КТО ПИШЕТ: {who}.",
    ]
    if directed:
        parts.append("ОБРАЩЕНИЕ: это сообщение адресовано тебе (упомянули / ответили / личка).")
    else:
        parts.append("ОБРАЩЕНИЕ: это сообщение НЕ тебе лично — ты комментируешь по желанию, коротко и к месту.")
    if recent_text:
        parts.append(f"НЕДАВНИЕ СООБЩЕНИЯ В ЭТОМ ЧАТЕ (для контекста):\n{recent_text}")
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
