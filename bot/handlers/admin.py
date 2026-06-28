"""Admin handler for Lyuba — owner-only commands."""

import logging

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot import database as db
from ai import ai as ai_client

logger = logging.getLogger("luba.admin")

admin_router = Router()


def _is_owner(message: Message) -> bool:
    return message.from_user is not None and message.from_user.id == config.OWNER_ID


@admin_router.message(Command("stats"))
async def cmd_stats(message: Message):
    if not _is_owner(message):
        return
    stats = ai_client.stats()
    lines = [
        f"📊 Статистика Любы v2.1:",
        f"Всего запросов: {stats['total']}",
        f"Статический фолбэк: {stats['static']}",
        "",
        "Провайдеры (по приоритету):",
    ]
    for name, pstats in stats["providers"].items():
        circuit = "🔴 OPEN" if pstats["circuit_open"] else "🟢 OK"
        lines.append(
            f"  {name:12} | ok={pstats['ok']:4} | errors={pstats['errors']} | {circuit}"
        )
    await message.answer("\n".join(lines))


@admin_router.message(Command("channel_on"))
async def cmd_channel_on(message: Message):
    if not _is_owner(message):
        return
    cid = message.text.split(maxsplit=1)
    if len(cid) < 2:
        await message.answer("Использование: /channel_on <chat_id>")
        return
    try:
        chat_id = int(cid[1])
        await db.set_channel_enabled(chat_id, True)
        await message.answer(f"✅ Комментарии для канала {chat_id} включены")
    except ValueError:
        await message.answer("chat_id должен быть числом")


@admin_router.message(Command("channel_off"))
async def cmd_channel_off(message: Message):
    if not _is_owner(message):
        return
    cid = message.text.split(maxsplit=1)
    if len(cid) < 2:
        await message.answer("Использование: /channel_off <chat_id>")
        return
    try:
        chat_id = int(cid[1])
        await db.set_channel_enabled(chat_id, False)
        await message.answer(f"🚫 Комментарии для канала {chat_id} выключены")
    except ValueError:
        await message.answer("chat_id должен быть числом")


@admin_router.message(Command("broadcast"))
async def cmd_broadcast(message: Message):
    """Owner-only: send a message to a chat. Usage: /broadcast <chat_id> <text>"""
    if not _is_owner(message):
        return
    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        await message.answer("Использование: /broadcast <chat_id> <текст>")
        return
    try:
        chat_id = int(parts[1])
        await message.bot.send_message(chat_id, parts[2])
        await message.answer("✅ отправлено")
    except Exception as e:
        await message.answer(f"❌ ошибка: {e}")


@admin_router.message(Command("reload_partners"))
async def cmd_reload_partners(message: Message):
    if not _is_owner(message):
        return
    from bot.partners import partner_manager
    await partner_manager.load(force=True)
    await message.answer(f"✅ Партнёры перезагружены: {len(partner_manager.campaigns)} программ")
