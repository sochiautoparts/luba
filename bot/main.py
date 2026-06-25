"""
Люба Bot — Main Entry Point (@asluba_bot)

Features:
- aiogram 3.x Telegram bot framework
- LOCAL-FIRST AI: RuadaptQwen3-4B (CPU, like Asya) + Pollinations free + optional cloud providers
- Active in groups (responds to all, proactive comments) — requires Privacy Mode OFF
- Comments on channel posts
- Private 1-on-1 chats with memory
- Dynamic mood system, time/space awareness, world knowledge
- Web verification of claims (DDG-based)
- Affiliate programs from sochiautoparts.ru/partners.json
- Image understanding (vision via Pollinations free)
- SQLite (aiosqlite) persistence, WAL mode
- GitHub Actions 24/7 via self-dispatch
"""

import asyncio
import logging
import signal
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import config
from bot import database as db
from bot.partners import partner_manager
from bot.mood import mood_loop, current_mood_descriptor
from bot import site_content as site_content
from ai.router import ai_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("luba.main")
for noisy in ["aiogram.event", "httpx", "httpcore", "aiosqlite"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Routers ───────────────────────────────────────────────────────────────────
from bot.handlers.chat import chat_router
from bot.handlers.groups import group_router
from bot.handlers.channels import channel_router
from bot.handlers.admin import admin_router


class LyubaBot:
    def __init__(self):
        if not config.BOT_TOKEN:
            raise RuntimeError("BOT_TOKEN not set")
        self.bot = Bot(
            token=config.BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=None),  # plain text — Lyuba writes plain Russian
        )
        self.dp = Dispatcher(storage=MemoryStorage())
        # Order matters: admin first, then chat (private), groups, channels
        self.dp.include_router(admin_router)
        self.dp.include_router(chat_router)
        self.dp.include_router(group_router)
        self.dp.include_router(channel_router)
        self._stop = asyncio.Event()

    async def start(self) -> None:
        logger.info("=== Люба Bot стартует ===")
        logger.info(f"Bot: {config.BOT_USERNAME} (id={config.BOT_ID}), owner={config.OWNER_ID}")

        # Init DB
        await db.init_db()
        logger.info("DB initialized")

        # Init partners
        try:
            await partner_manager.load()
            logger.info(f"Partners loaded: {len(partner_manager.campaigns)} campaigns")
        except Exception as e:
            logger.warning(f"Partner load failed (non-fatal): {e}")

        # Init site content (products + posts from sochiautoparts.ru)
        try:
            await site_content.init_site_content()
            logger.info(f"Site content: {len(site_content._product_cache)} products, "
                        f"{len(site_content._post_cache)} posts cached")
        except Exception as e:
            logger.warning(f"Site content load failed (non-fatal): {e}")

        # Init AI router (loads local model)
        try:
            await ai_router.initialize()
        except Exception as e:
            logger.error(f"AI router init error (will use cloud fallback): {e}")

        # Background tasks
        asyncio.create_task(mood_loop(), name="mood_loop")
        asyncio.create_task(db.run_periodic_cleanup(), name="cleanup_loop")
        asyncio.create_task(self._site_refresh_loop(), name="site_refresh")

        # Notify owner that bot is alive
        await self._notify_owner()

        # Delete webhook & start polling
        try:
            await self.bot.delete_webhook(drop_pending_updates=False)
        except Exception as e:
            logger.warning(f"delete_webhook: {e}")

        # Allowed updates: include channel_post for channel commenting
        allowed = ["message", "edited_message", "channel_post", "edited_channel_post"]

        logger.info("=== Люба в сети — слушаю сообщения ===")
        try:
            await self.dp.start_polling(self.bot, allowed_updates=allowed)
        finally:
            await ai_router.close()

    async def _notify_owner(self) -> None:
        """Send a startup greeting to the owner so they know Lyuba is alive."""
        mood = await current_mood_descriptor()
        try:
            await self.bot.send_message(
                config.OWNER_ID,
                f"я на связи 😊 сейчас я {mood}. "
                f"локальная модель: {'✅ загружена' if ai_router._local._model_loaded else '❌ недоступна (работаю на облаке)'}, "
                f"опциональные провайдеры: {config.optional_providers() or 'нет'}. "
                f"партнёров: {len(partner_manager.campaigns)}, "
                f"товаров сайта: {len(site_content._product_cache)}, "
                f"постов: {len(site_content._post_cache)}. "
                f"пиши мне в личку или добавь в группу — буду общаться 💬"
            )
        except Exception as e:
            logger.warning(f"Could not notify owner: {e}")

    async def _site_refresh_loop(self) -> None:
        """Periodically refresh site products + posts (every hour)."""
        import asyncio as _a
        while True:
            await _a.sleep(3600)
            try:
                await site_content.refresh_products(force=True)
                await site_content.refresh_posts(force=True)
            except Exception as e:
                logger.debug(f"site refresh error: {e}")


async def main():
    bot = LyubaBot()

    def _sig(*_):
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.dp.stop_polling())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(sig, _sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: None)

    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)
