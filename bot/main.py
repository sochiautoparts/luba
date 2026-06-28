"""
Люба Bot v2.0 — Main Entry Point (@asluba_bot).

Перепроектировано для максимальной производительности и стабильности:

  - НЕТ локальной модели → старт за ~30с (вместо 3-5 мин на компиляцию llama-cpp)
  - AI: HuggingFace Qwen2.5-7B (0.8с) → Pollinations free (backup)
  - Singleton httpx client с connection pooling
  - Circuit breaker для каждого провайдера
  - aiogram 3.x, SQLite (aiosqlite), GitHub Actions 24/7
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
from ai import ai as ai_client

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("luba.main")
for noisy in ["aiogram.event", "httpx", "httpcore", "aiosqlite"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Routers ──────────────────────────────────────────────────────────────────
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
            default=DefaultBotProperties(parse_mode=None),  # plain text
        )
        self.dp = Dispatcher(storage=MemoryStorage())
        # Order: admin → chat (private) → groups → channels
        self.dp.include_router(admin_router)
        self.dp.include_router(chat_router)
        self.dp.include_router(group_router)
        self.dp.include_router(channel_router)
        self._stop = asyncio.Event()

        # Error handler — логируем но НЕ крашим бота
        from aiogram.types import ErrorEvent

        @self.dp.error()
        async def on_error(event: ErrorEvent):
            try:
                exc = event.exception
                from aiogram.exceptions import TelegramRetryAfter
                if isinstance(exc, TelegramRetryAfter):
                    logger.warning(
                        f"Flood control (RetryAfter {exc.retry_after}s) — handled"
                    )
                else:
                    logger.error(
                        f"Handler error (suppressed): {type(exc).__name__}: {exc}",
                        exc_info=False,
                    )
            except Exception:
                pass

    async def start(self) -> None:
        logger.info("=== Люба Bot v2.0 стартует ===")
        logger.info(
            f"Bot: {config.BOT_USERNAME} (id={config.BOT_ID}), "
            f"owner={config.OWNER_ID}"
        )

        # Init DB
        await db.init_db()
        logger.info("DB initialized")

        # Init partners
        try:
            await partner_manager.load()
            logger.info(f"Partners loaded: {len(partner_manager.campaigns)} campaigns")
        except Exception as e:
            logger.warning(f"Partner load failed (non-fatal): {e}")

        # Init site content
        try:
            await site_content.init_site_content()
            logger.info(
                f"Site content: {len(site_content._product_cache)} products, "
                f"{len(site_content._post_cache)} posts cached"
            )
        except Exception as e:
            logger.warning(f"Site content load failed (non-fatal): {e}")

        # Init AI client (singleton httpx — мгновенно, без загрузки модели)
        await ai_client.initialize()
        logger.info(f"AI client ready — {config.providers_status()}")

        # Background tasks
        asyncio.create_task(mood_loop(), name="mood_loop")
        asyncio.create_task(db.run_periodic_cleanup(), name="cleanup_loop")
        asyncio.create_task(self._site_refresh_loop(), name="site_refresh")
        # Proactive topic starter
        try:
            from bot.proactive import proactive_loop, set_bot
            set_bot(self.bot)
            asyncio.create_task(proactive_loop(), name="proactive_loop")
            logger.info("Proactive topic starter enabled")
        except Exception as e:
            logger.warning(f"Proactive loop failed to start (non-fatal): {e}")

        # Notify owner
        await self._notify_owner()

        # Delete webhook & start polling
        try:
            await self.bot.delete_webhook(drop_pending_updates=False)
        except Exception as e:
            logger.warning(f"delete_webhook: {e}")

        allowed = ["message", "edited_message", "channel_post", "edited_channel_post"]
        logger.info("=== Люба в сети — слушаю сообщения ===")

        # Robust polling: aiogram retries internally, но как safety net — outer loop
        polling_retries = 0
        while True:
            try:
                await self.dp.start_polling(self.bot, allowed_updates=allowed)
                break
            except Exception as e:
                polling_retries += 1
                logger.error(
                    f"Polling error (attempt {polling_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                if polling_retries > 50:
                    logger.error("Too many polling retries — exiting")
                    break
                wait = 5 if polling_retries <= 5 else 10
                logger.warning(f"Retrying polling in {wait}s...")
                await asyncio.sleep(wait)

        # Cleanup
        try:
            await ai_client.close()
        except Exception:
            pass

    async def _notify_owner(self) -> None:
        """Startup greeting to owner."""
        mood = await current_mood_descriptor()
        stats = ai_client.stats()
        try:
            await self.bot.send_message(
                config.OWNER_ID,
                f"я на связи 😊 сейчас я {mood}. "
                f"провайдеры: {config.providers_status()}. "
                f"партнёров: {len(partner_manager.campaigns)}, "
                f"товаров: {len(site_content._product_cache)}, "
                f"постов: {len(site_content._post_cache)}. "
                f"пиши в личку или добавь в группу 💬"
            )
        except Exception as e:
            logger.warning(f"Could not notify owner: {e}")

    async def _site_refresh_loop(self) -> None:
        """Periodically refresh site products + posts (every hour)."""
        while True:
            await asyncio.sleep(3600)
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
