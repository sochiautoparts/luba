"""
Media Handler for Lyuba — downloads photos/media from Telegram and prepares
them for vision analysis (base64 data URI, avoids exposing bot token).

Also extracts captions and detects media types.
"""

import base64
import logging
from typing import Optional, Tuple

import httpx

logger = logging.getLogger("luba.media")


async def get_photo_data_uri(bot, photo) -> Optional[str]:
    """Download the largest photo size and return a base64 data URI.

    Args:
        bot: aiogram Bot instance
        photo: list of PhotoSize (message.photo)

    Returns: data URI string like "data:image/jpeg;base64,...." or None on failure.
    """
    if not photo:
        return None
    # Pick largest
    largest = max(photo, key=lambda p: (p.width or 0) * (p.height or 0))
    try:
        file = await bot.get_file(largest.file_id)
        if not file.file_path:
            return None
        # Download via bot session (uses bot token internally, no URL exposure)
        downloaded = await bot.download_file(file.file_path)
        data = downloaded.read()
        if not data:
            return None
        # Determine mime
        mime = "image/jpeg"
        if file.file_path.lower().endswith(".png"):
            mime = "image/png"
        elif file.file_path.lower().endswith(".webp"):
            mime = "image/webp"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.error(f"Photo download failed: {e}")
        return None


async def get_document_data_uri(bot, document) -> Optional[Tuple[str, str]]:
    """Download an image document. Returns (data_uri, mime) or None."""
    if not document:
        return None
    mime = document.mime_type or ""
    if not mime.startswith("image/"):
        return None
    try:
        file = await bot.get_file(document.file_id)
        if not file.file_path:
            return None
        downloaded = await bot.download_file(file.file_path)
        data = downloaded.read()
        if not data:
            return None
        b64 = base64.b64encode(data).decode("ascii")
        return (f"data:{mime};base64,{b64}", mime)
    except Exception as e:
        logger.error(f"Document download failed: {e}")
        return None


def extract_caption(message) -> str:
    """Get caption from photo/document message, or empty string."""
    return (message.caption or "").strip()
