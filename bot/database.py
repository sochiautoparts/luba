"""
Люба Bot Database — SQLite with aiosqlite (WAL mode).

Tables:
  users            — known users
  chat_history     — private chat memory per user
  group_messages   — recent messages in groups (context window per group)
  group_memory     — long-term per-group + per-user facts (what Люба remembers)
  channels         — channels where Lyuba comments (enabled flag)
  moods            — Lyuba's current mood state (single row)
  ai_cache         — response cache for repeated queries
  partner_clicks   — lightweight analytics for affiliate links
"""

import aiosqlite
import json
import time
import hashlib
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from bot.config import config

DB_PATH = config.DB_PATH

_DB_BUSY_TIMEOUT = 10000
_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY = 0.5


@asynccontextmanager
async def _connect_db():
    db = await aiosqlite.connect(DB_PATH)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute(f"PRAGMA busy_timeout={_DB_BUSY_TIMEOUT}")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA cache_size=-64000")
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT DEFAULT '',
    first_name TEXT DEFAULT '',
    last_name TEXT DEFAULT '',
    language_code TEXT DEFAULT 'ru',
    is_blocked INTEGER DEFAULT 0,
    is_admin INTEGER DEFAULT 0,
    first_seen REAL DEFAULT 0,
    last_seen REAL DEFAULT 0,
    message_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id, timestamp);

CREATE TABLE IF NOT EXISTS group_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    username TEXT DEFAULT '',
    first_name TEXT DEFAULT '',
    content TEXT NOT NULL,
    is_media INTEGER DEFAULT 0,
    media_caption TEXT DEFAULT '',
    is_bot INTEGER DEFAULT 0,
    timestamp REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_group_messages_chat ON group_messages(chat_id, timestamp);

CREATE TABLE IF NOT EXISTS group_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER DEFAULT 0,
    fact TEXT NOT NULL,
    created_at REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_group_memory_chat ON group_memory(chat_id, user_id, created_at);

CREATE TABLE IF NOT EXISTS channels (
    chat_id INTEGER PRIMARY KEY,
    username TEXT DEFAULT '',
    title TEXT DEFAULT '',
    enabled INTEGER DEFAULT 1,
    first_seen REAL DEFAULT 0,
    last_commented REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS moods (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    mood TEXT DEFAULT 'спокойная',
    energy REAL DEFAULT 0.5,
    updated_at REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ai_cache (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT DEFAULT '',
    created_at REAL DEFAULT 0,
    hit_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ai_cache_query ON ai_cache(query_hash);

CREATE TABLE IF NOT EXISTS partner_clicks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_name TEXT NOT NULL,
    url TEXT NOT NULL,
    chat_id INTEGER DEFAULT 0,
    created_at REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_partner_clicks_time ON partner_clicks(created_at);
"""


async def init_db() -> None:
    async with _connect_db() as db:
        await db.executescript(SCHEMA)
        await db.commit()
        # Seed mood row
        await db.execute(
            "INSERT OR IGNORE INTO moods (id, mood, energy, updated_at) VALUES (1, 'спокойная', 0.5, ?)",
            (time.time(),),
        )
        await db.commit()


# ── Users ─────────────────────────────────────────────────────────────────────

async def get_or_create_user(user_id: int, username: str = "", first_name: str = "",
                             last_name: str = "", language_code: str = "ru") -> Dict[str, Any]:
    now = time.time()
    async with _connect_db() as db:
        await db.execute(
            """INSERT OR IGNORE INTO users
               (user_id, username, first_name, last_name, language_code, first_seen, last_seen, message_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (user_id, username, first_name, last_name, language_code, now, now),
        )
        await db.execute(
            """UPDATE users SET username=?, first_name=?, last_name=?, language_code=?,
               last_seen=?, message_count=message_count+1 WHERE user_id=?""",
            (username, first_name, last_name, language_code, now, user_id),
        )
        await db.commit()
        async with db.execute("SELECT * FROM users WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else {}


async def is_user_blocked(user_id: int) -> bool:
    async with _connect_db() as db:
        async with db.execute("SELECT is_blocked FROM users WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            return bool(row and row["is_blocked"])


# ── Chat history (private) ────────────────────────────────────────────────────

async def add_chat_message(user_id: int, role: str, content: str) -> None:
    async with _connect_db() as db:
        await db.execute(
            "INSERT INTO chat_history (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, role, content[:2000], time.time()),
        )
        await db.commit()


async def get_chat_history(user_id: int, limit: int = 8) -> List[Dict[str, str]]:
    async with _connect_db() as db:
        async with db.execute(
            "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def clear_chat_history(user_id: int) -> None:
    async with _connect_db() as db:
        await db.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))
        await db.commit()


# ── Group messages (recent context window) ────────────────────────────────────

async def add_group_message(chat_id: int, user_id: int, username: str, first_name: str,
                            content: str, is_media: bool = False, media_caption: str = "",
                            is_bot: bool = False) -> None:
    async with _connect_db() as db:
        await db.execute(
            """INSERT INTO group_messages
               (chat_id, user_id, username, first_name, content, is_media, media_caption, is_bot, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, user_id, username, first_name, content[:1500],
             1 if is_media else 0, media_caption[:500], 1 if is_bot else 0, time.time()),
        )
        await db.commit()
        # Trim to last N per group
        await db.execute(
            """DELETE FROM group_messages WHERE chat_id=? AND id NOT IN (
                 SELECT id FROM group_messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?
               )""",
            (chat_id, chat_id, config.GROUP_MEMORY_SIZE),
        )
        await db.commit()


async def get_recent_group_messages(chat_id: int, limit: int = 12) -> List[Dict[str, Any]]:
    async with _connect_db() as db:
        async with db.execute(
            "SELECT * FROM group_messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    rows = list(reversed(rows))
    return [dict(r) for r in rows]


async def last_bot_message_time(chat_id: int) -> float:
    async with _connect_db() as db:
        async with db.execute(
            "SELECT MAX(timestamp) AS t FROM group_messages WHERE chat_id=? AND is_bot=1",
            (chat_id,),
        ) as cur:
            row = await cur.fetchone()
            return row["t"] if row and row["t"] else 0.0


# ── Group memory (long-term facts) ────────────────────────────────────────────

async def add_group_memory(chat_id: int, user_id: int, fact: str) -> None:
    fact = fact.strip()
    if not fact:
        return
    async with _connect_db() as db:
        await db.execute(
            "INSERT INTO group_memory (chat_id, user_id, fact, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, user_id, fact[:500], time.time()),
        )
        await db.commit()
        # Keep max 40 facts per chat
        await db.execute(
            """DELETE FROM group_memory WHERE chat_id=? AND id NOT IN (
                 SELECT id FROM group_memory WHERE chat_id=? ORDER BY created_at DESC LIMIT 40
               )""",
            (chat_id, chat_id),
        )
        await db.commit()


async def get_group_memory(chat_id: int, user_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
    async with _connect_db() as db:
        if user_id is not None:
            q = ("SELECT * FROM group_memory WHERE chat_id=? AND user_id=? "
                 "ORDER BY created_at DESC LIMIT ?")
            params = (chat_id, user_id, limit)
        else:
            q = ("SELECT * FROM group_memory WHERE chat_id=? "
                 "ORDER BY created_at DESC LIMIT ?")
            params = (chat_id, limit)
        async with db.execute(q, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ── Channels ──────────────────────────────────────────────────────────────────

async def upsert_channel(chat_id: int, username: str = "", title: str = "") -> None:
    async with _connect_db() as db:
        await db.execute(
            """INSERT INTO channels (chat_id, username, title, enabled, first_seen, last_commented)
               VALUES (?, ?, ?, 1, ?, 0)
               ON CONFLICT(chat_id) DO UPDATE SET username=excluded.username, title=excluded.title""",
            (chat_id, username, title, time.time()),
        )
        await db.commit()


async def set_channel_enabled(chat_id: int, enabled: bool) -> None:
    async with _connect_db() as db:
        await db.execute("UPDATE channels SET enabled=? WHERE chat_id=?", (1 if enabled else 0, chat_id))
        await db.commit()


async def is_channel_enabled(chat_id: int) -> bool:
    async with _connect_db() as db:
        async with db.execute("SELECT enabled FROM channels WHERE chat_id=?", (chat_id,)) as cur:
            row = await cur.fetchone()
            if row is None:
                return True  # unknown channel → default on
            return bool(row["enabled"])


async def touch_channel_comment(chat_id: int) -> None:
    async with _connect_db() as db:
        await db.execute(
            "UPDATE channels SET last_commented=? WHERE chat_id=?", (time.time(), chat_id)
        )
        await db.commit()


async def get_channel_last_commented(chat_id: int) -> float:
    async with _connect_db() as db:
        async with db.execute("SELECT last_commented FROM channels WHERE chat_id=?", (chat_id,)) as cur:
            row = await cur.fetchone()
            return row["last_commented"] if row else 0.0


# ── Mood ──────────────────────────────────────────────────────────────────────

async def get_mood() -> Dict[str, Any]:
    async with _connect_db() as db:
        async with db.execute("SELECT * FROM moods WHERE id=1") as cur:
            row = await cur.fetchone()
            return dict(row) if row else {"mood": "спокойная", "energy": 0.5}


async def set_mood(mood: str, energy: float) -> None:
    async with _connect_db() as db:
        await db.execute(
            "UPDATE moods SET mood=?, energy=?, updated_at=? WHERE id=1",
            (mood[:60], max(0.0, min(1.0, energy)), time.time()),
        )
        await db.commit()


# ── AI cache ──────────────────────────────────────────────────────────────────

async def get_ai_cached(query_hash: str) -> Optional[str]:
    async with _connect_db() as db:
        async with db.execute("SELECT response FROM ai_cache WHERE query_hash=?", (query_hash,)) as cur:
            row = await cur.fetchone()
            if row:
                await db.execute("UPDATE ai_cache SET hit_count=hit_count+1 WHERE query_hash=?", (query_hash,))
                await db.commit()
                return row["response"]
    return None


async def set_ai_cached(query_hash: str, query: str, response: str, model: str = "") -> None:
    async with _connect_db() as db:
        await db.execute(
            """INSERT OR REPLACE INTO ai_cache (query_hash, query, response, model, created_at, hit_count)
               VALUES (?, ?, ?, ?, ?, 0)""",
            (query_hash, query[:500], response[:4000], model, time.time()),
        )
        await db.commit()


async def cleanup_old_cache(max_age_days: int = 7) -> None:
    cutoff = time.time() - max_age_days * 86400
    async with _connect_db() as db:
        await db.execute("DELETE FROM ai_cache WHERE created_at < ?", (cutoff,))
        await db.commit()


async def run_periodic_cleanup() -> None:
    """Trim old group_messages beyond memory size and old cache."""
    try:
        await cleanup_old_cache(max_age_days=7)
    except Exception:
        pass
