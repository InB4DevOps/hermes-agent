#!/usr/bin/env python3
"""
SQLite State Store for Hermes Agent.

Provides persistent session storage with FTS5 full-text search, replacing
the per-session JSONL file approach. Stores session metadata, full message
history, and model configuration for CLI and gateway sessions.

Key design decisions:
- WAL mode for concurrent readers + one writer (gateway multi-platform)
- FTS5 virtual table for fast text search across all session messages
- Compression-triggered session splitting via parent_session_id chains
- Batch runner and RL trajectories are NOT stored here (separate systems)
- Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
"""

import json
import logging
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

# ============================================================================
# Constants for magic numbers
# ============================================================================
# Maximum character length for message content before it is classified as
# ephemeral.  This captures large file dumps, long terminal outputs, and
# verbose tool responses that bloat context but are safe to discard on
# /clear-context. 5000 chars ≈ 1250 tokens — well above a typical user
# prompt or short assistant reply, but short enough to catch file contents.
MAX_EPHEMERAL_CONTENT_LENGTH = 5000

# Default number of most-recent non-system messages to remove when the user
# runs /clear-context recent.  10 was chosen as a pragmatic balance: enough
# to reclaim meaningful context space (usually 2-5 tool call rounds) without
# blowing away the entire conversation.
DEFAULT_PRUNE_COUNT = 10

# LIKE patterns used in SQL candidate filters to identify ephemeral messages.
# Extracted as constants so they are testable, adjustable, and documented in
# one place — rather than embedded in raw SQL strings.
_EPHEMERAL_SQL_PATTERNS: tuple[str, ...] = (
    '%stdout:%',
    '%\nstdout:%',
    '%exit_code:%',
    '%\nexit_code:%',
    '%Error:%',
    '%Exception:%',
)
# Combined condition for use in SQL WHERE clauses (OR'd together).
_EPHEMERAL_LIKE_CONDITIONS = " OR ".join(
    f"content LIKE ? " for _ in _EPHEMERAL_SQL_PATTERNS
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

SCHEMA_VERSION = 10

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    task_protection_active INTEGER DEFAULT 0,
    task_definition_mode INTEGER DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    task_id TEXT,
    is_ephemeral INTEGER DEFAULT 0,
    protected INTEGER DEFAULT 0,
    task_definition INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class SessionDB:
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """

    # ── Write-contention tuning ──
    # With multiple hermes processes (gateway + CLI sessions + worktree agents)
    # all sharing one state.db, WAL write-lock contention causes visible TUI
    # freezes.  SQLite's built-in busy handler uses a deterministic sleep
    # schedule that causes convoy effects under high concurrency.
    #
    # Instead, we keep the SQLite timeout short (1s) and handle retries at the
    # application level with random jitter, which naturally staggers competing
    # writers and avoids the convoy.
    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    # Attempt a PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._write_count = 0
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            # Short timeout — application-level retry with random jitter
            # handles contention instead of sitting in SQLite's internal
            # busy handler for up to 30s.
            timeout=1.0,
            # Autocommit mode: Python's default isolation_level="" auto-starts
            # transactions on DML, which conflicts with our explicit
            # BEGIN IMMEDIATE.  None = we manage transactions ourselves.
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()

    # ── Core write helper ──

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random 20-150ms, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        Returns whatever *fn* returns.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                # Success — periodic best-effort checkpoint.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(
                            self._WRITE_RETRY_MIN_S,
                            self._WRITE_RETRY_MAX_S,
                        )
                        time.sleep(jitter)
                        continue
                # Non-lock error or retries exhausted — propagate.
                raise
        # Retries exhausted (shouldn't normally reach here).
        raise last_err or sqlite3.OperationalError(
            "database is locked after max retries"
        )

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint.  Never blocks, never raises.

        Flushes committed WAL frames back into the main DB file for any
        frames that no other connection currently needs.  Keeps the WAL
        from growing unbounded when many processes hold persistent
        connections.
        """
        try:
            with self._lock:
                result = self._conn.execute(
                    "PRAGMA wal_checkpoint(PASSIVE)"
                ).fetchone()
                if result and result[1] > 0:
                    logger.debug(
                        "WAL checkpoint: %d/%d pages checkpointed",
                        result[2], result[1],
                    )
        except Exception:
            pass  # Best effort — never fatal.

    def close(self):
        """Close the database connection.

        Attempts a PASSIVE WAL checkpoint first so that exiting processes
        help keep the WAL file from growing unbounded.
        """
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    def _init_schema(self):
        """Create tables and FTS if they don't exist, run migrations."""
        cursor = self._conn.cursor()

        cursor.executescript(SCHEMA_SQL)

        # Check schema version and run migrations
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        else:
            current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
            if current_version < 2:
                # v2: add finish_reason column to messages
                try:
                    cursor.execute("ALTER TABLE messages ADD COLUMN finish_reason TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 2")
            if current_version < 3:
                # v3: add title column to sessions
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 3")
            if current_version < 4:
                # v4: add unique index on title (NULLs allowed, only non-NULL must be unique)
                try:
                    cursor.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                        "ON sessions(title) WHERE title IS NOT NULL"
                    )
                except sqlite3.OperationalError:
                    pass  # Index already exists
                cursor.execute("UPDATE schema_version SET version = 4")
            if current_version < 5:
                new_columns: List[Tuple[str, str]] = [
                    ("cache_read_tokens", "INTEGER DEFAULT 0"),
                    ("cache_write_tokens", "INTEGER DEFAULT 0"),
                    ("reasoning_tokens", "INTEGER DEFAULT 0"),
                    ("billing_provider", "TEXT"),
                    ("billing_base_url", "TEXT"),
                    ("billing_mode", "TEXT"),
                    ("estimated_cost_usd", "REAL"),
                    ("actual_cost_usd", "REAL"),
                    ("cost_status", "TEXT"),
                    ("cost_source", "TEXT"),
                    ("pricing_version", "TEXT"),
                ]
                for name, column_type in new_columns:
                    try:
                        # name and column_type come from the hardcoded tuple above,
                        # not user input. Double-quote identifier escaping is applied
                        # as defense-in-depth; SQLite DDL cannot be parameterized.
                        safe_name: str = name.replace('"', '""')
                        cursor.execute(f'ALTER TABLE sessions ADD COLUMN "{safe_name}" {column_type}')
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Failed to add {name} column: {e}")
                cursor.execute("UPDATE schema_version SET version = 5")
            if current_version < 6:
                # v6: add reasoning columns to messages table — preserves assistant
                # reasoning text and structured reasoning_details across gateway
                # session turns.  Without these, reasoning chains are lost on
                # session reload, breaking multi-turn reasoning continuity for
                # providers that replay reasoning (OpenRouter, OpenAI, Nous).
                for col_name, col_type in [
                    ("reasoning", "TEXT"),
                    ("reasoning_details", "TEXT"),
                    ("codex_reasoning_items", "TEXT"),
                ]:
                    try:
                        safe = col_name.replace('"', '""')
                        cursor.execute(
                            f'ALTER TABLE messages ADD COLUMN "{safe}" {col_type}'
                        )
                    except sqlite3.OperationalError:
                        pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 6")
            if current_version < 7:
                # v7: add task_id and is_ephemeral columns for context management
                # task_id allows marking messages as belonging to a specific task
                # is_ephemeral marks messages that can be safely cleared (tool outputs, file contents)
                for col_name, col_type in [
                    ("task_id", "TEXT"),
                    ("is_ephemeral", "INTEGER DEFAULT 0"),
                ]:
                    try:
                        safe = col_name.replace('"', '""')
                        cursor.execute(
                            f'ALTER TABLE messages ADD COLUMN "{safe}" {col_type}'
                        )
                    except sqlite3.OperationalError:
                        pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 7")
            if current_version < 8:
                # v8: add protected column for task protection
                # Messages added while task_protection_active=1 on the session are
                # automatically marked as protected, surviving /clear-context
                try:
                    cursor.execute('ALTER TABLE messages ADD COLUMN "protected" INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 8")
            if current_version < 9:
                # v9: add task_protection_active to sessions for implicit tracking
                try:
                    cursor.execute('ALTER TABLE sessions ADD COLUMN "task_protection_active" INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 9")
            if current_version < 10:
                # v10: add task_definition_mode to sessions and task_definition to messages
                # for task train feature (/start-tasks ... /end-tasks with auto-clear)
                try:
                    cursor.execute('ALTER TABLE sessions ADD COLUMN "task_definition_mode" INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    pass  # Column already exists
                try:
                    cursor.execute('ALTER TABLE messages ADD COLUMN "task_definition" INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 10")

        # Unique title index — always ensure it exists (safe to run after migrations
        # since the title column is guaranteed to exist at this point)
        try:
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                "ON sessions(title) WHERE title IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass  # Index already exists

        # FTS5 setup (separate because CREATE VIRTUAL TABLE can't be in executescript with IF NOT EXISTS reliably)
        try:
            cursor.execute("SELECT * FROM messages_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)

        self._conn.commit()

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        parent_session_id: str = None,
    ) -> str:
        """Create a new session record. Returns the session_id."""
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config,
                   system_prompt, parent_session_id, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    user_id,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt,
                    parent_session_id,
                    time.time(),
                ),
            )
        self._execute_write(_do)
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                (time.time(), end_reason, session_id),
            )
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                (system_prompt, session_id),
            )
        self._execute_write(_do)

    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        absolute: bool = False,
    ) -> None:
        """Update token counters and backfill model if not already set.

        When *absolute* is False (default), values are **incremented** — use
        this for per-API-call deltas (CLI path).

        When *absolute* is True, values are **set directly** — use this when
        the caller already holds cumulative totals (gateway path, where the
        cached agent accumulates across messages).
        """
        if absolute:
            sql = """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?)
                   WHERE id = ?"""
        else:
            sql = """UPDATE sessions SET
                   input_tokens = input_tokens + ?,
                   output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?)
                   WHERE id = ?"""
        params = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            actual_cost_usd,
            actual_cost_usd,
            cost_status,
            cost_source,
            pricing_version,
            billing_provider,
            billing_base_url,
            billing_mode,
            model,
            session_id,
        )
        def _do(conn):
            conn.execute(sql, params)
        self._execute_write(_do)

    def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
    ) -> None:
        """Ensure a session row exists, creating it with minimal metadata if absent.

        Used by _flush_messages_to_session_db to recover from a failed
        create_session() call (e.g. transient SQLite lock at agent startup).
        INSERT OR IGNORE is safe to call even when the row already exists.
        """
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO sessions
                   (id, source, model, started_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, source, model, time.time()),
            )
        self._execute_write(_do)

    def set_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
    ) -> None:
        """Set token counters to absolute values (not increment).

        Use this when the caller provides cumulative totals from a completed
        conversation run (e.g. the gateway, where the cached agent's
        session_prompt_tokens already reflects the running total).
        """
        def _do(conn):
            conn.execute(
                """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = ?,
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?)
                   WHERE id = ?""",
                (
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    reasoning_tokens,
                    estimated_cost_usd,
                    actual_cost_usd,
                    actual_cost_usd,
                    cost_status,
                    cost_source,
                    pricing_version,
                    billing_provider,
                    billing_base_url,
                    billing_mode,
                    model,
                    session_id,
                ),
            )
        self._execute_write(_do)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session ID to the full ID.

        Returns the exact ID when it exists. Otherwise treats the input as a
        prefix and returns the single matching session ID if the prefix is
        unambiguous. Returns None for no matches or ambiguous prefixes.
        """
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]

        escaped = (
            session_id_or_prefix
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            )
            matches = [row["id"] for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        return None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None

        # Remove ASCII control characters (0x00-0x1F, 0x7F) but keep
        # whitespace chars (\t=0x09, \n=0x0A, \r=0x0D) so they can be
        # normalized to spaces by the whitespace collapsing step below
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', title)

        # Remove problematic Unicode control characters:
        # - Zero-width chars (U+200B-U+200F, U+FEFF)
        # - Directional overrides (U+202A-U+202E, U+2066-U+2069)
        # - Object replacement (U+FFFC), interlinear annotation (U+FFF9-U+FFFB)
        cleaned = re.sub(
            r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]',
            '', cleaned,
        )

        # Collapse internal whitespace runs and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            return None

        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )

        return cleaned

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).
        """
        title = self.sanitize_title(title)
        def _do(conn):
            if title:
                # Check uniqueness (allow the same session to keep its own title)
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE title = ? AND id != ?",
                    (title, session_id),
                )
                conflict = cursor.fetchone()
                if conflict:
                    raise ValueError(
                        f"Title '{title}' is already in use by session {conflict['id']}"
                    )
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            return cursor.rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return row["title"] if row else None

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE title = ?", (title,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest in a lineage.

        If the exact title exists, returns that session's ID.
        If not, searches for "title #N" variants and returns the latest one.
        If the exact title exists AND numbered variants exist, returns the
        latest numbered variant (the most recent continuation).
        """
        # First try exact match
        exact = self.get_session_by_title(title)

        # Also search for numbered variants: "title #2", "title #3", etc.
        # Escape SQL LIKE wildcards (%, _) in the title to prevent false matches
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, title, started_at FROM sessions "
                "WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
                (f"{escaped} #%",),
            )
            numbered = cursor.fetchall()

        if numbered:
            # Return the most recent numbered variant
            return numbered[0]["id"]
        elif exact:
            return exact["id"]
        return None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Generate the next title in a lineage (e.g., "my session" → "my session #2").

        Strips any existing " #N" suffix to find the base name, then finds
        the highest existing number and increments.
        """
        # Strip existing #N suffix to find the true base
        match = re.match(r'^(.*?) #(\d+)$', base_title)
        if match:
            base = match.group(1)
        else:
            base = base_title

        # Find all existing numbered variants
        # Escape SQL LIKE wildcards (%, _) in the base to prevent false matches
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
                (base, f"{escaped} #%"),
            )
            existing = [row["title"] for row in cursor.fetchall()]

        if not existing:
            return base  # No conflict, use the base name as-is

        # Find the highest number
        max_num = 1  # The unnumbered original counts as #1
        for t in existing:
            m = re.match(r'^.* #(\d+)$', t)
            if m:
                max_num = max(max_num, int(m.group(1)))

        return f"{base} #{max_num + 1}"

    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.

        By default, child sessions (subagent runs, compression continuations)
        are excluded.  Pass ``include_children=True`` to include them.
        """
        where_clauses = []
        params = []

        if not include_children:
            where_clauses.append("s.parent_session_id IS NULL")

        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            {where_sql}
            ORDER BY s.started_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        sessions = []
        for row in rows:
            s = dict(row)
            # Build the preview from the raw substring
            raw = s.pop("_preview_raw", "").strip()
            if raw:
                text = raw[:60]
                s["preview"] = text + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            sessions.append(s)

        return sessions

    # =========================================================================
    # Message storage
    # =========================================================================

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        protected: bool = False,
    ) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).

        Args:
            protected: When True, the message is excluded from clear-context
                and prune operations. Always set to True when the session's
                task_protection_active flag is 1 (implicit protection).
        """
        # Serialize structured fields to JSON before entering the write txn
        reasoning_details_json = (
            json.dumps(reasoning_details)
            if reasoning_details else None
        )
        codex_items_json = (
            json.dumps(codex_reasoning_items)
            if codex_reasoning_items else None
        )
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        # Pre-compute tool call count
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1

        def _do(conn):
            # Check if task protection is active for this session — if so,
            # auto-protect all new messages regardless of the protected param.
            is_protected = 1 if protected else 0
            if not is_protected:
                cursor = conn.execute(
                    "SELECT task_protection_active FROM sessions WHERE id = ?",
                    (session_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    is_protected = 1

            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, timestamp, token_count, finish_reason,
                   reasoning, reasoning_details, codex_reasoning_items, protected)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    time.time(),
                    token_count,
                    finish_reason,
                    reasoning,
                    reasoning_details_json,
                    codex_items_json,
                    is_protected,
                ),
            )
            msg_id = cursor.lastrowid

            # Update counters
            if num_tool_calls > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return msg_id

        return self._execute_write(_do)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages for a session, ordered by timestamp."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            )
            rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(msg)
        return result

    def get_messages_as_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT role, content, tool_call_id, tool_calls, tool_name, "
                "reasoning, reasoning_details, codex_reasoning_items "
                "FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            )
            rows = cursor.fetchall()
        messages = []
        for row in rows:
            msg = {"role": row["role"], "content": row["content"]}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        pass
            messages.append(msg)
        return messages

    # =========================================================================
    # Search
    # =========================================================================

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}`` and bare boolean operators (``AND``, ``OR``,
        ``NOT``) have special meaning.  Passing raw user input directly to
        MATCH can cause ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders.
        _quoted_parts: list = []

        def _preserve_quoted(m: re.Match) -> str:
            _quoted_parts.append(m.group(0))
            return f"\x00Q{len(_quoted_parts) - 1}\x00"

        sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)

        # Step 2: Strip remaining (unmatched) FTS5-special characters
        sanitized = re.sub(r'[+{}()\"^]', " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted dotted and/or hyphenated terms in double
        # quotes.  FTS5's tokenizer splits on dots and hyphens, turning
        # ``chat-send`` into ``chat AND send`` and ``P2.2`` into ``p2 AND 2``.
        # Quoting preserves phrase semantics.  A single pass avoids the
        # double-quoting bug that would occur if dotted and hyphenated
        # patterns were applied sequentially (e.g. ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[.-]\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across session messages using FTS5.

        Supports FTS5 query syntax:
          - Simple keywords: "docker deployment"
          - Phrases: '"exact phrase"'
          - Boolean: "docker OR kubernetes", "python NOT java"
          - Prefix: "deploy*"

        Returns matching messages with session metadata, content snippet,
        and surrounding context (1 message before and after the match).
        """
        if not query or not query.strip():
            return []

        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        # Build WHERE clauses dynamically
        where_clauses = ["messages_fts MATCH ?"]
        params: list = [query]

        if source_filter is not None:
            source_placeholders = ",".join("?" for _ in source_filter)
            where_clauses.append(f"s.source IN ({source_placeholders})")
            params.extend(source_filter)

        if exclude_sources is not None:
            exclude_placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({exclude_placeholders})")
            params.extend(exclude_sources)

        if role_filter:
            role_placeholders = ",".join("?" for _ in role_filter)
            where_clauses.append(f"m.role IN ({role_placeholders})")
            params.extend(role_filter)

        where_sql = " AND ".join(where_clauses)
        params.extend([limit, offset])

        sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {where_sql}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """

        with self._lock:
            try:
                cursor = self._conn.execute(sql, params)
            except sqlite3.OperationalError:
                # FTS5 query syntax error despite sanitization — return empty
                return []
            matches = [dict(row) for row in cursor.fetchall()]

        # Add surrounding context (1 message before + after each match).
        # Done outside the lock so we don't hold it across N sequential queries.
        for match in matches:
            try:
                with self._lock:
                    ctx_cursor = self._conn.execute(
                        """SELECT role, content FROM messages
                           WHERE session_id = ? AND id >= ? - 1 AND id <= ? + 1
                           ORDER BY id""",
                        (match["session_id"], match["id"], match["id"]),
                    )
                    context_msgs = [
                        {"role": r["role"], "content": (r["content"] or "")[:200]}
                        for r in ctx_cursor.fetchall()
                    ]
                match["context"] = context_msgs
            except Exception:
                match["context"] = []

        # Remove full content from result (snippet is enough, saves tokens)
        for match in matches:
            match.pop("content", None)

        return matches

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source."""
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    "SELECT * FROM sessions WHERE source = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (source, limit, offset),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(self, source: str = None) -> int:
        """Count sessions, optionally filtered by source."""
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE source = ?", (source,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    # =========================================================================
    # Export and cleanup
    # =========================================================================

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session, its child sessions, and all their messages.

        Child sessions (subagent runs, compression continuations) are deleted
        first to satisfy the ``parent_session_id`` foreign key constraint.
        Returns True if the session was found and deleted.
        """
        def _do(conn):
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
            )
            if cursor.fetchone()[0] == 0:
                return False
            # Delete child sessions first (FK constraint)
            child_ids = [r[0] for r in conn.execute(
                "SELECT id FROM sessions WHERE parent_session_id = ?",
                (session_id,),
            ).fetchall()]
            for cid in child_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (cid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (cid,))
            # Delete the session itself
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True
        return self._execute_write(_do)

    def prune_sessions(self, older_than_days: int = 90, source: str = None) -> int:
        """Delete sessions older than N days. Returns count of deleted sessions.

        Only prunes ended sessions (not active ones).  Child sessions whose
        parents are being pruned are deleted first to satisfy the
        ``parent_session_id`` foreign key constraint.
        """
        cutoff = time.time() - (older_than_days * 86400)

        def _do(conn):
            if source:
                cursor = conn.execute(
                    """SELECT id FROM sessions
                       WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?""",
                    (cutoff, source),
                )
            else:
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE started_at < ? AND ended_at IS NOT NULL",
                    (cutoff,),
                )
            session_ids = set(row["id"] for row in cursor.fetchall())

            # Delete children first whose parents are in the prune set
            # (avoids FK constraint errors)
            for sid in list(session_ids):
                child_ids = [r[0] for r in conn.execute(
                    "SELECT id FROM sessions WHERE parent_session_id = ?",
                    (sid,),
                ).fetchall()]
                for cid in child_ids:
                    conn.execute("DELETE FROM messages WHERE session_id = ?", (cid,))
                    conn.execute("DELETE FROM sessions WHERE id = ?", (cid,))
                    session_ids.discard(cid)  # don't double-delete

            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            return len(session_ids)

        return self._execute_write(_do)

    # =========================================================================
    # Context management (clear-context)
    # =========================================================================

    @staticmethod
    def is_ephemeral_message(msg: Dict[str, Any]) -> bool:
        """Determine if a message can be safely cleared.

        Ephemeral messages include:
        - Tool results (role='tool')
        - Long file contents (>5000 chars)
        - Terminal outputs (contain stdout/exit_code)
        - Error traces (unless critical)
        - Memory retrieval results

        Args:
            msg: A message dict with 'role' and 'content' keys

        Returns:
            True if the message should be cleared by /clear-context
        """
        content: str = msg.get("content", "") or ""
        role: str = msg.get("role", "")

        # Tool results are ephemeral
        if role == "tool":
            return True

        # Long file contents (likely from file_tools)
        if len(content) > MAX_EPHEMERAL_CONTENT_LENGTH:
            return True

        # Terminal outputs — require a structured pattern to avoid false
        # positives on casual mentions like "I'm parsing stdout" or
        # "the exit_code was 0".  We look for key/value-style formatting
        # that matches real tool output formats:
        #   JSON-style:   "stdout": ... / "exit_code": ...
        #   Line-start:   stdout: value / exit_code: value
        if re.search(r'"stdout"', content) or re.search(r'"exit_code"', content):
            return True
        if re.search(r'(?:^|\n)\s*stdout\s*:', content):
            return True
        if re.search(r'(?:^|\n)\s*exit_code\s*:', content):
            return True

        # Error traces (check if critical)
        if "Error:" in content or "Exception:" in content:
            # Skip critical errors that affect task execution
            if not any(crit in content.lower() for crit in ["fatal", "critical", "cannot", "failed to create"]):
                return True

        # Memory retrieval results
        if "memory" in content.lower() and ("retrieved" in content.lower() or "found" in content.lower()):
            return True

        return False

    def classify_messages(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Classify all messages into persistent and ephemeral.

        Args:
            session_id: The session to classify

        Returns:
            Dict with 'persistent' and 'ephemeral' lists of messages
        """
        messages = self.get_messages(session_id)
        persistent: List[Dict[str, Any]] = []
        ephemeral: List[Dict[str, Any]] = []

        for msg in messages:
            if self.is_ephemeral_message(msg):
                ephemeral.append(msg)
            else:
                persistent.append(msg)

        return {"persistent": persistent, "ephemeral": ephemeral}

    def clear_ephemeral_messages(self, session_id: str) -> int:
        """Remove all ephemeral messages for a session.

        Uses a two-phase approach for performance:
        1. Push cheap, SQL-expressible heuristics into a single SELECT to get
           candidate IDs (tool role, long content, stdout-like patterns).
        2. Filter candidates in Python for nuanced rules (critical errors,
           memory retrieval), then issue one batched DELETE.

        Keeps:
        - System messages
        - User prompts
        - Assistant responses (summaries, explanations)

        Removes:
        - Tool results
        - File contents
        - Terminal outputs
        - Error traces

        Args:
            session_id: The session to clear

        Returns:
            Number of messages removed
        """
        def _do(conn):
            # --- Phase 1: SQL-fast path for trivially-ephemeral messages,
            #            plus candidate IDs for nuanced Python filtering.
            #
            # SQL can directly express:
            #   role = 'tool'
            #   LENGTH(content) > 5000
            #   content contains exit_code or stdout patterns
            #   content contains Error:/Exception: (critical check done below)
            #   content contains memory retrieval patterns
            #
            # We fetch candidate rows and let Python make the final call on
            # each.  This avoids N+1 DELETEs while still honouring the
            # nuanced classification logic.

            # Build parameter list: session_id, max_length, then one param per
            # LIKE pattern in _EPHEMERAL_SQL_PATTERNS.
            like_params = list(_EPHEMERAL_SQL_PATTERNS)
            cursor = conn.execute(
                f"""SELECT id, role, content, protected FROM messages
                    WHERE session_id = ? AND protected = 0
                      AND (
                            role = 'tool'
                         OR LENGTH(content) > ?
                         OR {_EPHEMERAL_LIKE_CONDITIONS}
                         OR (LOWER(content) LIKE '%memory%' AND
                             (LOWER(content) LIKE '%retrieved%' OR LOWER(content) LIKE '%found%'))
                      )""",
                (session_id, MAX_EPHEMERAL_CONTENT_LENGTH, *like_params),
            )
            rows = cursor.fetchall()

            # --- Phase 2: Python-level nuanced filtering (critical errors). ---
            ids_to_delete: List[int] = []
            for row in rows:
                # row is (id, role, content, protected)
                msg = {"role": row[1], "content": row[2]}
                if row[3]:
                    # protected (row-level sanity check, though SQL already
                    # excluded protected=1)
                    continue
                if self.is_ephemeral_message(msg):
                    ids_to_delete.append(row[0])

            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    ids_to_delete,
                )

            return len(ids_to_delete)

        return self._execute_write(_do)

    def protect_messages_after(self, session_id: str, after_timestamp: float = None) -> int:
        """Mark messages as protected starting from a given timestamp.
        
        If after_timestamp is None, protects all the most recent messages
        (i.e., nothing is protected yet in this session).
        
        Args:
            session_id: The session to protect messages in
            after_timestamp: Protect all messages with timestamp > this value.
                If None, finds the latest existing message timestamp and protects
                everything strictly after it (i.e., new messages from now on).
        
        Returns:
            Number of messages now protected
        """
        cutoff = after_timestamp  # capture before _do to avoid UnboundLocalError
        
        def _do(conn):
            nonlocal cutoff
            if cutoff is None:
                # Find latest message timestamp for this session
                cursor = conn.execute(
                    "SELECT MAX(timestamp) FROM messages WHERE session_id = ?",
                    (session_id,),
                )
                row = cursor.fetchone()
                cutoff = row[0] if row and row[0] else 0
            
            cursor = conn.execute(
                "UPDATE messages SET protected = 1 WHERE session_id = ? AND timestamp > ? AND protected = 0",
                (session_id, cutoff),
            )
            updated = cursor.rowcount
            return updated
        
        return self._execute_write(_do)

    def unprotect_all_messages(self, session_id: str) -> int:
        """Remove protection from all messages in a session.
        
        Returns:
            Number of messages unprotected
        """
        def _do(conn):
            cursor = conn.execute(
                "UPDATE messages SET protected = 0 WHERE session_id = ? AND protected = 1",
                (session_id,),
            )
            return cursor.rowcount
        
        return self._execute_write(_do)

    def start_task_protection(self, session_id: str) -> bool:
        """Enable task protection for a session — new messages are auto-protected.

        Returns True if protection was activated (was not already active).
        """
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET task_protection_active = 1 WHERE id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

        return self._execute_write(_do)

    def end_task_protection(self, session_id: str) -> bool:
        """Disable task protection for a session — clears the flag.

        Returns True if protection was deactivated (was active).
        """
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET task_protection_active = 0 WHERE id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

        return self._execute_write(_do)

    def is_task_protection_active(self, session_id: str) -> bool:
        """Check whether task protection is currently active for a session."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT task_protection_active FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return bool(row and row[0])

    # =========================================================================
    # Task Definition Mode (for /start-tasks ... /end-tasks task train)
    # =========================================================================

    def enter_task_definition_mode(self, session_id: str) -> bool:
        """Enter task definition mode — messages are buffered, not sent to model.

        Returns True if mode was activated (was not already active).
        """
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET task_definition_mode = 1 WHERE id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

        return self._execute_write(_do)

    def exit_task_definition_mode(self, session_id: str) -> bool:
        """Exit task definition mode — buffer will be flushed.

        Returns True if mode was deactivated (was active).
        """
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET task_definition_mode = 0 WHERE id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

        return self._execute_write(_do)

    def is_task_definition_mode_active(self, session_id: str) -> bool:
        """Check whether task definition mode is currently active."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT task_definition_mode FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return bool(row and row[0])

    def get_task_definitions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all buffered task definitions for a session.

        Returns list of task messages (role='user' messages in task definition mode).
        """
        def _do(conn):
            cursor = conn.execute(
                """SELECT id, role, content, timestamp FROM messages
                   WHERE session_id = ? AND role = 'user' AND task_definition = 1
                   ORDER BY timestamp""",
                (session_id,),
            )
            return [{"id": row[0], "role": row[1], "content": row[2], "timestamp": row[3]} for row in cursor.fetchall()]

        return self._execute_write(_do)

    def store_task_definition(self, session_id: str, content: str) -> int:
        """Store a task definition message (buffered, not sent to model).

        Args:
            session_id: The session
            content: The task description

        Returns:
            The ID of the inserted message
        """
        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, timestamp, task_definition)
                   VALUES (?, 'user', ?, ?, 1)""",
                (session_id, content, time.time()),
            )
            return cursor.lastrowid

        return self._execute_write(_do)

    def clear_task_definitions(self, session_id: str) -> int:
        """Clear all buffered task definitions for a session.

        Returns:
            Number of definitions cleared
        """
        def _do(conn):
            cursor = conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND task_definition = 1",
                (session_id,),
            )
            return cursor.rowcount

        return self._execute_write(_do)

    def get_protected_messages(self, session_id: str) -> List[float]:
        """Get timestamps of all protected messages in a session.
        
        Returns a sorted list of timestamps to know which messages are protected
        and to help determine what to protect next.
        """
        def _do(conn):
            cursor = conn.execute(
                "SELECT timestamp FROM messages WHERE session_id = ? AND protected = 1 ORDER BY timestamp",
                (session_id,),
            )
            return [row[0] for row in cursor.fetchall()]
        
        # This read needs to see uncommitted writes from our own connection,
        # so use the direct connection in a lock.
        with self._lock:
            return _do(self._conn)

    def clear_task_window_ephemeral_messages(self, session_id: str) -> int:
        """Remove only ephemeral messages from the task window.

        Task-window messages are those marked as protected=1 (added between
        /start-tasks and /end-tasks).  This method clears the *noise* from
        a completed task — tool outputs, file dumps, terminal results — while
        preserving the user's task definition and the assistant's summary.

        Uses the same two-phase approach as clear_ephemeral_messages but
        targets protected=1 instead of protected=0.

        Args:
            session_id: The session to clear

        Returns:
            Number of task-window ephemeral messages removed
        """
        def _do(conn):
            # Phase 1: SQL candidate filter — only fetch protected messages
            # that look like they could be ephemeral.  This avoids pulling
            # every protected message into Python when most are short user/
            # assistant text.  The SQL filter is a superset; Python makes
            # the final call (Phase 2).
            like_params = list(_EPHEMERAL_SQL_PATTERNS)
            cursor = conn.execute(
                f"""SELECT id, role, content FROM messages
                    WHERE session_id = ? AND protected = 1
                      AND (role = 'tool'
                           OR LENGTH(content) > ?
                           OR {_EPHEMERAL_LIKE_CONDITIONS})""",
                (session_id, MAX_EPHEMERAL_CONTENT_LENGTH, *like_params),
            )
            rows = cursor.fetchall()

            # --- Phase 2: Python-level classification. ---
            ids_to_delete: List[int] = []
            for row in rows:
                msg = {"role": row[1], "content": row[2]}
                if self.is_ephemeral_message(msg):
                    ids_to_delete.append(row[0])

            if ids_to_delete:
                # Delete the ephemeral ones (they're no longer needed)
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    ids_to_delete,
                )

            # Unprotect ALL remaining protected messages in this session.
            # These are non-ephemeral user/assistant messages from the task
            # window (task definitions and summaries) that should survive as
            # normal messages after the task is complete.  A single UPDATE is
            # cheaper than fetching IDs one-by-one.
            conn.execute(
                "UPDATE messages SET protected = 0 WHERE session_id = ? AND protected = 1",
                (session_id,),
            )

            return len(ids_to_delete)

        return self._execute_write(_do)

    def prune_messages(self, session_id: str, count: int = DEFAULT_PRUNE_COUNT) -> int:
        """Remove the last N messages from a session (excluding system messages).

        Args:
            session_id: The session to prune
            count: Number of messages to remove (default: DEFAULT_PRUNE_COUNT)

        Returns:
            Number of messages removed
        """
        def _do(conn):
            cursor = conn.execute(
                """SELECT id FROM messages
                   WHERE session_id = ? AND role != 'system' AND protected = 0
                   ORDER BY timestamp DESC, id DESC
                   LIMIT ?""",
                (session_id, count),
            )
            message_ids: List[int] = [row["id"] for row in cursor.fetchall()]

            if message_ids:
                placeholders = ",".join("?" * len(message_ids))
                conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    message_ids,
                )

            return len(message_ids)

        return self._execute_write(_do)
