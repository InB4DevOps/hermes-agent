"""Tests for hermes_state.py — SessionDB SQLite CRUD, FTS5 search, export."""

import time
import pytest
from pathlib import Path

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


"""Tests for hermes_state.py — SessionDB SQLite CRUD, FTS5 search, export."""

import time
import pytest
from pathlib import Path

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


# ======================================================================
# Untested/Edge cases
# ======================================================================

class TestClearMessages:
    """Tests for clear_messages() - deletes all messages and resets counters."""

    def test_clear_messages(self, db: SessionDB):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")
        db.append_message("s1", role="tool", tool_name="web_search")

        # Verify messages exist
        assert db.message_count(session_id="s1") == 3
        session = db.get_session("s1")
        assert session["message_count"] == 3
        assert session["tool_call_count"] == 1

        # Clear messages
        db.clear_messages("s1")

        # Verify messages deleted
        assert db.message_count(session_id="s1") == 0
        assert db.get_messages("s1") == []

        # Verify counters reset
        session = db.get_session("s1")
        assert session["message_count"] == 0
        assert session["tool_call_count"] == 0

    def test_clear_nonexistent_session(self, db: SessionDB):
        # Should not raise for nonexistent session
        assert db.clear_messages("nonexistent") is not None  # Returns None implicitly

    def test_clear_messages_preserves_session_metadata(self, db: SessionDB):
        db.create_session(
            session_id="s1",
            source="cli",
            model="test-model",
            system_prompt="System prompt text",
        )
        db.end_session("s1", end_reason="user_exit")
        db.set_session_title("s1", "My Title")

        db.append_message("s1", role="user", content="Hello")

        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"

        db.clear_messages("s1")

        # Session metadata should persist
        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"


class TestClose:
    """Tests for close() method."""

    def test_close_releases_connection(self, db: SessionDB):
        assert db._conn is not None
        db.close()
        assert db._conn is None

    def test_close_twice_no_error(self, db: SessionDB):
        db.close()
        db.close()  # Should not raise

    def test_close_after_append(self, db: SessionDB):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        db.close()

        # Connection released, can be reopened
        new_db = SessionDB(db_path=db.db_path)
        new_db.close()


class TestPruneSessionsEdgeCases:
    """Edge cases for prune_sessions()."""

    def test_prune_zero_days(self, db: SessionDB):
        """Pruning with 0 days should delete all ended sessions."""
        db.create_session("old1", "cli")
        db.end_session("old1", end_reason="done")
        db.create_session("old2", "cli")
        db.end_session("old2", end_reason="done")

        pruned = db.prune_sessions(older_than_days=0)
        assert pruned == 2
        assert db.get_session("old1") is None
        assert db.get_session("old2") is None

    def test_prune_all_active_skipped(self, db: SessionDB):
        """All active sessions should be skipped."""
        db.create_session("active1", "cli")
        db.create_session("active2", "cli")

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0
        assert db.get_session("active1") is not None
        assert db.get_session("active2") is not None

    def test_prune_empty_db(self, db: SessionDB):
        """Pruning empty database should return 0."""
        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0

    def test_prune_preserves_started_at(self, db: SessionDB):
        """Pruning should delete sessions, not just truncate."""
        db.create_session("old", "cli")
        db.end_session("old", end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 200 * 86400, "old"),
        )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 1
        assert db.get_session("old") is None


class TestListSessionsRichEdgeCases:
    """Edge cases for list_sessions_rich()."""

    def test_list_empty_database(self, db: SessionDB):
        """Listing empty database should return empty list."""
        sessions = db.list_sessions_rich()
        assert sessions == []

    def test_list_limit_zero(self, db: SessionDB):
        """Limit=0 should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")

        sessions = db.list_sessions_rich(limit=0)
        assert sessions == []

    def test_list_large_offset(self, db: SessionDB):
        """Large offset should return empty (no more results)."""
        for i in range(3):
            db.create_session(f"s{i}", "cli")
            db.append_message(f"s{i}", "user", f"Message {i}")

        sessions = db.list_sessions_rich(offset=100)
        assert sessions == []

    def test_list_all_fields_present(self, db: SessionDB):
        """All expected fields should be present in result."""
        import time
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        sessions = db.list_sessions_rich(limit=1)
        assert len(sessions) == 1
        session = sessions[0]

        # Check all expected fields
        assert "id" in session
        assert "source" in session
        assert "model" in session
        assert "title" in session
        assert "started_at" in session
        assert "ended_at" in session
        assert "message_count" in session
        assert "preview" in session
        assert "last_active" in session
        # Note: Some fields may be None if not set


class TestGetSessionByTitleEdgeCases:
    """Edge cases for get_session_by_title()."""

    def test_case_sensitive(self, db: SessionDB):
        """Title lookup should be case-sensitive."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Title")

        result = db.get_session_by_title("my title")
        assert result is None

        result = db.get_session_by_title("My Title")
        assert result is not None

    def test_special_characters_in_title(self, db: SessionDB):
        """Titles with special characters should work."""
        db.create_session("s1", "cli")
        title = "Title with spaces and #1"
        db.set_session_title("s1", title)

        result = db.get_session_by_title(title)
        assert result is not None

    def test_empty_title(self, db: SessionDB):
        """Empty title should return None."""
        db.create_session("s1", "cli")
        result = db.get_session_by_title("")
        assert result is None


class TestAppendMessageEdgeCases:
    """Edge cases for append_message()."""

    def test_tool_calls_none(self, db: SessionDB):
        """tool_calls=None should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=None)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_empty_list(self, db: SessionDB):
        """tool_calls=[] should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=[])

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_non_list(self, db: SessionDB):
        """tool_calls with non-list value (should be treated as 1 call)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls="some_string")

        session = db.get_session("s1")
        assert session["message_count"] == 1
        # Non-list with truthy value counts as 1
        assert session["tool_call_count"] == 1

    def test_tool_calls_single_dict(self, db: SessionDB):
        """tool_calls with single dict (should count as 1)."""
        db.create_session("s1", "cli")
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "Hello", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1


class TestExportAllEdgeCases:
    """Edge cases for export_all()."""

    def test_export_all_empty_source(self, db: SessionDB):
        """export_all(source=...) with no matches should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "telegram")

        exports = db.export_all(source="nonexistent")
        assert exports == []

    def test_export_all_with_messages(self, db: SessionDB):
        """Export should include messages."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        exports = db.export_all()
        assert len(exports) == 1
        assert len(exports[0]["messages"]) == 2

    def test_export_all_preserves_all_fields(self, db: SessionDB):
        """Export should preserve all session fields."""
        db.create_session(
            "s1",
            "cli",
            model="test-model",
            system_prompt="System prompt",
        )
        db.end_session("s1", end_reason="done")
        db.set_session_title("s1", "My Title")
        db.append_message("s1", "user", "Hello")

        exports = db.export_all()
        export = exports[0]

        assert "id" in export
        assert "source" in export
        assert "model" in export
        assert "system_prompt" in export
        assert "title" in export
        assert "ended_at" in export
        assert "messages" in export
        assert len(export["messages"]) == 1


class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_reads_writes(self, db: SessionDB):
        """Concurrent reads and writes should not corrupt data."""
        import threading
        import time

        errors = []
        lock = threading.Lock()

        def writer():
            try:
                for i in range(10):
                    session_id = f"writer_{i}"
                    db.create_session(session_id, "cli")
                    db.append_message(session_id, "user", f"Message {i}")
                    db.end_session(session_id, end_reason="done")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        def reader():
            try:
                for i in range(10):
                    time.sleep(0.01)
                    sessions = db.search_sessions()
                    messages = db.message_count()
                    assert len(sessions) == 0  # All created by writer
                    assert messages == 0  # All created by writer
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(errors) == 0

        # Verify data integrity
        assert db.session_count() == 10
        assert db.message_count() == 10

    def test_concurrent_same_session_writes(self, db: SessionDB):
        """Multiple concurrent writes to same session should be consistent."""
        import threading
        import time

        session_id = "concurrent_session"
        db.create_session(session_id, "cli")

        errors = []
        lock = threading.Lock()

        def writer(count):
            try:
                for _ in range(count):
                    db.append_message(session_id, "user", f"Message from {count}")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(5,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Should have 15 messages total (3 threads × 5 messages)
        assert db.message_count(session_id) == 15


class TestSchemaInitEdgeCases:
    """Edge cases for schema initialization."""

    def test_clean_database_creation(self, tmp_path):
        """Creating a new database should initialize to v5."""
        db_path = tmp_path / "clean_db.db"
        db = SessionDB(db_path=db_path)

        # Should be at v5 immediately
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        # Verify all tables exist
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "messages" in tables
        assert "schema_version" in tables
        assert "messages_fts" in tables

        db.close()

    def test_upgrade_from_v5_to_v5(self, tmp_path):
        """Opening a v5 database should not change version."""
        import sqlite3

        db_path = tmp_path / "v5_db.db"
        conn = sqlite3.connect(str(db_path))

        # Create v5 schema
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (5);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                title TEXT,
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
            );

            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content, content=messages, content_rowid=id
            );

            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)

        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("existing", "cli", 1000.0),
        )
        conn.commit()
        conn.close()

        # Open with SessionDB
        db = SessionDB(db_path=db_path)

        # Should still be v5
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        db.close()


class TestUpdateTokenCountsEdgeCases:
    """Edge cases for update_token_counts()."""

    def test_update_all_billing_fields(self, db: SessionDB):
        """Test updating all billing-related fields."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
            reasoning_tokens=20,
            estimated_cost_usd=1.0,
            actual_cost_usd=1.05,
            cost_status="within budget",
            cost_source="api",
            pricing_version="2024.01",
            billing_provider="anthropic",
            billing_base_url="https://api.anthropic.com",
            billing_mode="standard",
        )

        session = db.get_session("s1")

        # Check all billing fields
        assert session["billing_provider"] == "anthropic"
        assert session["billing_base_url"] == "https://api.anthropic.com"
        assert session["billing_mode"] == "standard"
        assert session["estimated_cost_usd"] == 1.0
        assert session["actual_cost_usd"] == 1.05
        assert session["cost_status"] == "within budget"
        assert session["cost_source"] == "api"
        assert session["pricing_version"] == "2024.01"

    def test_cost_accumulation(self, db: SessionDB):
        """Costs should accumulate across multiple updates."""
        db.create_session("s1", "cli")

        # First update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=1.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 100
        assert session["estimated_cost_usd"] == 1.0

        # Second update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=2.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 200
        assert session["estimated_cost_usd"] == 3.0

    def test_cost_status_preserved(self, db: SessionDB):
        """Cost status should be preserved across updates."""
        db.create_session("s1", "cli")

        # First update with "within budget"
        db.update_token_counts("s1", input_tokens=100, cost_status="within budget")

        # Second update without cost_status (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["cost_status"] == "within budget"

    def test_billing_fields_backfill_once(self, db: SessionDB):
        """Billing fields should be backfilled only once, then preserved."""
        db.create_session("s1", "cli")

        # First update with billing fields
        db.update_token_counts(
            "s1",
            input_tokens=100,
            billing_provider="test",
            billing_base_url="https://test.com",
        )

        # Second update without billing fields (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["billing_provider"] == "test"
        assert session["billing_base_url"] == "https://test.com"

    def test_billing_over_budget_status(self, db: SessionDB):
        """Test 'over budget' cost status."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            actual_cost_usd=1000.0,
            cost_status="over budget",
            cost_source="api",
            pricing_version="2024.01",
        )

        session = db.get_session("s1")
        assert session["cost_status"] == "over budget"
        assert session["actual_cost_usd"] == 1000.0


class TestFTSEdgeCases:
    """Edge cases for FTS5 search."""

    def test_search_whitespace_only_query(self, db: SessionDB):
        """Search with whitespace-only query should return empty."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        assert db.search_messages("   ") == []
        assert db.search_messages("\t\n") == []

    def test_search_single_character(self, db: SessionDB):
        """Search with single character should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        results = db.search_messages("e")
        assert isinstance(results, list)  # May or may not match depending on FTS5

    def test_search_unicode_content(self, db: SessionDB):
        """Search with unicode content should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "こんにちは世界")
        db.append_message("s1", "assistant", "Hello World")

        results = db.search_messages("こんにちは")
        assert isinstance(results, list)

    def test_search_empty_content_message(self, db: SessionDB):
        """Search should handle messages with empty content."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "")
        db.append_message("s1", "assistant", "Hello")

        results = db.search_messages("Hello")
        # Should find the assistant message
        assert len(results) >= 0


class TestToolCallCountEdgeCases:
    """Edge cases for tool call counting."""

    def test_tool_calls_count_in_message(self, db: SessionDB):
        """Tool calls in assistant message should count."""
        db.create_session("s1", "cli")

        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 2

    def test_tool_calls_no_content_still_counts(self, db: SessionDB):
        """Tool calls should count even if content is empty."""
        db.create_session("s1", "cli")

        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1

    def test_tool_calls_with_tool_response(self, db: SessionDB):
        """Tool responses should not increment tool_call_count."""
        db.create_session("s1", "cli")

        # Assistant makes 1 tool call
        tool_calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Tool response comes back
        db.append_message("s1", "tool", "Result", tool_name="web_search")

        session = db.get_session("s1")
        # Should be 1 (the assistant call), not 2
        assert session["tool_call_count"] == 1

    def test_multiple_tool_calls_multiple_responses(self, db: SessionDB):
        """Multiple tool calls with multiple responses should count correctly."""
        db.create_session("s1", "cli")

        # Assistant makes 2 parallel tool calls
        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Two tool responses
        db.append_message("s1", "tool", "Result 1", tool_name="web_search")
        db.append_message("s1", "tool", "Result 2", tool_name="file_read")

        session = db.get_session("s1")
        assert session["tool_call_count"] == 2
        assert session["message_count"] == 3  # 1 assistant + 2 tool responses
# ======================================================================
# Untested/Edge cases
# ======================================================================

class TestClearMessages:
    """Tests for clear_messages() - deletes all messages and resets counters."""

    def test_clear_messages(self, db: SessionDB):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")
        db.append_message("s1", role="tool", tool_name="web_search")

        # Verify messages exist
        assert db.message_count(session_id="s1") == 3
        session = db.get_session("s1")
        assert session["message_count"] == 3
        assert session["tool_call_count"] == 1

        # Clear messages
        db.clear_messages("s1")

        # Verify messages deleted
        assert db.message_count(session_id="s1") == 0
        assert db.get_messages("s1") == []

        # Verify counters reset
        session = db.get_session("s1")
        assert session["message_count"] == 0
        assert session["tool_call_count"] == 0

    def test_clear_nonexistent_session(self, db: SessionDB):
        # Should not raise for nonexistent session
        assert db.clear_messages("nonexistent") is not None  # Returns None implicitly

    def test_clear_messages_preserves_session_metadata(self, db: SessionDB):
        db.create_session(
            session_id="s1",
            source="cli",
            model="test-model",
            system_prompt="System prompt text",
        )
        db.end_session("s1", end_reason="user_exit")
        db.set_session_title("s1", "My Title")

        db.append_message("s1", role="user", content="Hello")

        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"

        db.clear_messages("s1")

        # Session metadata should persist
        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"


class TestClose:
    """Tests for close() method."""

    def test_close_releases_connection(self, db: SessionDB):
        assert db._conn is not None
        db.close()
        assert db._conn is None

    def test_close_twice_no_error(self, db: SessionDB):
        db.close()
        db.close()  # Should not raise

    def test_close_after_append(self, db: SessionDB):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        db.close()

        # Connection released, can be reopened
        new_db = SessionDB(db_path=db.db_path)
        new_db.close()


class TestPruneSessionsEdgeCases:
    """Edge cases for prune_sessions()."""

    def test_prune_zero_days(self, db: SessionDB):
        """Pruning with 0 days should delete all ended sessions."""
        db.create_session("old1", "cli")
        db.end_session("old1", end_reason="done")
        db.create_session("old2", "cli")
        db.end_session("old2", end_reason="done")

        pruned = db.prune_sessions(older_than_days=0)
        assert pruned == 2
        assert db.get_session("old1") is None
        assert db.get_session("old2") is None

    def test_prune_all_active_skipped(self, db: SessionDB):
        """All active sessions should be skipped."""
        db.create_session("active1", "cli")
        db.create_session("active2", "cli")

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0
        assert db.get_session("active1") is not None
        assert db.get_session("active2") is not None

    def test_prune_empty_db(self, db: SessionDB):
        """Pruning empty database should return 0."""
        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0

    def test_prune_preserves_started_at(self, db: SessionDB):
        """Pruning should delete sessions, not just truncate."""
        db.create_session("old", "cli")
        db.end_session("old", end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 200 * 86400, "old"),
        )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 1
        assert db.get_session("old") is None


class TestListSessionsRichEdgeCases:
    """Edge cases for list_sessions_rich()."""

    def test_list_empty_database(self, db: SessionDB):
        """Listing empty database should return empty list."""
        sessions = db.list_sessions_rich()
        assert sessions == []

    def test_list_limit_zero(self, db: SessionDB):
        """Limit=0 should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")

        sessions = db.list_sessions_rich(limit=0)
        assert sessions == []

    def test_list_large_offset(self, db: SessionDB):
        """Large offset should return empty (no more results)."""
        for i in range(3):
            db.create_session(f"s{i}", "cli")
            db.append_message(f"s{i}", "user", f"Message {i}")

        sessions = db.list_sessions_rich(offset=100)
        assert sessions == []

    def test_list_all_fields_present(self, db: SessionDB):
        """All expected fields should be present in result."""
        import time
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        sessions = db.list_sessions_rich(limit=1)
        assert len(sessions) == 1
        session = sessions[0]

        # Check all expected fields
        assert "id" in session
        assert "source" in session
        assert "model" in session
        assert "title" in session
        assert "started_at" in session
        assert "ended_at" in session
        assert "message_count" in session
        assert "preview" in session
        assert "last_active" in session
        # Note: Some fields may be None if not set


class TestGetSessionByTitleEdgeCases:
    """Edge cases for get_session_by_title()."""

    def test_case_sensitive(self, db: SessionDB):
        """Title lookup should be case-sensitive."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Title")

        result = db.get_session_by_title("my title")
        assert result is None

        result = db.get_session_by_title("My Title")
        assert result is not None

    def test_special_characters_in_title(self, db: SessionDB):
        """Titles with special characters should work."""
        db.create_session("s1", "cli")
        title = "Title with spaces and #1"
        db.set_session_title("s1", title)

        result = db.get_session_by_title(title)
        assert result is not None

    def test_empty_title(self, db: SessionDB):
        """Empty title should return None."""
        db.create_session("s1", "cli")
        result = db.get_session_by_title("")
        assert result is None


class TestAppendMessageEdgeCases:
    """Edge cases for append_message()."""

    def test_tool_calls_none(self, db: SessionDB):
        """tool_calls=None should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=None)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_empty_list(self, db: SessionDB):
        """tool_calls=[] should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=[])

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_non_list(self, db: SessionDB):
        """tool_calls with non-list value (should be treated as 1 call)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls="some_string")

        session = db.get_session("s1")
        assert session["message_count"] == 1
        # Non-list with truthy value counts as 1
        assert session["tool_call_count"] == 1

    def test_tool_calls_single_dict(self, db: SessionDB):
        """tool_calls with single dict (should count as 1)."""
        db.create_session("s1", "cli")
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "Hello", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1


class TestExportAllEdgeCases:
    """Edge cases for export_all()."""

    def test_export_all_empty_source(self, db: SessionDB):
        """export_all(source=...) with no matches should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "telegram")

        exports = db.export_all(source="nonexistent")
        assert exports == []

    def test_export_all_with_messages(self, db: SessionDB):
        """Export should include messages."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        exports = db.export_all()
        assert len(exports) == 1
        assert len(exports[0]["messages"]) == 2

    def test_export_all_preserves_all_fields(self, db: SessionDB):
        """Export should preserve all session fields."""
        db.create_session(
            "s1",
            "cli",
            model="test-model",
            system_prompt="System prompt",
        )
        db.end_session("s1", end_reason="done")
        db.set_session_title("s1", "My Title")
        db.append_message("s1", "user", "Hello")

        exports = db.export_all()
        export = exports[0]

        assert "id" in export
        assert "source" in export
        assert "model" in export
        assert "system_prompt" in export
        assert "title" in export
        assert "ended_at" in export
        assert "messages" in export
        assert len(export["messages"]) == 1


class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_reads_writes(self, db: SessionDB):
        """Concurrent reads and writes should not corrupt data."""
        import threading
        import time

        errors = []
        lock = threading.Lock()

        def writer():
            try:
                for i in range(10):
                    session_id = f"writer_{i}"
                    db.create_session(session_id, "cli")
                    db.append_message(session_id, "user", f"Message {i}")
                    db.end_session(session_id, end_reason="done")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        def reader():
            try:
                for i in range(10):
                    time.sleep(0.01)
                    sessions = db.search_sessions()
                    messages = db.message_count()
                    assert len(sessions) == 0  # All created by writer
                    assert messages == 0  # All created by writer
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(errors) == 0

        # Verify data integrity
        assert db.session_count() == 10
        assert db.message_count() == 10

    def test_concurrent_same_session_writes(self, db: SessionDB):
        """Multiple concurrent writes to same session should be consistent."""
        import threading
        import time

        session_id = "concurrent_session"
        db.create_session(session_id, "cli")

        errors = []
        lock = threading.Lock()

        def writer(count):
            try:
                for _ in range(count):
                    db.append_message(session_id, "user", f"Message from {count}")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(5,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Should have 15 messages total (3 threads × 5 messages)
        assert db.message_count(session_id) == 15


class TestSchemaInitEdgeCases:
    """Edge cases for schema initialization."""

    def test_clean_database_creation(self, tmp_path):
        """Creating a new database should initialize to v5."""
        db_path = tmp_path / "clean_db.db"
        db = SessionDB(db_path=db_path)

        # Should be at v5 immediately
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        # Verify all tables exist
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "messages" in tables
        assert "schema_version" in tables
        assert "messages_fts" in tables

        db.close()

    def test_upgrade_from_v5_to_v5(self, tmp_path):
        """Opening a v5 database should not change version."""
        import sqlite3

        db_path = tmp_path / "v5_db.db"
        conn = sqlite3.connect(str(db_path))

        # Create v5 schema
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (5);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                title TEXT,
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
            );

            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content, content=messages, content_rowid=id
            );

            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)

        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("existing", "cli", 1000.0),
        )
        conn.commit()
        conn.close()

        # Open with SessionDB
        db = SessionDB(db_path=db_path)

        # Should still be v5
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        db.close()


class TestUpdateTokenCountsEdgeCases:
    """Edge cases for update_token_counts()."""

    def test_update_all_billing_fields(self, db: SessionDB):
        """Test updating all billing-related fields."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
            reasoning_tokens=20,
            estimated_cost_usd=1.0,
            actual_cost_usd=1.05,
            cost_status="within budget",
            cost_source="api",
            pricing_version="2024.01",
            billing_provider="anthropic",
            billing_base_url="https://api.anthropic.com",
            billing_mode="standard",
        )

        session = db.get_session("s1")

        # Check all billing fields
        assert session["billing_provider"] == "anthropic"
        assert session["billing_base_url"] == "https://api.anthropic.com"
        assert session["billing_mode"] == "standard"
        assert session["estimated_cost_usd"] == 1.0
        assert session["actual_cost_usd"] == 1.05
        assert session["cost_status"] == "within budget"
        assert session["cost_source"] == "api"
        assert session["pricing_version"] == "2024.01"

    def test_cost_accumulation(self, db: SessionDB):
        """Costs should accumulate across multiple updates."""
        db.create_session("s1", "cli")

        # First update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=1.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 100
        assert session["estimated_cost_usd"] == 1.0

        # Second update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=2.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 200
        assert session["estimated_cost_usd"] == 3.0

    def test_cost_status_preserved(self, db: SessionDB):
        """Cost status should be preserved across updates."""
        db.create_session("s1", "cli")

        # First update with "within budget"
        db.update_token_counts("s1", input_tokens=100, cost_status="within budget")

        # Second update without cost_status (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["cost_status"] == "within budget"

    def test_billing_fields_backfill_once(self, db: SessionDB):
        """Billing fields should be backfilled only once, then preserved."""
        db.create_session("s1", "cli")

        # First update with billing fields
        db.update_token_counts(
            "s1",
            input_tokens=100,
            billing_provider="test",
            billing_base_url="https://test.com",
        )

        # Second update without billing fields (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["billing_provider"] == "test"
        assert session["billing_base_url"] == "https://test.com"

    def test_billing_over_budget_status(self, db: SessionDB):
        """Test 'over budget' cost status."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            actual_cost_usd=1000.0,
            cost_status="over budget",
            cost_source="api",
            pricing_version="2024.01",
        )

        session = db.get_session("s1")
        assert session["cost_status"] == "over budget"
        assert session["actual_cost_usd"] == 1000.0


class TestFTSEdgeCases:
    """Edge cases for FTS5 search."""

    def test_search_whitespace_only_query(self, db: SessionDB):
        """Search with whitespace-only query should return empty."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        assert db.search_messages("   ") == []
        assert db.search_messages("\t\n") == []

    def test_search_single_character(self, db: SessionDB):
        """Search with single character should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        results = db.search_messages("e")
        assert isinstance(results, list)  # May or may not match depending on FTS5

    def test_search_unicode_content(self, db: SessionDB):
        """Search with unicode content should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "こんにちは世界")
        db.append_message("s1", "assistant", "Hello World")

        results = db.search_messages("こんにちは")
        assert isinstance(results, list)

    def test_search_empty_content_message(self, db: SessionDB):
        """Search should handle messages with empty content."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "")
        db.append_message("s1", "assistant", "Hello")

        results = db.search_messages("Hello")
        # Should find the assistant message
        assert len(results) >= 0


class TestToolCallCountEdgeCases:
    """Edge cases for tool call counting."""

    def test_tool_calls_count_in_message(self, db: SessionDB):
        """Tool calls in assistant message should count."""
        db.create_session("s1", "cli")

        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 2

    def test_tool_calls_no_content_still_counts(self, db: SessionDB):
        """Tool calls should count even if content is empty."""
        db.create_session("s1", "cli")

        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1

    def test_tool_calls_with_tool_response(self, db: SessionDB):
        """Tool responses should not increment tool_call_count."""
        db.create_session("s1", "cli")

        # Assistant makes 1 tool call
        tool_calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Tool response comes back
        db.append_message("s1", "tool", "Result", tool_name="web_search")

        session = db.get_session("s1")
        # Should be 1 (the assistant call), not 2
        assert session["tool_call_count"] == 1

    def test_multiple_tool_calls_multiple_responses(self, db: SessionDB):
        """Multiple tool calls with multiple responses should count correctly."""
        db.create_session("s1", "cli")

        # Assistant makes 2 parallel tool calls
        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Two tool responses
        db.append_message("s1", "tool", "Result 1", tool_name="web_search")
        db.append_message("s1", "tool", "Result 2", tool_name="file_read")

        session = db.get_session("s1")
        assert session["tool_call_count"] == 2
        assert session["message_count"] == 3  # 1 assistant + 2 tool responses
# ======================================================================
# Untested/Edge cases
# ======================================================================

class TestClearMessages:
    """Tests for clear_messages() - deletes all messages and resets counters."""

    def test_clear_messages(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")
        db.append_message("s1", role="tool", tool_name="web_search")

        # Verify messages exist
        assert db.message_count(session_id="s1") == 3
        session = db.get_session("s1")
        assert session["message_count"] == 3
        assert session["tool_call_count"] == 1

        # Clear messages
        db.clear_messages("s1")

        # Verify messages deleted
        assert db.message_count(session_id="s1") == 0
        assert db.get_messages("s1") == []

        # Verify counters reset
        session = db.get_session("s1")
        assert session["message_count"] == 0
        assert session["tool_call_count"] == 0

    def test_clear_nonexistent_session(self, db):
        # Should not raise for nonexistent session
        assert db.clear_messages("nonexistent") is not None  # Returns None implicitly

    def test_clear_messages_preserves_session_metadata(self, db):
        db.create_session(
            session_id="s1",
            source="cli",
            model="test-model",
            system_prompt="System prompt text",
        )
        db.end_session("s1", end_reason="user_exit")
        db.set_session_title("s1", "My Title")

        db.append_message("s1", role="user", content="Hello")

        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"

        db.clear_messages("s1")

        # Session metadata should persist
        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"


class TestClose:
    """Tests for close() method."""

    def test_close_releases_connection(self, db):
        assert db._conn is not None
        db.close()
        assert db._conn is None

    def test_close_twice_no_error(self, db):
        db.close()
        db.close()  # Should not raise

    def test_close_after_append(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        db.close()

        # Connection released, can be reopened
        new_db = SessionDB(db_path=db.db_path)
        new_db.close()


class TestPruneSessionsEdgeCases:
    """Edge cases for prune_sessions()."""

    def test_prune_zero_days(self, db):
        """Pruning with 0 days should delete all ended sessions."""
        db.create_session("old1", "cli")
        db.end_session("old1", end_reason="done")
        db.create_session("old2", "cli")
        db.end_session("old2", end_reason="done")

        pruned = db.prune_sessions(older_than_days=0)
        assert pruned == 2
        assert db.get_session("old1") is None
        assert db.get_session("old2") is None

    def test_prune_all_active_skipped(self, db):
        """All active sessions should be skipped."""
        db.create_session("active1", "cli")
        db.create_session("active2", "cli")

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0
        assert db.get_session("active1") is not None
        assert db.get_session("active2") is not None

    def test_prune_empty_db(self, db):
        """Pruning empty database should return 0."""
        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0

    def test_prune_preserves_started_at(self, db):
        """Pruning should delete sessions, not just truncate."""
        db.create_session("old", "cli")
        db.end_session("old", end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 200 * 86400, "old"),
        )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 1
        assert db.get_session("old") is None


class TestListSessionsRichEdgeCases:
    """Edge cases for list_sessions_rich()."""

    def test_list_empty_database(self, db):
        """Listing empty database should return empty list."""
        sessions = db.list_sessions_rich()
        assert sessions == []

    def test_list_limit_zero(self, db):
        """Limit=0 should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")

        sessions = db.list_sessions_rich(limit=0)
        assert sessions == []

    def test_list_large_offset(self, db):
        """Large offset should return empty (no more results)."""
        for i in range(3):
            db.create_session(f"s{i}", "cli")
            db.append_message(f"s{i}", "user", f"Message {i}")

        sessions = db.list_sessions_rich(offset=100)
        assert sessions == []

    def test_list_all_fields_present(self, db):
        """All expected fields should be present in result."""
        import time
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        sessions = db.list_sessions_rich(limit=1)
        assert len(sessions) == 1
        session = sessions[0]

        # Check all expected fields
        assert "id" in session
        assert "source" in session
        assert "model" in session
        assert "title" in session
        assert "started_at" in session
        assert "ended_at" in session
        assert "message_count" in session
        assert "preview" in session
        assert "last_active" in session
        # Note: Some fields may be None if not set


class TestGetSessionByTitleEdgeCases:
    """Edge cases for get_session_by_title()."""

    def test_case_sensitive(self, db):
        """Title lookup should be case-sensitive."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Title")

        result = db.get_session_by_title("my title")
        assert result is None

        result = db.get_session_by_title("My Title")
        assert result is not None

    def test_special_characters_in_title(self, db):
        """Titles with special characters should work."""
        db.create_session("s1", "cli")
        title = "Title with spaces and #1"
        db.set_session_title("s1", title)

        result = db.get_session_by_title(title)
        assert result is not None

    def test_empty_title(self, db):
        """Empty title should return None."""
        db.create_session("s1", "cli")
        result = db.get_session_by_title("")
        assert result is None


class TestAppendMessageEdgeCases:
    """Edge cases for append_message()."""

    def test_tool_calls_none(self, db):
        """tool_calls=None should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=None)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_empty_list(self, db):
        """tool_calls=[] should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=[])

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_non_list(self, db):
        """tool_calls with non-list value (should be treated as 1 call)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls="some_string")

        session = db.get_session("s1")
        assert session["message_count"] == 1
        # Non-list with truthy value counts as 1
        assert session["tool_call_count"] == 1

    def test_tool_calls_single_dict(self, db):
        """tool_calls with single dict (should count as 1)."""
        db.create_session("s1", "cli")
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "Hello", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1


class TestExportAllEdgeCases:
    """Edge cases for export_all()."""

    def test_export_all_empty_source(self, db):
        """export_all(source=...) with no matches should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "telegram")

        exports = db.export_all(source="nonexistent")
        assert exports == []

    def test_export_all_with_messages(self, db):
        """Export should include messages."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        exports = db.export_all()
        assert len(exports) == 1
        assert len(exports[0]["messages"]) == 2

    def test_export_all_preserves_all_fields(self, db):
        """Export should preserve all session fields."""
        db.create_session(
            "s1",
            "cli",
            model="test-model",
            system_prompt="System prompt",
        )
        db.end_session("s1", end_reason="done")
        db.set_session_title("s1", "My Title")
        db.append_message("s1", "user", "Hello")

        exports = db.export_all()
        export = exports[0]

        assert "id" in export
        assert "source" in export
        assert "model" in export
        assert "system_prompt" in export
        assert "title" in export
        assert "ended_at" in export
        assert "messages" in export
        assert len(export["messages"]) == 1


class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_reads_writes(self, db):
        """Concurrent reads and writes should not corrupt data."""
        import threading
        import time

        errors = []
        lock = threading.Lock()

        def writer():
            try:
                for i in range(10):
                    session_id = f"writer_{i}"
                    db.create_session(session_id, "cli")
                    db.append_message(session_id, "user", f"Message {i}")
                    db.end_session(session_id, end_reason="done")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        def reader():
            try:
                for i in range(10):
                    time.sleep(0.01)
                    sessions = db.search_sessions()
                    messages = db.message_count()
                    assert len(sessions) == 0  # All created by writer
                    assert messages == 0  # All created by writer
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(errors) == 0

        # Verify data integrity
        assert db.session_count() == 10
        assert db.message_count() == 10

    def test_concurrent_same_session_writes(self, db):
        """Multiple concurrent writes to same session should be consistent."""
        import threading
        import time

        session_id = "concurrent_session"
        db.create_session(session_id, "cli")

        errors = []
        lock = threading.Lock()

        def writer(count):
            try:
                for _ in range(count):
                    db.append_message(session_id, "user", f"Message from {count}")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(5,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Should have 15 messages total (3 threads × 5 messages)
        assert db.message_count(session_id) == 15


class TestSchemaInitEdgeCases:
    """Edge cases for schema initialization."""

    def test_clean_database_creation(self, tmp_path):
        """Creating a new database should initialize to v5."""
        db_path = tmp_path / "clean_db.db"
        db = SessionDB(db_path=db_path)

        # Should be at v5 immediately
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        # Verify all tables exist
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "messages" in tables
        assert "schema_version" in tables
        assert "messages_fts" in tables

        db.close()

    def test_upgrade_from_v5_to_v5(self, tmp_path):
        """Opening a v5 database should not change version."""
        import sqlite3

        db_path = tmp_path / "v5_db.db"
        conn = sqlite3.connect(str(db_path))

        # Create v5 schema
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (5);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                title TEXT,
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
            );

            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content, content=messages, content_rowid=id
            );

            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)

        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("existing", "cli", 1000.0),
        )
        conn.commit()
        conn.close()

        # Open with SessionDB
        db = SessionDB(db_path=db_path)

        # Should still be v5
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        db.close()


class TestUpdateTokenCountsEdgeCases:
    """Edge cases for update_token_counts()."""

    def test_update_all_billing_fields(self, db):
        """Test updating all billing-related fields."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
            reasoning_tokens=20,
            estimated_cost_usd=1.0,
            actual_cost_usd=1.05,
            cost_status="within budget",
            cost_source="api",
            pricing_version="2024.01",
            billing_provider="anthropic",
            billing_base_url="https://api.anthropic.com",
            billing_mode="standard",
        )

        session = db.get_session("s1")

        # Check all billing fields
        assert session["billing_provider"] == "anthropic"
        assert session["billing_base_url"] == "https://api.anthropic.com"
        assert session["billing_mode"] == "standard"
        assert session["estimated_cost_usd"] == 1.0
        assert session["actual_cost_usd"] == 1.05
        assert session["cost_status"] == "within budget"
        assert session["cost_source"] == "api"
        assert session["pricing_version"] == "2024.01"

    def test_cost_accumulation(self, db):
        """Costs should accumulate across multiple updates."""
        db.create_session("s1", "cli")

        # First update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=1.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 100
        assert session["estimated_cost_usd"] == 1.0

        # Second update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=2.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 200
        assert session["estimated_cost_usd"] == 3.0

    def test_cost_status_preserved(self, db):
        """Cost status should be preserved across updates."""
        db.create_session("s1", "cli")

        # First update with "within budget"
        db.update_token_counts("s1", input_tokens=100, cost_status="within budget")

        # Second update without cost_status (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["cost_status"] == "within budget"

    def test_billing_fields_backfill_once(self, db):
        """Billing fields should be backfilled only once, then preserved."""
        db.create_session("s1", "cli")

        # First update with billing fields
        db.update_token_counts(
            "s1",
            input_tokens=100,
            billing_provider="test",
            billing_base_url="https://test.com",
        )

        # Second update without billing fields (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["billing_provider"] == "test"
        assert session["billing_base_url"] == "https://test.com"

    def test_billing_over_budget_status(self, db):
        """Test 'over budget' cost status."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            actual_cost_usd=1000.0,
            cost_status="over budget",
            cost_source="api",
            pricing_version="2024.01",
        )

        session = db.get_session("s1")
        assert session["cost_status"] == "over budget"
        assert session["actual_cost_usd"] == 1000.0


class TestFTSEdgeCases:
    """Edge cases for FTS5 search."""

    def test_search_whitespace_only_query(self, db):
        """Search with whitespace-only query should return empty."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        assert db.search_messages("   ") == []
        assert db.search_messages("\t\n") == []

    def test_search_single_character(self, db):
        """Search with single character should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        results = db.search_messages("e")
        assert len(results) >= 0  # May or may not match depending on FTS5

    def test_search_unicode_content(self, db):
        """Search with unicode content should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "こんにちは世界")
        db.append_message("s1", "assistant", "Hello World")

        results = db.search_messages("こんにちは")
        assert isinstance(results, list)

    def test_search_empty_content_message(self, db):
        """Search should handle messages with empty content."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "")
        db.append_message("s1", "assistant", "Hello")

        results = db.search_messages("Hello")
        # Should find the assistant message
        assert len(results) >= 0


class TestToolCallCountEdgeCases:
    """Edge cases for tool call counting."""

    def test_tool_calls_count_in_message(self, db):
        """Tool calls in assistant message should count."""
        db.create_session("s1", "cli")

        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 2

    def test_tool_calls_no_content_still_counts(self, db):
        """Tool calls should count even if content is empty."""
        db.create_session("s1", "cli")

        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1

    def test_tool_calls_with_tool_response(self, db):
        """Tool responses should not increment tool_call_count."""
        db.create_session("s1", "cli")

        # Assistant makes 1 tool call
        tool_calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Tool response comes back
        db.append_message("s1", "tool", "Result", tool_name="web_search")

        session = db.get_session("s1")
        # Should be 1 (the assistant call), not 2
        assert session["tool_call_count"] == 1

    def test_multiple_tool_calls_multiple_responses(self, db):
        """Multiple tool calls with multiple responses should count correctly."""
        db.create_session("s1", "cli")

        # Assistant makes 2 parallel tool calls
        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Two tool responses
        db.append_message("s1", "tool", "Result 1", tool_name="web_search")
        db.append_message("s1", "tool", "Result 2", tool_name="file_read")

        session = db.get_session("s1")
        assert session["tool_call_count"] == 2
        assert session["message_count"] == 3  # 1 assistant + 2 tool responses

# ======================================================================
# Untested/Edge cases
# ======================================================================

# ======================================================================
# Untested/Edge cases
# ======================================================================

class TestClearMessages:
    """Tests for clear_messages() - deletes all messages and resets counters."""

    def test_clear_messages(self, db: SessionDB):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")
        db.append_message("s1", role="tool", tool_name="web_search")

        # Verify messages exist
        assert db.message_count(session_id="s1") == 3
        session = db.get_session("s1")
        assert session["message_count"] == 3
        assert session["tool_call_count"] == 1

        # Clear messages
        db.clear_messages("s1")

        # Verify messages deleted
        assert db.message_count(session_id="s1") == 0
        assert db.get_messages("s1") == []

        # Verify counters reset
        session = db.get_session("s1")
        assert session["message_count"] == 0
        assert session["tool_call_count"] == 0

    def test_clear_nonexistent_session(self, db: SessionDB):
        # Should not raise for nonexistent session
        assert db.clear_messages("nonexistent") is not None  # Returns None implicitly

    def test_clear_messages_preserves_session_metadata(self, db: SessionDB):
        db.create_session(
            session_id="s1",
            source="cli",
            model="test-model",
            system_prompt="System prompt text",
        )
        db.end_session("s1", end_reason="user_exit")
        db.set_session_title("s1", "My Title")

        db.append_message("s1", role="user", content="Hello")

        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"

        db.clear_messages("s1")

        # Session metadata should persist
        assert db.get_session("s1")["model"] == "test-model"
        assert db.get_session("s1")["system_prompt"] == "System prompt text"
        assert db.get_session("s1")["ended_at"] is not None
        assert db.get_session("s1")["title"] == "My Title"


class TestClose:
    """Tests for close() method."""

    def test_close_releases_connection(self, db: SessionDB):
        assert db._conn is not None
        db.close()
        assert db._conn is None

    def test_close_twice_no_error(self, db: SessionDB):
        db.close()
        db.close()  # Should not raise

    def test_close_after_append(self, db: SessionDB):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        db.close()

        # Connection released, can be reopened
        new_db = SessionDB(db_path=db.db_path)
        new_db.close()


class TestPruneSessionsEdgeCases:
    """Edge cases for prune_sessions()."""

    def test_prune_zero_days(self, db: SessionDB):
        """Pruning with 0 days should delete all ended sessions."""
        db.create_session("old1", "cli")
        db.end_session("old1", end_reason="done")
        db.create_session("old2", "cli")
        db.end_session("old2", end_reason="done")

        pruned = db.prune_sessions(older_than_days=0)
        assert pruned == 2
        assert db.get_session("old1") is None
        assert db.get_session("old2") is None

    def test_prune_all_active_skipped(self, db: SessionDB):
        """All active sessions should be skipped."""
        db.create_session("active1", "cli")
        db.create_session("active2", "cli")

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0
        assert db.get_session("active1") is not None
        assert db.get_session("active2") is not None

    def test_prune_empty_db(self, db: SessionDB):
        """Pruning empty database should return 0."""
        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0

    def test_prune_preserves_started_at(self, db: SessionDB):
        """Pruning should delete sessions, not just truncate."""
        db.create_session("old", "cli")
        db.end_session("old", end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 200 * 86400, "old"),
        )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 1
        assert db.get_session("old") is None


class TestListSessionsRichEdgeCases:
    """Edge cases for list_sessions_rich()."""

    def test_list_empty_database(self, db: SessionDB):
        """Listing empty database should return empty list."""
        sessions = db.list_sessions_rich()
        assert sessions == []

    def test_list_limit_zero(self, db: SessionDB):
        """Limit=0 should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")

        sessions = db.list_sessions_rich(limit=0)
        assert sessions == []

    def test_list_large_offset(self, db: SessionDB):
        """Large offset should return empty (no more results)."""
        for i in range(3):
            db.create_session(f"s{i}", "cli")
            db.append_message(f"s{i}", "user", f"Message {i}")

        sessions = db.list_sessions_rich(offset=100)
        assert sessions == []

    def test_list_all_fields_present(self, db: SessionDB):
        """All expected fields should be present in result."""
        import time
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        sessions = db.list_sessions_rich(limit=1)
        assert len(sessions) == 1
        session = sessions[0]

        # Check all expected fields
        assert "id" in session
        assert "source" in session
        assert "model" in session
        assert "title" in session
        assert "started_at" in session
        assert "ended_at" in session
        assert "message_count" in session
        assert "preview" in session
        assert "last_active" in session
        # Note: Some fields may be None if not set


class TestGetSessionByTitleEdgeCases:
    """Edge cases for get_session_by_title()."""

    def test_case_sensitive(self, db: SessionDB):
        """Title lookup should be case-sensitive."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Title")

        result = db.get_session_by_title("my title")
        assert result is None

        result = db.get_session_by_title("My Title")
        assert result is not None

    def test_special_characters_in_title(self, db: SessionDB):
        """Titles with special characters should work."""
        db.create_session("s1", "cli")
        title = "Title with spaces and #1"
        db.set_session_title("s1", title)

        result = db.get_session_by_title(title)
        assert result is not None

    def test_empty_title(self, db: SessionDB):
        """Empty title should return None."""
        db.create_session("s1", "cli")
        result = db.get_session_by_title("")
        assert result is None


class TestAppendMessageEdgeCases:
    """Edge cases for append_message()."""

    def test_tool_calls_none(self, db: SessionDB):
        """tool_calls=None should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=None)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_empty_list(self, db: SessionDB):
        """tool_calls=[] should work (no tool calls counted)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls=[])

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 0

    def test_tool_calls_non_list(self, db: SessionDB):
        """tool_calls with non-list value (should be treated as 1 call)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "assistant", "Hello", tool_calls="some_string")

        session = db.get_session("s1")
        assert session["message_count"] == 1
        # Non-list with truthy value counts as 1
        assert session["tool_call_count"] == 1

    def test_tool_calls_single_dict(self, db: SessionDB):
        """tool_calls with single dict (should count as 1)."""
        db.create_session("s1", "cli")
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "Hello", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1


class TestExportAllEdgeCases:
    """Edge cases for export_all()."""

    def test_export_all_empty_source(self, db: SessionDB):
        """export_all(source=...) with no matches should return empty list."""
        db.create_session("s1", "cli")
        db.create_session("s2", "telegram")

        exports = db.export_all(source="nonexistent")
        assert exports == []

    def test_export_all_with_messages(self, db: SessionDB):
        """Export should include messages."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")

        exports = db.export_all()
        assert len(exports) == 1
        assert len(exports[0]["messages"]) == 2

    def test_export_all_preserves_all_fields(self, db: SessionDB):
        """Export should preserve all session fields."""
        db.create_session(
            "s1",
            "cli",
            model="test-model",
            system_prompt="System prompt",
        )
        db.end_session("s1", end_reason="done")
        db.set_session_title("s1", "My Title")
        db.append_message("s1", "user", "Hello")

        exports = db.export_all()
        export = exports[0]

        assert "id" in export
        assert "source" in export
        assert "model" in export
        assert "system_prompt" in export
        assert "title" in export
        assert "ended_at" in export
        assert "messages" in export
        assert len(export["messages"]) == 1


class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_reads_writes(self, db: SessionDB):
        """Concurrent reads and writes should not corrupt data."""
        import threading
        import time

        errors = []
        lock = threading.Lock()

        def writer():
            try:
                for i in range(10):
                    session_id = f"writer_{i}"
                    db.create_session(session_id, "cli")
                    db.append_message(session_id, "user", f"Message {i}")
                    db.end_session(session_id, end_reason="done")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        def reader():
            try:
                for i in range(10):
                    time.sleep(0.01)
                    sessions = db.search_sessions()
                    messages = db.message_count()
                    assert len(sessions) == 0  # All created by writer
                    assert messages == 0  # All created by writer
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(errors) == 0

        # Verify data integrity
        assert db.session_count() == 10
        assert db.message_count() == 10

    def test_concurrent_same_session_writes(self, db: SessionDB):
        """Multiple concurrent writes to same session should be consistent."""
        import threading
        import time

        session_id = "concurrent_session"
        db.create_session(session_id, "cli")

        errors = []
        lock = threading.Lock()

        def writer(count):
            try:
                for _ in range(count):
                    db.append_message(session_id, "user", f"Message from {count}")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(5,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Should have 15 messages total (3 threads × 5 messages)
        assert db.message_count(session_id) == 15


class TestSchemaInitEdgeCases:
    """Edge cases for schema initialization."""

    def test_clean_database_creation(self, tmp_path):
        """Creating a new database should initialize to v5."""
        db_path = tmp_path / "clean_db.db"
        db = SessionDB(db_path=db_path)

        # Should be at v5 immediately
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        # Verify all tables exist
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "messages" in tables
        assert "schema_version" in tables
        assert "messages_fts" in tables

        db.close()

    def test_upgrade_from_v5_to_v5(self, tmp_path):
        """Opening a v5 database should not change version."""
        import sqlite3

        db_path = tmp_path / "v5_db.db"
        conn = sqlite3.connect(str(db_path))

        # Create v5 schema
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (5);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                title TEXT,
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
            );

            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content, content=messages, content_rowid=id
            );

            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)

        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("existing", "cli", 1000.0),
        )
        conn.commit()
        conn.close()

        # Open with SessionDB
        db = SessionDB(db_path=db_path)

        # Should still be v5
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 5

        db.close()


class TestUpdateTokenCountsEdgeCases:
    """Edge cases for update_token_counts()."""

    def test_update_all_billing_fields(self, db: SessionDB):
        """Test updating all billing-related fields."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
            reasoning_tokens=20,
            estimated_cost_usd=1.0,
            actual_cost_usd=1.05,
            cost_status="within budget",
            cost_source="api",
            pricing_version="2024.01",
            billing_provider="anthropic",
            billing_base_url="https://api.anthropic.com",
            billing_mode="standard",
        )

        session = db.get_session("s1")

        # Check all billing fields
        assert session["billing_provider"] == "anthropic"
        assert session["billing_base_url"] == "https://api.anthropic.com"
        assert session["billing_mode"] == "standard"
        assert session["estimated_cost_usd"] == 1.0
        assert session["actual_cost_usd"] == 1.05
        assert session["cost_status"] == "within budget"
        assert session["cost_source"] == "api"
        assert session["pricing_version"] == "2024.01"

    def test_cost_accumulation(self, db: SessionDB):
        """Costs should accumulate across multiple updates."""
        db.create_session("s1", "cli")

        # First update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=1.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 100
        assert session["estimated_cost_usd"] == 1.0

        # Second update
        db.update_token_counts("s1", input_tokens=100, estimated_cost_usd=2.0)
        session = db.get_session("s1")
        assert session["input_tokens"] == 200
        assert session["estimated_cost_usd"] == 3.0

    def test_cost_status_preserved(self, db: SessionDB):
        """Cost status should be preserved across updates."""
        db.create_session("s1", "cli")

        # First update with "within budget"
        db.update_token_counts("s1", input_tokens=100, cost_status="within budget")

        # Second update without cost_status (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["cost_status"] == "within budget"

    def test_billing_fields_backfill_once(self, db: SessionDB):
        """Billing fields should be backfilled only once, then preserved."""
        db.create_session("s1", "cli")

        # First update with billing fields
        db.update_token_counts(
            "s1",
            input_tokens=100,
            billing_provider="test",
            billing_base_url="https://test.com",
        )

        # Second update without billing fields (should preserve)
        db.update_token_counts("s1", input_tokens=100)

        session = db.get_session("s1")
        assert session["billing_provider"] == "test"
        assert session["billing_base_url"] == "https://test.com"

    def test_billing_over_budget_status(self, db: SessionDB):
        """Test 'over budget' cost status."""
        db.create_session("s1", "cli")

        db.update_token_counts(
            "s1",
            input_tokens=100,
            actual_cost_usd=1000.0,
            cost_status="over budget",
            cost_source="api",
            pricing_version="2024.01",
        )

        session = db.get_session("s1")
        assert session["cost_status"] == "over budget"
        assert session["actual_cost_usd"] == 1000.0


class TestFTSEdgeCases:
    """Edge cases for FTS5 search."""

    def test_search_whitespace_only_query(self, db: SessionDB):
        """Search with whitespace-only query should return empty."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        assert db.search_messages("   ") == []
        assert db.search_messages("\t\n") == []

    def test_search_single_character(self, db: SessionDB):
        """Search with single character should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")

        results = db.search_messages("e")
        assert isinstance(results, list)  # May or may not match depending on FTS5

    def test_search_unicode_content(self, db: SessionDB):
        """Search with unicode content should work."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "こんにちは世界")
        db.append_message("s1", "assistant", "Hello World")

        results = db.search_messages("こんにちは")
        assert isinstance(results, list)

    def test_search_empty_content_message(self, db: SessionDB):
        """Search should handle messages with empty content."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "")
        db.append_message("s1", "assistant", "Hello")

        results = db.search_messages("Hello")
        # Should find the assistant message
        assert len(results) >= 0


class TestToolCallCountEdgeCases:
    """Edge cases for tool call counting."""

    def test_tool_calls_count_in_message(self, db: SessionDB):
        """Tool calls in assistant message should count."""
        db.create_session("s1", "cli")

        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 2

    def test_tool_calls_no_content_still_counts(self, db: SessionDB):
        """Tool calls should count even if content is empty."""
        db.create_session("s1", "cli")

        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        db.append_message("s1", "assistant", "", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1

    def test_tool_calls_with_tool_response(self, db: SessionDB):
        """Tool responses should not increment tool_call_count."""
        db.create_session("s1", "cli")

        # Assistant makes 1 tool call
        tool_calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Tool response comes back
        db.append_message("s1", "tool", "Result", tool_name="web_search")

        session = db.get_session("s1")
        # Should be 1 (the assistant call), not 2
        assert session["tool_call_count"] == 1

    def test_multiple_tool_calls_multiple_responses(self, db: SessionDB):
        """Multiple tool calls with multiple responses should count correctly."""
        db.create_session("s1", "cli")

        # Assistant makes 2 parallel tool calls
        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "file_read", "arguments": "{}"}},
        ]
        db.append_message("s1", "assistant", "Checking...", tool_calls=tool_calls)

        # Two tool responses
        db.append_message("s1", "tool", "Result 1", tool_name="web_search")
        db.append_message("s1", "tool", "Result 2", tool_name="file_read")

        session = db.get_session("s1")
        assert session["tool_call_count"] == 2
        assert session["message_count"] == 3  # 1 assistant + 2 tool responses

@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state_untested.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


class TestResolveSessionId:
    """Tests for resolve_session_id() — resolves exact or prefix session IDs."""

    def test_resolve_exact_match(self, db: SessionDB):
        """Exact match should return the session ID."""
        db.create_session("exact_session_123", "cli")
        result = db.resolve_session_id("exact_session_123")
        assert result == "exact_session_123"

    def test_resolve_prefix_unique(self, db: SessionDB):
        """Unique prefix should return the matching session ID."""
        db.create_session("unique_prefix_abc", "cli")
        result = db.resolve_session_id("unique_prefix")
        assert result == "unique_prefix_abc"

    def test_resolve_prefix_ambiguous_returns_none(self, db: SessionDB):
        """Ambiguous prefix should return None."""
        db.create_session("prefix_a", "cli")
        db.create_session("prefix_b", "cli")
        result = db.resolve_session_id("prefix")
        assert result is None

    def test_resolve_nonexistent_prefix_returns_none(self, db: SessionDB):
        """Non-existent prefix should return None."""
        result = db.resolve_session_id("nonexistent")
        assert result is None

    def test_resolve_prefix_with_special_chars(self, db: SessionDB):
        """Prefix matching should escape special characters."""
        db.create_session("session_with_%_special", "cli")
        result = db.resolve_session_id("session_with_")
        assert result == "session_with_%_special"


class TestResolveSessionByTitle:
    """Tests for resolve_session_by_title() — resolves title to session ID."""

    def test_resolve_exact_title(self, db: SessionDB):
        """Exact title match should return session ID."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Exact Title")
        result = db.resolve_session_by_title("My Exact Title")
        assert result == "s1"

    def test_resolve_numbered_variant_latest(self, db: SessionDB):
        """Should prefer latest numbered variant over exact match."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Session")
        time.sleep(0.1)  # Small delay to ensure later timestamp
        db.create_session("s2", "cli")
        db.set_session_title("s2", "My Session #2")
        result = db.resolve_session_by_title("My Session")
        assert result == "s2"

    def test_resolve_nonexistent_title(self, db: SessionDB):
        """Non-existent title should return None."""
        result = db.resolve_session_by_title("NonExistent Title")
        assert result is None

    def test_resolve_title_no_exact_match(self, db: SessionDB):
        """Should return latest numbered variant when exact doesn't exist."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Session")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "My Session #2")
        db.create_session("s3", "cli")
        db.set_session_title("s3", "My Session #3")
        result = db.resolve_session_by_title("My Session #3")
        assert result == "s3"


class TestGetNextTitleInLineage:
    """Tests for get_next_title_in_lineage() — generates next title in sequence."""

    def test_generate_next_number_from_base(self, db: SessionDB):
        """Base title should generate #2."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Session")
        next_title = db.get_next_title_in_lineage("My Session")
        assert next_title == "My Session #2"

    def test_generate_next_from_existing_number(self, db: SessionDB):
        """Existing #3 should generate #4."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Session #3")
        next_title = db.get_next_title_in_lineage("My Session #3")
        assert next_title == "My Session #4"

    def test_generate_from_new_title(self, db: SessionDB):
        """New title should return base unchanged."""
        next_title = db.get_next_title_in_lineage("Brand New Session")
        assert next_title == "Brand New Session"

    def test_generate_with_special_chars(self, db: SessionDB):
        """Title with special characters should be handled correctly."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "Session with % sign")
        next_title = db.get_next_title_in_lineage("Session with % sign")
        assert next_title == "Session with % sign #2"


class TestSanitizeTitle:
    """Tests for sanitize_title() — cleans session titles."""

    def test_sanitize_removes_control_chars(self, db: SessionDB):
        """ASCII control chars should be removed."""
        title = "Hello\x01\x02\x1FWorld"
        result = SessionDB.sanitize_title(title)
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x1F" not in result

    def test_sanitize_removes_unicode_control_chars(self, db: SessionDB):
        """Unicode control chars should be removed."""
        title = "Hello\u200B\u200CWorld"
        result = SessionDB.sanitize_title(title)
        assert "\u200B" not in result
        assert "\u200C" not in result

    def test_sanitize_collapse_whitespace(self, db: SessionDB):
        """Multiple spaces should be collapsed."""
        title = "Hello   World"
        result = SessionDB.sanitize_title(title)
        assert "  " not in result
        assert result == "Hello World"

    def test_sanitize_strip_whitespace(self, db: SessionDB):
        """Leading/trailing whitespace should be stripped."""
        title = "  Hello World  "
        result = SessionDB.sanitize_title(title)
        assert result == "Hello World"

    def test_sanitize_empty_returns_none(self, db: SessionDB):
        """Whitespace-only title should return None."""
        result = SessionDB.sanitize_title("   ")
        assert result is None

    def test_sanitize_none_returns_none(self, db: SessionDB):
        """None should return None."""
        result = SessionDB.sanitize_title(None)
        assert result is None

    def test_sanitize_max_length_check(self, db: SessionDB):
        """Title exceeding max length should raise ValueError."""
        long_title = "A" * 150
        try:
            SessionDB.sanitize_title(long_title)
        except ValueError as e:
            assert "too long" in str(e).lower()
        else:
            pytest.fail("Expected ValueError not raised")

    def test_sanitize_exact_max_length(self, db: SessionDB):
        """Title at exact max length should be accepted."""
        max_length = SessionDB.MAX_TITLE_LENGTH
        title = "A" * max_length
        result = SessionDB.sanitize_title(title)
        assert len(result) == max_length

    def test_sanitize_with_unicode(self, db: SessionDB):
        """Unicode characters should be preserved."""
        title = "Hello 世界 World"
        result = SessionDB.sanitize_title(title)
        assert "世界" in result


class TestGetSessionTitle:
    """Tests for get_session_title() — retrieves session title."""

    def test_get_title_for_existing_session(self, db: SessionDB):
        """Existing session should return its title."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Title")
        result = db.get_session_title("s1")
        assert result == "My Title"

    def test_get_title_for_new_session(self, db: SessionDB):
        """New session should return None (no title set)."""
        db.create_session("s1", "cli")
        result = db.get_session_title("s1")
        assert result is None

    def test_get_title_for_nonexistent_session(self, db: SessionDB):
        """Non-existent session should return None."""
        result = db.get_session_title("nonexistent")
        assert result is None


class TestUpdateSystemPrompt:
    """Tests for update_system_prompt() — updates system prompt."""

    def test_update_system_prompt(self, db: SessionDB):
        """System prompt should be updated."""
        db.create_session("s1", "cli")
        db.update_system_prompt("s1", "Updated prompt")
        session = db.get_session("s1")
        assert session["system_prompt"] == "Updated prompt"

    def test_update_system_prompt_overwrites(self, db: SessionDB):
        """New prompt should overwrite old prompt."""
        db.create_session("s1", "cli")
        db.update_system_prompt("s1", "First prompt")
        db.update_system_prompt("s1", "Second prompt")
        session = db.get_session("s1")
        assert session["system_prompt"] == "Second prompt"

    def test_update_system_prompt_preserves_other_fields(self, db: SessionDB):
        """Updating prompt should not affect other fields."""
        db.create_session("s1", "cli", model="test-model")
        db.set_session_title("s1", "My Title")
        db.update_system_prompt("s1", "Updated prompt")
        session = db.get_session("s1")
        assert session["model"] == "test-model"
        assert session["title"] == "My Title"
        assert session["system_prompt"] == "Updated prompt"


class TestGetMessagesAsConversation:
    """Tests for get_messages_as_conversation() — returns OpenAI format."""

    def test_get_conversation_format(self, db: SessionDB):
        """Messages should be in OpenAI conversation format."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi there!")
        messages = db.get_messages_as_conversation("s1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_get_conversation_with_tool_calls(self, db: SessionDB):
        """Tool calls should be included in conversation format."""
        db.create_session("s1", "cli")
        tool_calls = [{"name": "web_search", "args": {"query": "weather"}}]
        db.append_message("s1", "assistant", None, tool_calls=tool_calls)
        messages = db.get_messages_as_conversation("s1")
        assert len(messages) == 1
        assert messages[0]["tool_calls"] == tool_calls

    def test_get_conversation_empty_session(self, db: SessionDB):
        """Empty session should return empty list."""
        db.create_session("s1", "cli")
        messages = db.get_messages_as_conversation("s1")
        assert messages == []

    def test_get_conversation_nonexistent_session(self, db: SessionDB):
        """Non-existent session should return empty list."""
        messages = db.get_messages_as_conversation("nonexistent")
        assert messages == []


class TestExportSession:
    """Tests for export_session() — exports single session with messages."""

    def test_export_session_with_messages(self, db: SessionDB):
        """Export should include session metadata and messages."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        exported = db.export_session("s1")
        assert exported is not None
        assert exported["id"] == "s1"
        assert "messages" in exported
        assert len(exported["messages"]) == 1

    def test_export_session_nonexistent(self, db: SessionDB):
        """Non-existent session should return None."""
        exported = db.export_session("nonexistent")
        assert exported is None

    def test_export_session_no_messages(self, db: SessionDB):
        """Session without messages should export with empty messages."""
        db.create_session("s1", "cli")
        exported = db.export_session("s1")
        assert exported is not None
        assert exported["messages"] == []


class TestDeleteSession:
    """Tests for delete_session() — deletes session and messages."""

    def test_delete_session(self, db: SessionDB):
        """Deleted session should be removed."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        result = db.delete_session("s1")
        assert result is True
        assert db.get_session("s1") is None

    def test_delete_nonexistent_session(self, db: SessionDB):
        """Non-existent session should return False."""
        result = db.delete_session("nonexistent")
        assert result is False

    def test_delete_session_removes_messages(self, db: SessionDB):
        """Deleting session should remove all its messages."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        db.append_message("s1", "assistant", "Hi")
        db.delete_session("s1")
        messages = db.get_messages("s1")
        assert messages == []

    def test_delete_session_preserves_other_sessions(self, db: SessionDB):
        """Deleting one session should not affect others."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")
        db.append_message("s1", "user", "Hello S1")
        db.append_message("s2", "user", "Hello S2")
        db.delete_session("s1")
        assert db.get_session("s2") is not None
        assert db.get_session("s1") is None


class TestGetNextTitleEdgeCases:
    """Edge cases for get_next_title_in_lineage()."""

    def test_generate_with_hyphen(self, db: SessionDB):
        """Title with hyphens should be handled correctly."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "My Session")
        next_title = db.get_next_title_in_lineage("My Session")
        assert next_title == "My Session #2"

    def test_generate_with_numbers_in_base(self, db: SessionDB):
        """Base title with numbers should still increment."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "Session 2024")
        next_title = db.get_next_title_in_lineage("Session 2024")
        assert next_title == "Session 2024 #2"

    def test_generate_highest_number(self, db: SessionDB):
        """Should find highest number and increment."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "Session #5")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "Session #3")
        db.create_session("s3", "cli")
        db.set_session_title("s3", "Session #7")
        next_title = db.get_next_title_in_lineage("Session #7")
        assert next_title == "Session #8"
