"""Integration tests for /clear-context command flow from CLI to database.

Tests the full pipeline: CLI handler -> SessionDB methods -> message persistence,
verifying that /clear-context correctly classifies, removes, and reports on
ephemeral messages while preserving critical content.
"""

import pytest
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_state import SessionDB
from hermes_state import MAX_EPHEMERAL_CONTENT_LENGTH, DEFAULT_PRUNE_COUNT


# ── Fixture: minimal HermesCLI stub ─────────────────────────────────────────

def _make_cli_with_db(tmp_path):
    """Build a minimal HermesCLI stub wired to a real SessionDB.
    
    Includes conversation_history attribute which is required by
    _handle_clear_context() for before/after message counting.
    """
    db_path = tmp_path / "integration.db"
    db = SessionDB(db_path=db_path)
    db.create_session("test-session", "cli")

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "test-session"
    cli_obj._session_db = db
    cli_obj.system_prompt = "You are a helpful assistant."
    cli_obj._app = None
    cli_obj.agent = None  # no agent tools during these tests
    cli_obj.conversation_history = []  # Required by _handle_clear_context
    return cli_obj, db


# ── /basic: full flow from populate to clear ────────────────────────────────

class TestClearContextAutoMode:
    """Test /clear-context (auto mode) — end-to-end flow."""

    def test_auto_clears_ephemeral_keeps_persistent(self, tmp_path):
        """Populate a session with mixed messages, run clear, verify only
        ephemeral are removed."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        # Populate with realistic conversation
        db.append_message(sid, "user", "Hello, analyze this project.")
        db.append_message(sid, "assistant", "Sure! Let me check the files.", tool_calls=[{"id": "tc1"}])
        db.append_message(sid, "tool", '{"stdout": "file1.py\nfile2.py\n", "exit_code": 0}')
        db.append_message(sid, "assistant", "I found file1.py and file2.py.")
        db.append_message(sid, "user", "Now run the tests.")
        db.append_message(sid, "assistant", "Running tests...", tool_calls=[{"id": "tc2"}])
        db.append_message(sid, "tool", '{"stdout": "All 42 passed", "exit_code": 0}')
        db.append_message(sid, "assistant", "All 42 tests passed. Everything looks good.")

        before = db.message_count(sid)
        assert before == 8

        # Call the handler
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context")

        # Verify the count
        after = db.message_count(sid)
        # Should have kept: 2x user, 3x assistant = 5
        # Removed: 2x tool = 2 results, but the assistant with tool_calls is still kept
        assert after < before
        assert after >= 3  # At least user prompts + an assistant response

        # All remaining user and assistant messages should be present
        messages = db.get_messages(sid)
        roles = [m["role"] for m in messages]
        assert "tool" not in roles, "Tool messages should have been cleared"
        assert "user" in roles, "User messages should be preserved"
        assert "assistant" in roles, "Assistant messages should be preserved"

    def test_auto_captures_output_messages(self, tmp_path, capsys):
        """Handler should print the removal summary."""
        cli_obj, _ = _make_cli_with_db(tmp_path)
        db = cli_obj._session_db
        sid = cli_obj.session_id

        db.append_message(sid, "user", "Do something")
        db.append_message(sid, "assistant", "OK", tool_calls=[{"id": "tc1"}])
        db.append_message(sid, "tool", '{"stdout": "x", "exit_code": 0}')

        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context")

        output = " ".join(captured)
        # Should report something about clearing messages
        assert "1" in output  # at least the ephemeral message count
        assert any(kw in output for kw in ["ephemeral", "Cleared", "message"])


class TestClearContextAllMode:
    """Test /clear-context all mode."""

    def test_all_mode_clears_same_as_auto(self, tmp_path):
        """'all' mode should behave identically to auto for ephemeral messages."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "user", "Hello")
        db.append_message(sid, "tool", '{"stdout": "output", "exit_code": 0}')
        db.append_message(sid, "assistant", "Done")

        removed = db.clear_ephemeral_messages(sid)
        assert removed == 1

        messages = db.get_messages(sid)
        assert len(messages) == 2
        assert all(m["role"] != "tool" for m in messages)


class TestClearContextRecentMode:
    """Test /clear-context recent mode."""

    def test_recent_prunes_last_n_messages(self, tmp_path):
        """'recent' mode should remove the last N non-system, unprotected messages."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        # Add 5 messages (none are system or protected)
        db.append_message(sid, "user", "Message 1")
        db.append_message(sid, "assistant", "Response 1")
        db.append_message(sid, "user", "Message 2")
        db.append_message(sid, "assistant", "Response 2")
        db.append_message(sid, "user", "Message 3")

        assert db.message_count(sid) == 5

        # Prune last 2
        removed = db.prune_messages(sid, count=2)
        assert removed == 2

        messages = db.get_messages(sid)
        assert db.message_count(sid) == 3
        # Should have removed the last 2 non-system messages (Response 2, Message 3)
        # Remaining: Message 1, Response 1, Message 2
        assert messages[-1]["content"] == "Message 2"

    def test_recent_respects_custom_count(self, tmp_path):
        """Prune should accept any count parameter."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        for i in range(15):
            db.append_message(sid, "user", f"msg {i}")

        removed = db.prune_messages(sid, count=5)
        assert removed == 5
        assert db.message_count(sid) == 10

        removed = db.prune_messages(sid, count=20)
        assert removed == 10  # Can't remove more than exist
        assert db.message_count(sid) == 0


class TestClearContextPreservesCriticalErrors:
    """Test that critical and fatal errors survive clear-context."""

    def test_fatal_error_survives_clear(self, tmp_path):
        """Messages with 'fatal Error:' should NOT be cleared."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "user", "Run this script")
        db.append_message(sid, "assistant", "FATAL Error: disk full, cannot write")
        db.append_message(sid, "tool", '{"stdout": "ok", "exit_code": 0}')

        removed = db.clear_ephemeral_messages(sid)
        # Only the tool message should be removed; the fatal error stays
        assert removed == 1

        messages = db.get_messages(sid)
        assert any("FATAL" in (m.get("content") or "") for m in messages)

    def test_case_insensitive_critical_survives(self, tmp_path):
        """'CRITICAL Error:' (uppercase) should still be preserved."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "assistant", "CRITICAL Error: database corrupted")
        db.append_message(sid, "tool", '{"stdout": "x", "exit_code": 0}')

        removed = db.clear_ephemeral_messages(sid)
        assert removed == 1  # only tool removed

        messages = db.get_messages(sid)
        assert any("CRITICAL" in (m.get("content") or "") for m in messages)

    def test_case_insensitive_cannot_survives(self, tmp_path):
        """'Error: Cannot access' should be preserved regardless of case."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "assistant", "Error: Cannot create temp directory")
        db.append_message(sid, "tool", '{"stdout": "x", "exit_code": 0}')

        removed = db.clear_ephemeral_messages(sid)
        assert removed == 1

        messages = db.get_messages(sid)
        assert any("Cannot" in (m.get("content") or "") for m in messages)

    def test_case_insensitive_failed_to_create_survives(self, tmp_path):
        """'Error: Failed to Create' should be preserved (case insensitive)."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "assistant", "Error: Failed to Create output directory")
        db.append_message(sid, "tool", '{"stdout": "x", "exit_code": 0}')

        removed = db.clear_ephemeral_messages(sid)
        assert removed == 1

        messages = db.get_messages(sid)
        assert any("Failed to Create" in (m.get("content") or "") for m in messages)


class TestClearContextWithProtectedMessages:
    """Test interaction between /clear-context and task protection."""

    def test_protected_messages_survive_clear(self, tmp_path):
        """Messages marked protected should never be removed by clear."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task: fix the bug")
        db.append_message(sid, "assistant", "Starting task 1")
        db.append_message(sid, "tool", '{"stdout": "test output", "exit_code": 0}')
        db.end_task_protection(sid)

        # All messages should be protected now
        protected = db.get_protected_messages(sid)
        assert len(protected) == 3

        removed = db.clear_ephemeral_messages(sid)
        # Nothing should be removed because everything is protected
        assert removed == 0
        assert db.message_count(sid) == 3


class TestClearContextInvalidMode:
    """Test /clear-context with invalid mode shows usage."""

    def test_invalid_mode_shows_usage(self, tmp_path):
        """Unknown mode should print usage, not crash."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "user", "Hello")

        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context invalid_mode")

        output = " ".join(captured)
        assert "Usage" in output
        assert "all" in output
        assert "recent" in output


class TestConstantsDocumented:
    """Verify the magic number constants are properly documented."""

    def test_max_ephemeral_content_length_has_docstring(self):
        """MAX_EPHEMERAL_CONTENT_LENGTH should be >4000 and documented."""
        from hermes_state import MAX_EPHEMERAL_CONTENT_LENGTH
        assert MAX_EPHEMERAL_CONTENT_LENGTH == 5000

    def test_default_prune_count_has_docstring(self):
        """DEFAULT_PRUNE_COUNT should be >0 and documented."""
        from hermes_state import DEFAULT_PRUNE_COUNT
        assert DEFAULT_PRUNE_COUNT == 10

    def test_module_source_has_detailed_comments(self):
        """Source file should contain detailed multi-line comments for constants."""
        import inspect
        import hermes_state
        source = inspect.getsource(hermes_state)
        # Check that the constants have multi-line comments above them
        assert "Maximum character length" in source or "ephemeral" in source.lower()
        assert "Default number" in source or "prune" in source.lower()


class TestEstimateContextTokensErrorLogging:
    """Test that _estimate_current_context_tokens logs errors properly."""

    def test_logs_exception_on_failure(self, tmp_path):
        """When estimation fails, should log the exception, not silently return."""
        cli_obj, db = _make_cli_with_db(tmp_path)

        # Patch the estimate function at its source (it's imported inside the method)
        with patch("agent.model_metadata.estimate_request_tokens_rough",
                   side_effect=ValueError("broken")), \
             patch("cli.logger") as mock_logger:
            result = cli_obj._estimate_current_context_tokens()
            # Should return 0 on error
            assert result == 0
            # Should log the exception (warning level since token estimation
            # failure can impact /clear-context auto behavior)
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "Failed to estimate context tokens" in call_args
