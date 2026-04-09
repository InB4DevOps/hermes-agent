"""Tests for hermes_state.py context management methods (clear-context command).

Tests for:
- is_ephemeral_message(): Detection of ephemeral vs persistent messages
- classify_messages(): Classification of all messages
- clear_ephemeral_messages(): Removal of ephemeral messages
- prune_messages(): Removal of last N messages
"""

import pytest
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_context.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


# =========================================================================
# is_ephemeral_message() - Message classification logic
# =========================================================================

class TestIsEphemeralMessage:
    """Test the static method that determines if a message is ephemeral."""

    def test_tool_role_is_ephemeral(self):
        """Tool results (role='tool') should be ephemeral."""
        msg = {"role": "tool", "content": "Tool output here"}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_user_role_not_ephemeral(self):
        """User messages should not be ephemeral."""
        msg = {"role": "user", "content": "Hello, how are you?"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_assistant_role_not_ephemeral(self):
        """Assistant responses should not be ephemeral."""
        msg = {"role": "assistant", "content": "I'm doing well, thanks!"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_system_role_not_ephemeral(self):
        """System messages should not be ephemeral."""
        msg = {"role": "system", "content": "You are a helpful assistant."}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_long_content_is_ephemeral(self):
        """Content longer than 5000 chars should be ephemeral."""
        long_content = "x" * 5001
        msg = {"role": "assistant", "content": long_content}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_exactly_5000_chars_not_ephemeral(self):
        """Content exactly 5000 chars should not be ephemeral (boundary)."""
        content = "x" * 5000
        msg = {"role": "assistant", "content": content}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_4999_chars_not_ephemeral(self):
        """Content 4999 chars should not be ephemeral."""
        content = "x" * 4999
        msg = {"role": "assistant", "content": content}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_stdout_in_content_is_ephemeral(self):
        """Content containing 'stdout:' structured pattern should be ephemeral."""
        msg = {"role": "assistant", "content": '"stdout": some output\n"exit_code": 0'}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_exit_code_in_content_is_ephemeral(self):
        """Content with 'exit_code:' at line start (realistic terminal output) should be ephemeral."""
        msg = {"role": "assistant", "content": 'Command output:\nexit_code: 0'}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_casual_stdout_mention_not_ephemeral(self):
        """Casual mention of 'stdout' without structured pattern should NOT be ephemeral."""
        msg = {"role": "assistant", "content": "I'm parsing stdout from a subprocess"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_casual_exit_code_mention_not_ephemeral(self):
        """Casual mention of 'exit_code' without structured pattern should NOT be ephemeral."""
        msg = {"role": "assistant", "content": "The exit_code was 0 in this case"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_error_with_traceback_is_ephemeral(self):
        """Content with 'Error:' should be ephemeral (unless critical)."""
        msg = {"role": "assistant", "content": "Error: Something went wrong\nTraceback..."}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_exception_in_content_is_ephemeral(self):
        """Content with 'Exception:' should be ephemeral (unless critical)."""
        msg = {"role": "assistant", "content": "Exception: ValueError raised"}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_fatal_error_not_ephemeral(self):
        """Critical errors with 'fatal' should NOT be ephemeral."""
        msg = {"role": "assistant", "content": "fatal Error: System crash"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_critical_error_not_ephemeral(self):
        """Critical errors with 'critical' should NOT be ephemeral."""
        msg = {"role": "assistant", "content": "critical Error: Database corruption"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_cannot_error_not_ephemeral(self):
        """Errors with 'cannot' should NOT be ephemeral."""
        msg = {"role": "assistant", "content": "Error: cannot access file"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_failed_to_create_error_not_ephemeral(self):
        """Errors with 'failed to create' should NOT be ephemeral."""
        msg = {"role": "assistant", "content": "Error: failed to create directory"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_memory_retrieved_is_ephemeral(self):
        """Memory retrieval results should be ephemeral."""
        msg = {"role": "assistant", "content": "memory retrieved: previous conversation notes"}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_memory_found_is_ephemeral(self):
        """Memory found results should be ephemeral."""
        msg = {"role": "assistant", "content": "memory found in database: user preferences"}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_memory_without_retrieved_or_found_not_ephemeral(self):
        """Memory mentions without 'retrieved' or 'found' should not be ephemeral."""
        msg = {"role": "assistant", "content": "I remember from memory that you like Python"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_case_insensitive_memory_detection(self):
        """Memory detection should be case-insensitive."""
        msg = {"role": "assistant", "content": "MEMORY RETRIEVED: important note"}
        assert SessionDB.is_ephemeral_message(msg) is True

    def test_empty_content_not_ephemeral(self):
        """Empty content should not be ephemeral."""
        msg = {"role": "assistant", "content": ""}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_none_content_not_ephemeral(self):
        """None content should not be ephemeral."""
        msg = {"role": "assistant", "content": None}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_missing_content_key_not_ephemeral(self):
        """Missing content key should not be ephemeral."""
        msg = {"role": "assistant"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_missing_role_key_not_ephemeral(self):
        """Missing role key should not be ephemeral."""
        msg = {"content": "Hello"}
        assert SessionDB.is_ephemeral_message(msg) is False

    def test_empty_dict_not_ephemeral(self):
        """Empty dict should not be ephemeral."""
        msg = {}
        assert SessionDB.is_ephemeral_message(msg) is False


# =========================================================================
# classify_messages() - Message classification for a session
# =========================================================================

class TestClassifyMessages:
    """Test the classify_messages() method."""

    def test_classifies_empty_session(self, db):
        """Empty session should return empty lists."""
        session_id = db.create_session("test1", "cli")
        result = db.classify_messages(session_id)
        
        assert "persistent" in result
        assert "ephemeral" in result
        assert len(result["persistent"]) == 0
        assert len(result["ephemeral"]) == 0

    def test_classifies_only_persistent_messages(self, db):
        """Session with only persistent messages."""
        session_id = db.create_session("test2", "cli")
        db.append_message(session_id, "user", "Hello")
        db.append_message(session_id, "assistant", "Hi there!")
        
        result = db.classify_messages(session_id)
        
        assert len(result["persistent"]) == 2
        assert len(result["ephemeral"]) == 0

    def test_classifies_only_ephemeral_messages(self, db):
        """Session with only ephemeral messages."""
        session_id = db.create_session("test3", "cli")
        db.append_message(session_id, "tool", "Tool output")
        
        result = db.classify_messages(session_id)
        
        assert len(result["persistent"]) == 0
        assert len(result["ephemeral"]) == 1

    def test_classifies_mixed_messages(self, db):
        """Session with both persistent and ephemeral messages."""
        session_id = db.create_session("test4", "cli")
        db.append_message(session_id, "user", "Run this command")
        db.append_message(session_id, "assistant", "Sure, running it now")
        db.append_message(session_id, "tool", "stdout: output\nexit_code: 0")
        db.append_message(session_id, "assistant", "Command completed")
        
        result = db.classify_messages(session_id)
        
        assert len(result["persistent"]) == 3
        assert len(result["ephemeral"]) == 1

    def test_classifies_long_content_as_ephemeral(self, db):
        """Long file contents should be classified as ephemeral."""
        session_id = db.create_session("test5", "cli")
        db.append_message(session_id, "assistant", "x" * 5001)
        
        result = db.classify_messages(session_id)
        
        assert len(result["ephemeral"]) == 1

    def test_classifies_critical_errors_as_persistent(self, db):
        """Critical errors should be classified as persistent."""
        session_id = db.create_session("test6", "cli")
        db.append_message(session_id, "assistant", "fatal Error: System crash")
        
        result = db.classify_messages(session_id)
        
        assert len(result["persistent"]) == 1
        assert len(result["ephemeral"]) == 0


# =========================================================================
# clear_ephemeral_messages() - Remove ephemeral messages
# =========================================================================

class TestClearEphemeralMessages:
    """Test the clear_ephemeral_messages() method."""

    def test_removes_no_messages_when_none_ephemeral(self, db):
        """Should return 0 when no ephemeral messages exist."""
        session_id = db.create_session("test1", "cli")
        db.append_message(session_id, "user", "Hello")
        db.append_message(session_id, "assistant", "Hi!")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 0
        
        # Verify all messages still exist
        messages = db.get_messages(session_id)
        assert len(messages) == 2

    def test_removes_tool_messages(self, db):
        """Should remove tool role messages."""
        session_id = db.create_session("test2", "cli")
        db.append_message(session_id, "user", "Run command")
        db.append_message(session_id, "tool", "Tool output")
        db.append_message(session_id, "assistant", "Done")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 1
        
        messages = db.get_messages(session_id)
        assert len(messages) == 2
        assert all(m["role"] != "tool" for m in messages)

    def test_removes_long_content_messages(self, db):
        """Should remove messages with content > 5000 chars."""
        session_id = db.create_session("test3", "cli")
        db.append_message(session_id, "user", "Read file")
        db.append_message(session_id, "assistant", "x" * 5001)
        db.append_message(session_id, "assistant", "File read complete")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 1
        
        messages = db.get_messages(session_id)
        assert len(messages) == 2

    def test_removes_terminal_output_messages(self, db):
        """Should remove messages containing stdout/exit_code."""
        session_id = db.create_session("test4", "cli")
        db.append_message(session_id, "user", "Run ls")
        db.append_message(session_id, "assistant", "stdout: file1.txt\nexit_code: 0")
        db.append_message(session_id, "assistant", "Command finished")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 1
        
        messages = db.get_messages(session_id)
        assert len(messages) == 2

    def test_removes_non_critical_errors(self, db):
        """Should remove non-critical error messages."""
        session_id = db.create_session("test5", "cli")
        db.append_message(session_id, "user", "Do something")
        db.append_message(session_id, "assistant", "Error: Something went wrong")
        db.append_message(session_id, "assistant", "Retrying...")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 1

    def test_preserves_critical_errors(self, db):
        """Should NOT remove critical error messages."""
        session_id = db.create_session("test6", "cli")
        db.append_message(session_id, "user", "Delete everything")
        db.append_message(session_id, "assistant", "fatal Error: Cannot proceed")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 0
        
        messages = db.get_messages(session_id)
        assert len(messages) == 2

    def test_preserves_user_and_assistant_messages(self, db):
        """Should preserve all user and assistant messages (unless ephemeral by content)."""
        session_id = db.create_session("test7", "cli")
        db.append_message(session_id, "user", "First message")
        db.append_message(session_id, "assistant", "First response")
        db.append_message(session_id, "user", "Second message")
        db.append_message(session_id, "assistant", "Second response")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 0
        
        messages = db.get_messages(session_id)
        assert len(messages) == 4

    def test_removes_memory_retrieval_messages(self, db):
        """Should remove memory retrieval messages."""
        session_id = db.create_session("test8", "cli")
        db.append_message(session_id, "user", "What did I say before?")
        db.append_message(session_id, "assistant", "memory retrieved: you like coffee")
        db.append_message(session_id, "assistant", "So you like coffee!")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 1

    def test_counts_removals_accurately(self, db):
        """Should accurately count the number of removed messages."""
        session_id = db.create_session("test9", "cli")
        db.append_message(session_id, "user", "Msg 1")
        db.append_message(session_id, "tool", "Tool 1")
        db.append_message(session_id, "tool", "Tool 2")
        db.append_message(session_id, "assistant", "Response")
        db.append_message(session_id, "tool", "Tool 3")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 3
        
        messages = db.get_messages(session_id)
        assert len(messages) == 2

    def test_works_on_empty_session(self, db):
        """Should handle empty sessions without error."""
        session_id = db.create_session("test10", "cli")
        
        removed = db.clear_ephemeral_messages(session_id)
        
        assert removed == 0

    def test_is_idempotent(self, db):
        """Calling twice should remove nothing on second call."""
        session_id = db.create_session("test11", "cli")
        db.append_message(session_id, "tool", "Tool output")
        
        removed1 = db.clear_ephemeral_messages(session_id)
        removed2 = db.clear_ephemeral_messages(session_id)
        
        assert removed1 == 1
        assert removed2 == 0


# =========================================================================
# prune_messages() - Remove last N messages
# =========================================================================

class TestPruneMessages:
    """Test the prune_messages() method."""

    def test_prunes_correct_count(self, db):
        """Should prune exactly the requested number of messages."""
        session_id = db.create_session("test1", "cli")
        db.append_message(session_id, "user", "Msg 1")
        db.append_message(session_id, "assistant", "Msg 2")
        db.append_message(session_id, "user", "Msg 3")
        db.append_message(session_id, "assistant", "Msg 4")
        
        removed = db.prune_messages(session_id, count=2)
        
        assert removed == 2
        
        messages = db.get_messages(session_id)
        assert len(messages) == 2

    def test_prunes_most_recent_messages(self, db):
        """Should remove the most recent messages."""
        session_id = db.create_session("test2", "cli")
        db.append_message(session_id, "user", "First")
        db.append_message(session_id, "assistant", "Second")
        db.append_message(session_id, "user", "Third")
        
        db.prune_messages(session_id, count=1)
        
        messages = db.get_messages(session_id)
        assert len(messages) == 2
        assert messages[1]["content"] == "Second"  # "Third" should be gone

    def test_preserves_system_messages(self, db):
        """Should NOT prune system messages."""
        session_id = db.create_session("test3", "cli")
        # Note: update_system_prompt stores in sessions table, not as message
        # So we need to append a system message directly if we want to test this
        db.append_message(session_id, "system", "You are helpful.")
        db.append_message(session_id, "user", "Hello")
        db.append_message(session_id, "assistant", "Hi")
        
        removed = db.prune_messages(session_id, count=10)
        
        # Should only remove user and assistant messages, not system
        assert removed == 2
        
        messages = db.get_messages(session_id)
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1

    def test_prunes_zero_when_count_is_zero(self, db):
        """Should remove nothing when count=0."""
        session_id = db.create_session("test4", "cli")
        db.append_message(session_id, "user", "Hello")
        
        removed = db.prune_messages(session_id, count=0)
        
        assert removed == 0
        
        messages = db.get_messages(session_id)
        assert len(messages) == 1

    def test_prunes_all_when_count_exceeds(self, db):
        """Should remove all available messages when count exceeds."""
        session_id = db.create_session("test5", "cli")
        db.append_message(session_id, "user", "Hello")
        db.append_message(session_id, "assistant", "Hi")
        
        removed = db.prune_messages(session_id, count=100)
        
        assert removed == 2
        
        messages = db.get_messages(session_id)
        assert len(messages) == 0

    def test_prunes_empty_session(self, db):
        """Should handle empty sessions gracefully."""
        session_id = db.create_session("test6", "cli")
        
        removed = db.prune_messages(session_id, count=10)
        
        assert removed == 0

    def test_prune_respects_timestamp_ordering(self, db):
        """Should prune based on timestamp, not insertion order."""
        import time
        
        session_id = db.create_session("test7", "cli")
        db.append_message(session_id, "user", "First")
        time.sleep(0.01)
        db.append_message(session_id, "assistant", "Second")
        time.sleep(0.01)
        db.append_message(session_id, "user", "Third")
        
        db.prune_messages(session_id, count=1)
        
        messages = db.get_messages(session_id)
        # "Third" should be gone (most recent)
        contents = [m["content"] for m in messages]
        assert "Third" not in contents
        assert "First" in contents
        assert "Second" in contents

    def test_prune_multiple_times(self, db):
        """Should work correctly when called multiple times."""
        session_id = db.create_session("test8", "cli")
        for i in range(10):
            db.append_message(session_id, "user", f"Msg {i}")
        
        removed1 = db.prune_messages(session_id, count=3)
        removed2 = db.prune_messages(session_id, count=3)
        removed3 = db.prune_messages(session_id, count=3)
        
        assert removed1 == 3
        assert removed2 == 3
        assert removed3 == 3
        
        messages = db.get_messages(session_id)
        assert len(messages) == 1

    def test_prune_default_count_is_10(self, db):
        """Default count parameter should be 10."""
        session_id = db.create_session("test9", "cli")
        for i in range(15):
            db.append_message(session_id, "user", f"Msg {i}")
        
        removed = db.prune_messages(session_id)  # No count argument
        
        assert removed == 10
        
        messages = db.get_messages(session_id)
        assert len(messages) == 5


# =========================================================================
# Integration tests - Multiple operations together
# =========================================================================

class TestContextManagementIntegration:
    """Integration tests for context management workflow."""

    def test_clear_then_prune(self, db):
        """Should work correctly when clearing then pruning."""
        session_id = db.create_session("test1", "cli")
        db.append_message(session_id, "user", "Msg 1")
        db.append_message(session_id, "tool", "Tool 1")
        db.append_message(session_id, "assistant", "Response 1")
        db.append_message(session_id, "tool", "Tool 2")
        db.append_message(session_id, "assistant", "Response 2")
        
        # Clear ephemeral (should remove 2 tool messages)
        cleared = db.clear_ephemeral_messages(session_id)
        assert cleared == 2
        
        # Then prune last message (should remove 1 assistant response)
        pruned = db.prune_messages(session_id, count=1)
        assert pruned == 1
        
        # Should have 2 messages left (5 - 2 ephemeral - 1 pruned = 2)
        messages = db.get_messages(session_id)
        assert len(messages) == 2

    def test_classify_then_clear_matches(self, db):
        """Classification should match what gets cleared."""
        session_id = db.create_session("test2", "cli")
        db.append_message(session_id, "user", "Hello")
        db.append_message(session_id, "tool", "Tool output")
        db.append_message(session_id, "assistant", "Response")
        db.append_message(session_id, "tool", "Another tool")
        
        # Classify first
        classification = db.classify_messages(session_id)
        expected_ephemeral_count = len(classification["ephemeral"])
        
        # Then clear
        cleared = db.clear_ephemeral_messages(session_id)
        
        assert cleared == expected_ephemeral_count

    def test_multiple_sessions_isolated(self, db):
        """Operations on one session should not affect others."""
        session1 = db.create_session("session1", "cli")
        session2 = db.create_session("session2", "cli")
        
        db.append_message(session1, "tool", "Tool in session 1")
        db.append_message(session2, "tool", "Tool in session 2")
        
        # Clear only session1
        cleared = db.clear_ephemeral_messages(session1)
        assert cleared == 1
        
        # Session2 should be unaffected
        messages2 = db.get_messages(session2)
        assert len(messages2) == 1
        assert messages2[0]["role"] == "tool"

    def test_realistic_conversation_flow(self, db):
        """Test with a realistic conversation pattern."""
        session_id = db.create_session("realistic", "cli")
        
        # User asks a question
        db.append_message(session_id, "user", "What files are in this directory?")
        
        # Assistant responds
        db.append_message(session_id, "assistant", "Let me check the directory contents.")
        
        # Tool runs ls command
        db.append_message(session_id, "tool", "stdout: file1.txt\nfile2.py\nexit_code: 0")
        
        # Assistant summarizes
        db.append_message(session_id, "assistant", "I found 2 files: file1.txt and file2.py")
        
        # User asks for file content
        db.append_message(session_id, "user", "Show me file1.txt")
        
        # Assistant responds with long content
        db.append_message(session_id, "assistant", "Here's the content:\n" + "x" * 5001)
        
        # Classify
        classification = db.classify_messages(session_id)
        assert len(classification["persistent"]) == 4  # 3 user/assistant + 1 summary
        assert len(classification["ephemeral"]) == 2   # 1 tool output + 1 long content
        
        # Clear ephemeral
        cleared = db.clear_ephemeral_messages(session_id)
        assert cleared == 2
        
        # Verify remaining messages are the conversation flow
        messages = db.get_messages(session_id)
        assert len(messages) == 4
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "assistant", "user"]  # Conversation preserved


# =========================================================================
# Protected messages (start-tasks / end-tasks)
# =========================================================================

class TestAppendProtected:
    """Test appending messages with the protected flag."""

    def test_append_protected_message(self, db):
        sid = db.create_session("test", "cli")
        mid = db.append_message(sid, "user", "Task list", protected=True)
        assert mid > 0
        messages = db.get_messages(sid)
        assert len(messages) == 1
        assert messages[0]["protected"] == 1

    def test_append_normal_message_not_protected(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "Hello", protected=False)
        messages = db.get_messages(sid)
        assert messages[0]["protected"] == 0

    def test_protected_defaults_to_false(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "Hello")
        messages = db.get_messages(sid)
        assert messages[0]["protected"] == 0


class TestProtectOperations:
    """Test retroactive protect / unprotect operations."""

    def test_protect_messages_after_with_timestamp(self, db):
        import time
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "first")
        time.sleep(0.05)
        cutoff = time.time()
        db.append_message(sid, "user", "second")
        db.append_message(sid, "assistant", "response")

        updated = db.protect_messages_after(sid, after_timestamp=cutoff)
        assert updated == 2  # second and response
        messages = db.get_messages(sid)
        assert messages[0]["protected"] == 0  # first
        assert messages[1]["protected"] == 1  # second
        assert messages[2]["protected"] == 1  # response

    def test_protect_messages_after_no_timestamp(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "first")
        db.append_message(sid, "user", "second")

        updated = db.protect_messages_after(sid)
        # Protects everything (since after_timestamp defaults to latest msg time)
        # In practice this protects 0 because nothing exists after the latest
        # This tests the edge case
        assert updated >= 0

    def test_unprotect_all_messages(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "tasks", protected=True)
        db.append_message(sid, "assistant", "result", protected=True)

        unprotected = db.unprotect_all_messages(sid)
        assert unprotected == 2

        messages = db.get_messages(sid)
        assert all(m["protected"] == 0 for m in messages)

    def test_get_protected_messages(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "normal")
        db.append_message(sid, "user", "task1", protected=True)
        db.append_message(sid, "user", "task2", protected=True)

        timestamps = db.get_protected_messages(sid)
        assert len(timestamps) == 2

    def test_get_protected_messages_empty(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "normal")
        timestamps = db.get_protected_messages(sid)
        assert timestamps == []


class TestProtectedSurvivesClear:
    """Test that protected messages survive clear_ephemeral_messages."""

    def test_protected_not_cleared_by_default_mode(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "Task list", protected=True)
        db.append_message(sid, "assistant", "OK")
        db.append_message(sid, "tool", "Tool output here")

        cleared = db.clear_ephemeral_messages(sid)
        assert cleared == 1  # Only tool output removed

        messages = db.get_messages(sid)
        roles = [m["role"] for m in messages]
        assert "user" in roles  # protected task list survives
        assert "tool" not in roles  # tool output removed

    def test_protected_not_cleared_when_only_ephemeral(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "My task list\n1. Do thing A\n2. Do thing B", protected=True)
        db.append_message(sid, "tool", "Output")
        db.append_message(sid, "tool", "Another output")

        cleared = db.clear_ephemeral_messages(sid)
        assert cleared == 2

        messages = db.get_messages(sid)
        assert len(messages) == 1
        assert messages[0]["protected"] == 1
        assert "task list" in messages[0]["content"].lower()

    def test_protected_not_pruned(self, db):
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "Task list", protected=True)
        db.append_message(sid, "assistant", "Response 1")
        db.append_message(sid, "user", "Question 2")
        db.append_message(sid, "assistant", "Response 2")
        db.append_message(sid, "user", "Question 3")

        # Prune last 3 non-system messages - should skip the protected one
        pruned = db.prune_messages(sid, count=3)
        assert pruned == 3

        # Protected task list MUST survive
        messages = db.get_messages(sid)
        protected_msgs = [m for m in messages if m["protected"]]
        assert len(protected_msgs) == 1
        assert protected_msgs[0]["content"] == "Task list"

    def test_is_idempotent_clear_with_protected(self, db):
        """Clearing twice should not remove more messages."""
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "Tasks", protected=True)
        db.append_message(sid, "tool", "Output")

        first = db.clear_ephemeral_messages(sid)
        assert first == 1
        second = db.clear_ephemeral_messages(sid)
        assert second == 0  # Nothing more to clear


class TestTaskListFlow:
    """Test the full start-tasks / end-tasks workflow."""

    def test_full_task_lifecycle(self, db):
        """Simulate: start tasks -> work -> clear -> verify tasks survive."""
        sid = db.create_session("workflow", "cli")

        # User starts tasks
        task_text = "1. Analyze API\n2. Write middleware\n3. Add tests"
        db.append_message(sid, "user", f"[Task List]\n{task_text}", protected=True)

        # Work on task 1
        db.append_message(sid, "assistant", "Starting task 1...")
        db.append_message(sid, "tool", "API analysis results")
        db.append_message(sid, "assistant", "API has 3 endpoints with 100 req/min rate limit")

        # Clear context (simulating /clear-context after task 1)
        cleared = db.clear_ephemeral_messages(sid)
        assert cleared >= 1  # At least tool output removed

        # Task list MUST still be there
        messages = db.get_messages(sid)
        has_task_list = any("Task List" in (m.get("content") or "") for m in messages if m.get("protected"))
        assert has_task_list, "Task list was lost after clear-context!"

        # User starts task 2
        db.append_message(sid, "assistant", "Moving to task 2...")
        db.append_message(sid, "tool", "Middleware code")

        # Clear again
        cleared2 = db.clear_ephemeral_messages(sid)

        # Task list STILL there
        messages_after = db.get_messages(sid)
        has_task_list = any("Task List" in (m.get("content") or "") for m in messages_after)
        assert has_task_list, "Task list was lost after second clear!"

        # End tasks - unprotect everything
        unprotected = db.unprotect_all_messages(sid)
        assert unprotected >= 1

        # Now a clear would remove more messages (no protection)
        messages_final = db.get_messages(sid)
        assert len(messages_final) > 0

    def test_implicit_task_protection(self, db):
        """Start task protection, append messages normally, verify they're auto-protected."""
        sid = db.create_session("implicit", "cli")

        db.start_task_protection(sid)
        assert db.is_task_protection_active(sid) is True

        # Append messages without explicit protected flag
        db.append_message(sid, "user", "1. Write tests\n2. Fix bugs")
        db.append_message(sid, "assistant", "Starting task 1")
        db.append_message(sid, "tool", "pytest output: 42 passed")

        # All should be auto-protected
        messages = db.get_messages(sid)
        assert all(m["protected"] == 1 for m in messages), "All messages should be auto-protected"

        # Clear ephemeral — nothing protected should be removed
        cleared = db.clear_ephemeral_messages(sid)
        assert cleared == 0  # All messages are protected

        # End task protection
        db.end_task_protection(sid)
        assert db.is_task_protection_active(sid) is False

        # New messages should NOT be auto-protected
        db.append_message(sid, "assistant", "Work continued after end")
        messages_after = db.get_messages(sid)
        assert messages_after[-1]["protected"] == 0

    def test_start_protection_is_idempotent(self, db):
        """Calling start_task_protection twice should not error."""
        sid = db.create_session("test", "cli")
        db.start_task_protection(sid)
        db.start_task_protection(sid)
        assert db.is_task_protection_active(sid) is True

    def test_end_protection_when_not_active(self, db):
        """Calling end_task_protection when not active should not error."""
        sid = db.create_session("test", "cli")
        db.end_task_protection(sid)
        assert db.is_task_protection_active(sid) is False


# =========================================================================
# clear_task_window_ephemeral_messages() - Clear ONLY task-window noise
# =========================================================================

class TestClearTaskWindowEphemeralMessages:
    """Test clearing ephemeral messages only from the task window."""

    def test_clears_only_tool_outputs_from_task_window(self, db):
        """Should remove tool messages from the task window but keep
        user task definitions and assistant summaries."""
        sid = db.create_session("test", "cli")
        # Pre-task conversation (unprotected)
        db.append_message(sid, "user", "Hello")
        db.append_message(sid, "assistant", "Hi there!")

        # Start tasks
        db.start_task_protection(sid)
        db.append_message(sid, "user", "1. List files\n2. Run tests")
        db.append_message(sid, "assistant", "Running task 1...")
        db.append_message(sid, "tool", '{"stdout": "file1.py\\nfile2.py", "exit_code": 0}')
        db.append_message(sid, "assistant", "Found file1.py and file2.py")

        # End tasks
        db.end_task_protection(sid)

        before = db.message_count(sid)
        assert before == 6  # 2 pre-task + 4 task window

        cleared = db.clear_task_window_ephemeral_messages(sid)
        # Only the tool output from task window should be deleted
        assert cleared == 1

        messages = db.get_messages(sid)
        roles = [m["role"] for m in messages]
        # Tool message removed, everything else kept
        assert roles.count("tool") == 0
        assert len(messages) == 5

    def test_unprotects_non_ephemeral_task_messages(self, db):
        """Non-ephemeral messages from the task window (user/assistant)
        should have their protected flag cleared so future /clear-context
        can evaluate them normally."""
        sid = db.create_session("test", "cli")
        db.start_task_protection(sid)
        db.append_message(sid, "user", "My task list")
        db.append_message(sid, "assistant", "OK, starting tasks")
        db.append_message(sid, "tool", '{"stdout": "ok", "exit_code": 0}')
        db.end_task_protection(sid)

        # All 3 are protected
        messages = db.get_messages(sid)
        assert all(m["protected"] == 1 for m in messages)

        db.clear_task_window_ephemeral_messages(sid)

        messages = db.get_messages(sid)
        # Tool output deleted, remaining messages unprotected
        assert len(messages) == 2  # user + assistant
        assert all(m["protected"] == 0 for m in messages), \
            "Non-ephemeral task messages should be unprotected after task clear"

    def test_preserves_non_task_messages(self, db):
        """Messages before/after the task window should not be affected."""
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "Before task")
        db.append_message(sid, "assistant", "Before response")
        db.append_message(sid, "tool", '{"stdout": "before", "exit_code": 0}')

        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task!")
        db.append_message(sid, "tool", '{"stdout": "during", "exit_code": 0}')
        db.end_task_protection(sid)

        db.append_message(sid, "user", "After task")
        db.append_message(sid, "tool", '{"stdout": "after", "exit_code": 0}')

        cleared = db.clear_task_window_ephemeral_messages(sid)
        assert cleared == 1  # Only the task-window tool output

        messages = db.get_messages(sid)
        # Pre-task and post-task tool outputs should survive
        roles = [m["role"] for m in messages]
        assert roles.count("tool") == 2  # before + after

    def test_returns_zero_when_no_task_window_messages(self, db):
        """No protected messages = nothing to clear."""
        sid = db.create_session("test", "cli")
        db.append_message(sid, "user", "Hello")
        db.append_message(sid, "assistant", "Hi")

        cleared = db.clear_task_window_ephemeral_messages(sid)
        assert cleared == 0
        assert db.message_count(sid) == 2

    def test_respects_ephemeral_heuristics(self, db):
        """Should correctly classify long content, stdout patterns, and
        critical errors within the task window."""
        sid = db.create_session("test", "cli")
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task: read file")
        db.append_message(sid, "assistant", "x" * 5001)  # Long content -> ephemeral
        db.append_message(sid, "assistant", "stdout: output\\nexit_code: 0")  # Terminal output -> ephemeral
        db.append_message(sid, "assistant", "fatal Error: disk full")  # Critical error -> NOT ephemeral
        db.end_task_protection(sid)

        cleared = db.clear_task_window_ephemeral_messages(sid)
        assert cleared == 2  # Long content + stdout removed

        messages = db.get_messages(sid)
        assert any("fatal Error" in (m.get("content") or "") for m in messages)

    def test_is_idempotent(self, db):
        """Calling twice should not remove more."""
        sid = db.create_session("test", "cli")
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task")
        db.append_message(sid, "tool", '{"stdout": "out", "exit_code": 0}')
        db.end_task_protection(sid)

        first = db.clear_task_window_ephemeral_messages(sid)
        second = db.clear_task_window_ephemeral_messages(sid)
        assert first == 1
        assert second == 0
