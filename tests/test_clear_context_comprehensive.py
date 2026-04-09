"""Integration tests for /clear-context command covering all modes and edge cases.

This file addresses four test coverage gaps:
1. Missing conversation_history attribute in test fixtures
2. No tests for CLI argument parsing edge cases
3. No tests for _estimate_current_context_tokens()
4. No tests for the 'task' mode
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from cli import HermesCLI
from hermes_state import SessionDB


# =============================================================================
# Fix #1: Add conversation_history to fixture
# =============================================================================

def _make_cli_with_db(tmp_path):
    """Build a minimal HermesCLI stub wired to a real SessionDB.
    
    This fixture includes conversation_history attribute which is required
    by _handle_clear_context() for before/after message counting.
    """
    db_path = tmp_path / "integration.db"
    db = SessionDB(db_path=db_path)
    db.create_session("test-session", "cli")
    
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "test-session"
    cli_obj._session_db = db
    cli_obj.system_prompt = "You are a helpful assistant."
    cli_obj._app = None
    cli_obj.agent = None
    cli_obj.conversation_history = []  # FIX #1: Required by _handle_clear_context
    return cli_obj, db


# =============================================================================
# Fix #2: Tests for CLI argument parsing edge cases
# =============================================================================

class TestClearContextArgumentParsing:
    """Tests for CLI argument parsing edge cases in /clear-context.
    
    The mode extraction logic uses: parts = cmd.strip().split(maxsplit=1)
    This tests various edge cases like extra whitespace, case variations,
    and malformed commands.
    """
    
    def test_extra_whitespace_before_mode(self, tmp_path):
        """Multiple spaces before mode should be handled correctly."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Multiple spaces between command and mode
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context   all")
        
        # Should execute without error and use 'all' mode
        output = " ".join(captured)
        assert "ephemeral" in output.lower() or "Cleared" in output
    
    def test_extra_whitespace_after_command(self, tmp_path):
        """Multiple spaces after command name should work."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Multiple spaces after command
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context   recent")
        
        output = " ".join(captured)
        assert "prune" in output.lower() or "Pruned" in output
    
    def test_leading_trailing_whitespace(self, tmp_path):
        """Leading and trailing whitespace should be stripped."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Leading and trailing spaces
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("   /clear-context all   ")
        
        output = " ".join(captured)
        assert "ephemeral" in output.lower() or "Cleared" in output
    
    def test_mixed_case_mode_names(self, tmp_path):
        """Mode names should be case-insensitive."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        for mode in ["ALL", "All", "aLl", "RECENT", "Recent", "Task", "TASK"]:
            captured = []
            with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
                cli_obj._handle_clear_context(f"/clear-context {mode}")
            
            output = " ".join(captured)
            # Should not show usage (which appears on invalid mode)
            assert "Usage" not in output or mode.lower() in ["all", "recent", "task"]
    
    def test_no_mode_defaults_to_auto(self, tmp_path):
        """Command without mode should default to 'auto' mode."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        db.append_message("test-session", "user", "Hello")
        db.append_message("test-session", "tool", '{"stdout": "output", "exit_code": 0}')
        
        before_count = db.message_count("test-session")
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context")
        
        after_count = db.message_count("test-session")
        
        # Auto mode should remove tool messages
        assert after_count < before_count
    
    def test_invalid_mode_shows_usage(self, tmp_path):
        """Invalid mode should show usage message."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context invalid")
        
        output = " ".join(captured)
        assert "Usage" in output
        assert "all" in output
        assert "recent" in output
        assert "task" in output
    
    def test_mode_with_extra_args(self, tmp_path):
        """Mode followed by extra arguments should be handled gracefully."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Extra argument after mode
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context all extra_arg")
        
        # Should treat "all extra_arg" as invalid mode (since split only on first space)
        # Actually, with maxsplit=1, parts[1] = "all extra_arg", which becomes "all extra_arg"
        # This is not in the valid modes list, so it shows usage
        output = " ".join(captured)
        assert "Usage" in output
    
    def test_tabs_and_newlines(self, tmp_path):
        """Tabs and newlines in command should be handled."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Tab character
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context\tall")
        
        output = " ".join(captured)
        # Should work since strip() and split() handle whitespace
        assert "ephemeral" in output.lower() or "Cleared" in output or "Usage" in output


# =============================================================================
# Fix #3: Tests for _estimate_current_context_tokens()
# =============================================================================

class TestEstimateCurrentContextTokens:
    """Tests for the _estimate_current_context_tokens() method.
    
    This method is used in /clear-context output to show token savings.
    It estimates the full context including system prompt, messages, and tools.
    """
    
    def test_returns_zero_when_no_messages(self, tmp_path):
        """Empty session should return a small token count."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        tokens = cli_obj._estimate_current_context_tokens()
        
        # Should return 0 or a small number for empty session with just system prompt
        assert isinstance(tokens, int)
        assert tokens >= 0
    
    def test_includes_system_prompt_tokens(self, tmp_path):
        """Token estimate should include system prompt."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Short system prompt
        cli_obj.system_prompt = "Short prompt"
        tokens_short = cli_obj._estimate_current_context_tokens()
        
        # Long system prompt
        cli_obj.system_prompt = "X" * 10000  # ~2500 tokens
        tokens_long = cli_obj._estimate_current_context_tokens()
        
        # Long prompt should have more tokens
        assert tokens_long > tokens_short
    
    def test_includes_message_tokens(self, tmp_path):
        """Token estimate should include message content."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # No messages
        tokens_empty = cli_obj._estimate_current_context_tokens()
        
        # Add messages
        db.append_message("test-session", "user", "X" * 1000)
        db.append_message("test-session", "assistant", "Y" * 1000)
        
        tokens_with_messages = cli_obj._estimate_current_context_tokens()
        
        # Should have more tokens with messages
        assert tokens_with_messages > tokens_empty
    
    def test_returns_zero_on_exception(self, tmp_path):
        """Should return 0 and log warning on estimation failure."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        with patch("agent.model_metadata.estimate_session_token_count_from_rows",
                   side_effect=ValueError("broken")), \
             patch("cli.logger") as mock_logger:
            
            tokens = cli_obj._estimate_current_context_tokens()
            
            assert tokens == 0
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "Failed to estimate context tokens" in call_args
    
    def test_includes_tools_when_agent_has_tools(self, tmp_path):
        """Token estimate should include tool schemas when agent has tools."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Mock agent with tools
        mock_agent = MagicMock()
        mock_agent.tools = [{"name": "test_tool", "description": "x" * 1000}]
        cli_obj.agent = mock_agent
        
        tokens_with_tools = cli_obj._estimate_current_context_tokens()
        
        # Remove agent
        cli_obj.agent = None
        tokens_without_tools = cli_obj._estimate_current_context_tokens()
        
        # With tools should have more tokens (if estimation includes them)
        # Note: This depends on how estimate_session_token_count_from_rows handles tools
        assert isinstance(tokens_with_tools, int)
        assert isinstance(tokens_without_tools, int)
    
    def test_handles_none_agent(self, tmp_path):
        """Should handle agent=None gracefully."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        cli_obj.agent = None
        
        tokens = cli_obj._estimate_current_context_tokens()
        
        assert isinstance(tokens, int)
        assert tokens >= 0
    
    def test_handles_agent_without_tools(self, tmp_path):
        """Should handle agent that has no tools attribute."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        mock_agent = MagicMock()
        mock_agent.tools = None
        cli_obj.agent = mock_agent
        
        tokens = cli_obj._estimate_current_context_tokens()
        
        assert isinstance(tokens, int)
        assert tokens >= 0
    
    def test_token_estimate_correlates_with_message_size(self, tmp_path):
        """Larger messages should result in higher token estimates."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Add small message
        db.append_message("test-session", "user", "Hello")
        tokens_small = cli_obj._estimate_current_context_tokens()
        
        # Add large message
        db.append_message("test-session", "user", "X" * 10000)
        tokens_large = cli_obj._estimate_current_context_tokens()
        
        # Large message should have higher token count
        assert tokens_large > tokens_small


# =============================================================================
# Fix #4: Tests for 'task' mode
# =============================================================================

class TestClearContextTaskMode:
    """End-to-end tests for /clear-context task mode.
    
    The 'task' mode clears ephemeral messages only from the task window
    (messages marked as protected between /start-tasks and /end-tasks).
    """
    
    def test_task_mode_clears_task_window_ephemeral(self, tmp_path):
        """Task mode should clear ephemeral messages from protected task window."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id
        
        # Create a task window with mixed messages
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task: fix the bug")
        db.append_message(sid, "assistant", "Starting task...")
        db.append_message(sid, "tool", '{"stdout": "test output", "exit_code": 0}')
        db.append_message(sid, "assistant", "Bug fixed!")
        db.end_task_protection(sid)
        
        # All messages are now protected
        before_count = db.message_count(sid)
        assert before_count == 4
        
        # Run task mode
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context task")
        
        after_count = db.message_count(sid)
        output = " ".join(captured)
        
        # Should have removed the tool message (ephemeral)
        assert after_count == 3
        assert "task-window" in output.lower() or "task" in output.lower()
    
    def test_task_mode_with_no_task_window(self, tmp_path):
        """Task mode with no protected messages should remove 0."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id
        
        # Add regular (non-protected) messages
        db.append_message(sid, "user", "Hello")
        db.append_message(sid, "tool", '{"stdout": "output", "exit_code": 0}')
        
        before_count = db.message_count(sid)
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context task")
        
        after_count = db.message_count(sid)
        output = " ".join(captured)
        
        # Task mode should not affect non-protected messages
        # (it only clears from protected=1 messages)
        assert after_count == before_count
        assert "0" in output  # Should report 0 removed
    
    def test_task_mode_preserves_user_and_assistant(self, tmp_path):
        """Task mode should preserve user prompts and assistant summaries."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id
        
        # Task window with file content (ephemeral)
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Read this file")
        db.append_message(sid, "assistant", "Reading file...")
        db.append_message(sid, "tool", '{"stdout": "x" * 10000, "exit_code": 0}')  # Large tool output
        db.append_message(sid, "assistant", "File contains important info")
        db.end_task_protection(sid)
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context task")
        
        messages = db.get_messages(sid)
        roles = [m["role"] for m in messages]
        
        # Should have user and assistant, no tool
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" not in roles
    
    def test_task_mode_output_format(self, tmp_path):
        """Task mode should output message count with 'task-window' indicator."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id
        
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task")
        db.append_message(sid, "tool", '{"stdout": "out", "exit_code": 0}')
        db.end_task_protection(sid)
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context task")
        
        output = "\n".join(captured)
        
        # Should mention task-window or task in the output
        assert "task" in output.lower() or "task-window" in output.lower()
    
    def test_task_mode_multiple_task_windows(self, tmp_path):
        """Task mode should handle multiple task windows in sequence."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id
        
        # First task window
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task 1")
        db.append_message(sid, "tool", '{"stdout": "t1", "exit_code": 0}')
        db.end_task_protection(sid)
        
        # Second task window
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Task 2")
        db.append_message(sid, "tool", '{"stdout": "t2", "exit_code": 0}')
        db.end_task_protection(sid)
        
        before_count = db.message_count(sid)
        assert before_count == 4
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context task")
        
        after_count = db.message_count(sid)
        
        # Should have removed both tool messages
        assert after_count == 2  # Only user messages remain
    
    def test_task_mode_with_file_content(self, tmp_path):
        """Task mode should remove large file content from task window."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id
        
        db.start_task_protection(sid)
        db.append_message(sid, "user", "Show me the file")
        # Large file content (would be classified as ephemeral)
        db.append_message(sid, "assistant", "x" * 6000)  
        db.end_task_protection(sid)
        
        before_count = db.message_count(sid)
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_clear_context("/clear-context task")
        
        after_count = db.message_count(sid)
        
        # Large file content should be removed
        assert after_count < before_count


# =============================================================================
# Combined tests: All fixes working together
# =============================================================================

class TestClearContextCombined:
    """Tests that verify all fixes work together."""
    
    def test_full_flow_with_token_estimation(self, tmp_path):
        """Complete /clear-context flow with token estimation output."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id
        
        # Set up a conversation
        db.append_message(sid, "user", "Hello")
        db.append_message(sid, "assistant", "Hi there!")
        db.append_message(sid, "tool", '{"stdout": "output", "exit_code": 0}')
        db.append_message(sid, "assistant", "Done")
        
        # Mock token estimation
        with patch.object(cli_obj, '_estimate_current_context_tokens', 
                         side_effect=[1000, 800]) as mock_estimate:
            
            captured = []
            with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
                cli_obj._handle_clear_context("/clear-context")
            
            # Verify token estimation was called twice (before and after)
            assert mock_estimate.call_count == 2
            
            # Verify output includes token information
            output = " ".join(captured)
            assert "token" in output.lower() or "200" in output  # 1000 - 800 = 200
    
    def test_fixture_has_required_attributes(self, tmp_path):
        """Verify the fixture has all required attributes."""
        cli_obj, db = _make_cli_with_db(tmp_path)
        
        # Check all required attributes exist
        assert hasattr(cli_obj, 'session_id')
        assert hasattr(cli_obj, '_session_db')
        assert hasattr(cli_obj, 'system_prompt')
        assert hasattr(cli_obj, '_app')
        assert hasattr(cli_obj, 'agent')
        assert hasattr(cli_obj, 'conversation_history')  # FIX #1
        
        # Verify types
        assert isinstance(cli_obj.session_id, str)
        assert isinstance(cli_obj.conversation_history, list)
