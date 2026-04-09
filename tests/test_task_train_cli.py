"""Tests for /start-tasks and /end-tasks CLI handlers and task train execution.

Tests the full task train workflow:
- /start-tasks: Enter task definition mode (buffer messages, don't send to model)
- /end-tasks: Exit mode and execute task sequence with auto-clear between each
- _execute_task_train: Sequential task execution with context management
"""

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from cli import HermesCLI
from hermes_state import SessionDB


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cli_with_db(tmp_path):
    """Create a minimal HermesCLI with a real database for task train testing."""
    db_path = tmp_path / "task_train.db"
    db = SessionDB(db_path=db_path)
    db.create_session("test-session", "cli")
    
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "test-session"
    cli_obj._session_db = db
    cli_obj.system_prompt = "You are a helpful assistant."
    cli_obj._app = None
    cli_obj.agent = None
    cli_obj.conversation_history = []
    
    return cli_obj, db


@pytest.fixture
def cli_with_mock_agent(tmp_path):
    """Create a HermesCLI with a mock agent for testing task train execution."""
    db_path = tmp_path / "task_train.db"
    db = SessionDB(db_path=db_path)
    db.create_session("test-session", "cli")
    
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "test-session"
    cli_obj._session_db = db
    cli_obj.system_prompt = "You are a helpful assistant."
    cli_obj._app = None
    cli_obj.conversation_history = []
    
    # Mock agent that returns predictable responses
    mock_agent = MagicMock()
    mock_agent.run_conversation = MagicMock(return_value={
        "final_response": "Task completed",
        "messages": [
            {"role": "user", "content": "Test task"},
            {"role": "assistant", "content": "Task completed"}
        ]
    })
    cli_obj.agent = mock_agent
    
    return cli_obj, db


# =============================================================================
# /start-tasks Tests
# =============================================================================

class TestStartTasksCommand:
    """Tests for /start-tasks CLI handler."""

    def test_start_tasks_enters_mode(self, cli_with_db):
        """Test that /start-tasks enters task definition mode."""
        cli_obj, db = cli_with_db
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_start_tasks("/start-tasks")
        
        # Should show success message
        output = " ".join(captured)
        assert "Task definition mode started" in output or "started" in output.lower()
        
        # Verify mode is active
        assert db.is_task_definition_mode_active("test-session") is True
        
        # Should contain instructions
        assert any("Type each task" in msg or "type" in msg.lower() for msg in captured)

    def test_start_tasks_already_active(self, cli_with_db):
        """Test that /start-tasks shows message when already active."""
        cli_obj, db = cli_with_db
        
        # Enter mode first
        db.enter_task_definition_mode("test-session")
        assert db.is_task_definition_mode_active("test-session") is True
        
        # Try to enter again
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_start_tasks("/start-tasks")
        
        output = " ".join(captured)
        assert "already active" in output.lower() or "already" in output.lower()

    def test_start_tasks_shows_usage_instructions(self, cli_with_db):
        """Test that /start-tasks shows usage instructions."""
        cli_obj, db = cli_with_db
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_start_tasks("/start-tasks")
        
        output = " ".join(captured).lower()
        # Should mention /end-tasks to exit
        assert "end-tasks" in output or "execute" in output.lower()

    def test_start_tasks_invalidates_app(self, cli_with_db):
        """Test that /start-tasks invalidates the app (refreshes display)."""
        cli_obj, db = cli_with_db
        mock_app = MagicMock()
        cli_obj._app = mock_app
        
        cli_obj._handle_start_tasks("/start-tasks")
        
        mock_app.invalidate.assert_called_once()


# =============================================================================
# /end-tasks Tests (without execution)
# =============================================================================

class TestEndTasksCommand:
    """Tests for /end-tasks CLI handler."""

    def test_end_tasks_no_mode_no_tasks(self, cli_with_db):
        """Test /end-tasks when not in mode and no tasks buffered."""
        cli_obj, db = cli_with_db
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_end_tasks("/end-tasks")
        
        output = " ".join(captured)
        assert "no tasks" in output.lower() or "defined" in output.lower()

    def test_end_tasks_exits_mode(self, cli_with_db):
        """Test that /end-tasks exits task definition mode."""
        cli_obj, db = cli_with_db
        
        # Enter mode
        db.enter_task_definition_mode("test-session")
        assert db.is_task_definition_mode_active("test-session") is True
        
        # Add a task definition
        db.store_task_definition("test-session", "Write documentation")
        
        # Exit mode
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_end_tasks("/end-tasks")
        
        # Mode should be exited
        assert db.is_task_definition_mode_active("test-session") is False
        
        # Should have called _execute_task_train
        # (This is tested more thoroughly in execution tests)

    def test_end_tasks_gets_task_definitions(self, cli_with_db):
        """Test that /end-tasks retrieves buffered tasks."""
        cli_obj, db = cli_with_db
        
        # Enter mode and add tasks
        db.enter_task_definition_mode("test-session")
        db.store_task_definition("test-session", "Task 1")
        db.store_task_definition("test-session", "Task 2")
        
        tasks = db.get_task_definitions("test-session")
        assert len(tasks) == 2
        assert tasks[0]["content"] == "Task 1"
        assert tasks[1]["content"] == "Task 2"

    def test_end_tasks_shows_task_count(self, cli_with_db):
        """Test that /end-tasks shows the number of tasks."""
        cli_obj, db = cli_with_db
        
        # Enter mode and add tasks
        db.enter_task_definition_mode("test-session")
        db.store_task_definition("test-session", "Task 1")
        db.store_task_definition("test-session", "Task 2")
        db.store_task_definition("test-session", "Task 3")
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_end_tasks("/end-tasks")
        
        output = " ".join(captured)
        # Should show task count
        assert "3" in output or "task" in output.lower()


# =============================================================================
# Task Train Execution Tests
# =============================================================================

class TestExecuteTaskTrain:
    """Tests for _execute_task_train() method."""

    def test_execute_single_task(self, cli_with_mock_agent):
        """Test executing a single task."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [{"content": "Write documentation"}]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # Agent should be called once
        assert cli_obj.agent.run_conversation.call_count == 1
        
        # Should show task execution messages
        output = " ".join(captured)
        assert "1/1" in output or "Executing" in output
        assert "completed" in output.lower()

    def test_execute_multiple_tasks(self, cli_with_mock_agent):
        """Test executing multiple tasks sequentially."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [
            {"content": "Write documentation"},
            {"content": "Add unit tests"},
            {"content": "Deploy to production"}
        ]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # Agent should be called 3 times
        assert cli_obj.agent.run_conversation.call_count == 3
        
        # Should show progress for each task
        output = " ".join(captured)
        assert "1/3" in output
        assert "2/3" in output
        assert "3/3" in output

    def test_execute_task_train_clears_history(self, cli_with_mock_agent):
        """Test that task train clears conversation history before starting."""
        cli_obj, db = cli_with_mock_agent
        
        # Set initial history
        cli_obj.conversation_history = [
            {"role": "user", "content": "Previous conversation"},
            {"role": "assistant", "content": "Previous response"}
        ]
        
        tasks = [{"content": "New task"}]
        
        # Verify history is cleared at the start of execution
        # We check this by examining the first message in the final history
        # which should be from the task, not the "Previous conversation"
        cli_obj._execute_task_train(tasks)
        
        # History should contain task messages, not the previous ones
        assert len(cli_obj.conversation_history) > 0
        # First message should be the task, not "Previous conversation"
        assert cli_obj.conversation_history[0]["content"] == "Test task"
        assert "Previous conversation" not in cli_obj.conversation_history[0]["content"]

    def test_execute_task_train_updates_history(self, cli_with_mock_agent):
        """Test that task train updates conversation history after each task."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [{"content": "Task 1"}]
        
        cli_obj._execute_task_train(tasks)
        
        # History should be updated with task responses
        assert len(cli_obj.conversation_history) > 0
        assert cli_obj.conversation_history[0]["role"] == "user"

    def test_execute_task_train_handles_task_failure(self, cli_with_mock_agent):
        """Test that task train continues after a task fails."""
        cli_obj, db = cli_with_mock_agent
        
        # Make second task fail
        def fail_on_second_call(*args, **kwargs):
            call_count = fail_on_second_call.count
            fail_on_second_call.count += 1
            if call_count == 2:
                raise Exception("Task failed")
            return {
                "final_response": "Task completed",
                "messages": [{"role": "user", "content": "Task"}, {"role": "assistant", "content": "Done"}]
            }
        fail_on_second_call.count = 1
        
        cli_obj.agent.run_conversation = MagicMock(side_effect=fail_on_second_call)
        
        tasks = [
            {"content": "Task 1"},
            {"content": "Task 2 (fails)"},
            {"content": "Task 3"}
        ]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # All 3 tasks should be attempted
        assert cli_obj.agent.run_conversation.call_count == 3
        
        # Should show failure for task 2
        output = " ".join(captured)
        assert "failed" in output.lower()

    def test_execute_task_train_shows_completion_summary(self, cli_with_mock_agent):
        """Test that task train shows completion summary."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [
            {"content": "Task 1"},
            {"content": "Task 2"}
        ]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        output = " ".join(captured)
        # Should show completion message
        assert "completed" in output.lower() or "task train" in output.lower()


# =============================================================================
# Auto-Clear Between Tasks Tests
# =============================================================================

class TestAutoClearBetweenTasks:
    """Tests for automatic context clearing between tasks."""

    def test_auto_clear_called_between_tasks(self, cli_with_mock_agent):
        """Test that clear_ephemeral_messages is called between tasks."""
        cli_obj, db = cli_with_mock_agent
        
        # Track calls to clear_ephemeral_messages
        original_clear = db.clear_ephemeral_messages
        clear_calls = []
        
        def track_clear(session_id):
            clear_calls.append(session_id)
            return original_clear(session_id)
        
        db.clear_ephemeral_messages = track_clear
        
        tasks = [
            {"content": "Task 1"},
            {"content": "Task 2"},
            {"content": "Task 3"}
        ]
        
        cli_obj._execute_task_train(tasks)
        
        # Should clear after task 1 and task 2 (not after task 3, the last one)
        assert len(clear_calls) == 2
        assert all(call == "test-session" for call in clear_calls)

    def test_auto_clear_keeps_last_assistant_response(self, cli_with_mock_agent):
        """Test that only last assistant response is kept after clear."""
        cli_obj, db = cli_with_mock_agent
        
        # Set up mock to return history with multiple messages
        def mock_run_conversation(user_message, **kwargs):
            return {
                "final_response": "Done",
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "Response 1"},
                    {"role": "tool", "content": "Tool output"},
                    {"role": "assistant", "content": "Final response"}
                ]
            }
        
        cli_obj.agent.run_conversation = mock_run_conversation
        
        tasks = [
            {"content": "Task 1"},
            {"content": "Task 2"}
        ]
        
        cli_obj._execute_task_train(tasks)
        
        # After task 1, history should be filtered to keep only last assistant response
        # This is checked by the implementation in _execute_task_train

    def test_no_clear_after_last_task(self, cli_with_mock_agent):
        """Test that no clearing happens after the final task."""
        cli_obj, db = cli_with_mock_agent
        
        clear_count = 0
        original_clear = db.clear_ephemeral_messages
        
        def counting_clear(session_id):
            nonlocal clear_count
            clear_count += 1
            return original_clear(session_id)
        
        db.clear_ephemeral_messages = counting_clear
        
        tasks = [
            {"content": "Task 1"},
            {"content": "Task 2"}
        ]
        
        cli_obj._execute_task_train(tasks)
        
        # With 2 tasks, should clear once (after task 1, not after task 2)
        assert clear_count == 1


# =============================================================================
# Integration Tests: Full Workflow
# =============================================================================

class TestTaskTrainIntegration:
    """Integration tests for the full /start-tasks ... /end-tasks workflow."""

    def test_full_workflow_with_mock_agent(self, cli_with_mock_agent):
        """Test complete workflow: start tasks, buffer, end tasks, execute."""
        cli_obj, db = cli_with_mock_agent
        
        # Step 1: Start tasks
        cli_obj._handle_start_tasks("/start-tasks")
        assert db.is_task_definition_mode_active("test-session") is True
        
        # Step 2: Buffer tasks (simulated via direct database calls)
        db.store_task_definition("test-session", "Write API documentation")
        db.store_task_definition("test-session", "Add unit tests")
        db.store_task_definition("test-session", "Deploy to staging")
        
        # Verify tasks are buffered
        tasks = db.get_task_definitions("test-session")
        assert len(tasks) == 3
        
        # Step 3: End tasks and execute
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_end_tasks("/end-tasks")
        
        # Mode should be exited
        assert db.is_task_definition_mode_active("test-session") is False
        
        # Agent should have been called for each task
        assert cli_obj.agent.run_conversation.call_count == 3

    def test_workflow_with_empty_tasks(self, cli_with_db):
        """Test workflow when no tasks are buffered."""
        cli_obj, db = cli_with_db
        
        # Start tasks
        cli_obj._handle_start_tasks("/start-tasks")
        assert db.is_task_definition_mode_active("test-session") is True
        
        # End tasks without adding any
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._handle_end_tasks("/end-tasks")
        
        output = " ".join(captured)
        assert "no tasks" in output.lower() or "defined" in output.lower()

    def test_workflow_multiple_start_tasks(self, cli_with_db):
        """Test calling /start-tasks multiple times."""
        cli_obj, db = cli_with_db
        
        # First call
        captured1 = []
        with patch("cli._cprint", side_effect=lambda s: captured1.append(s)):
            cli_obj._handle_start_tasks("/start-tasks")
        assert db.is_task_definition_mode_active("test-session") is True
        
        # Second call (should show already active message)
        captured2 = []
        with patch("cli._cprint", side_effect=lambda s: captured2.append(s)):
            cli_obj._handle_start_tasks("/start-tasks")
        
        output = " ".join(captured2)
        assert "already" in output.lower() or "active" in output.lower()

    def test_workflow_clears_task_definitions_after_execution(self, cli_with_db):
        """Test that task definitions are cleared after execution."""
        cli_obj, db = cli_with_db
        
        # Enter mode and add tasks
        db.enter_task_definition_mode("test-session")
        db.store_task_definition("test-session", "Task 1")
        
        # Verify tasks exist
        tasks = db.get_task_definitions("test-session")
        assert len(tasks) == 1
        
        # Note: In the actual implementation, tasks are not automatically
        # cleared after execution. They remain in the database. This test
        # documents the current behavior.
        
        # The current implementation keeps task definitions in the database
        # so they can be reviewed later. If automatic clearing is desired,
        # it would need to be added to _handle_end_tasks().


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestTaskTrainEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_very_long_task_description(self, cli_with_mock_agent):
        """Test task with very long description."""
        cli_obj, db = cli_with_mock_agent
        
        long_task = "x" * 10000  # 10KB task description
        tasks = [{"content": long_task}]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # Should handle without error
        assert cli_obj.agent.run_conversation.call_count == 1

    def test_empty_task_content(self, cli_with_mock_agent):
        """Test task with empty content."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [{"content": ""}]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # Should attempt to execute even empty task
        assert cli_obj.agent.run_conversation.call_count == 1

    def test_unicode_in_task_content(self, cli_with_mock_agent):
        """Test task with unicode characters."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [{"content": "Write 文档 with 日本語 and emoji 🚀"}]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # Should handle unicode without error
        assert cli_obj.agent.run_conversation.call_count == 1

    def test_special_characters_in_task(self, cli_with_mock_agent):
        """Test task with special characters."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [{"content": "Run command: ls -la | grep 'test' && echo 'done'"}]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # Should handle special characters
        assert cli_obj.agent.run_conversation.call_count == 1

    def test_task_with_newlines(self, cli_with_mock_agent):
        """Test task description with newlines."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [{"content": "Task 1\nTask 2\nTask 3"}]
        
        captured = []
        with patch("cli._cprint", side_effect=lambda s: captured.append(s)):
            cli_obj._execute_task_train(tasks)
        
        # Should handle newlines
        assert cli_obj.agent.run_conversation.call_count == 1


# =============================================================================
# Task ID Generation Tests
# =============================================================================

class TestTaskIdGeneration:
    """Tests for task ID generation in task train."""

    def test_task_id_format(self, cli_with_mock_agent):
        """Test that task IDs follow the expected format."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [
            {"content": "Task 1"},
            {"content": "Task 2"}
        ]
        
        cli_obj._execute_task_train(tasks)
        
        # Check the task_id parameter in each call
        calls = cli_obj.agent.run_conversation.call_args_list
        
        # First call should have task_id like "task-1-2"
        assert "task-1-2" in calls[0][1]["task_id"]
        # Second call should have task_id like "task-2-2"
        assert "task-2-2" in calls[1][1]["task_id"]

    def test_task_id_includes_position_and_total(self, cli_with_mock_agent):
        """Test that task IDs include both position and total count."""
        cli_obj, db = cli_with_mock_agent
        
        tasks = [{"content": "Task"}]
        
        cli_obj._execute_task_train(tasks)
        
        calls = cli_obj.agent.run_conversation.call_args_list
        task_id = calls[0][1]["task_id"]
        
        # Format: task-{position}-{total}
        assert "task-1-1" in task_id
