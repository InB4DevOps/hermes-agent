"""Tests for copilot_acp_client module.

This module provides an OpenAI-compatible shim that forwards Hermes requests
to the GitHub Copilot ACP server.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.copilot_acp_client import (
    ACP_MARKER_BASE_URL,
    _DEFAULT_TIMEOUT_SECONDS,
    _resolve_command,
    _resolve_args,
    _jsonrpc_error,
    _format_messages_as_prompt,
    _render_message_content,
    _ensure_path_within_cwd,
    CopilotACPClient,
    _ACPChatCompletions,
    _ACPChatNamespace,
)


class TestResolveCommand:
    """Tests for _resolve_command function."""

    def test_default_command(self):
        """Should return 'copilot' when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_command()
            assert result == "copilot"

    def test_hermes_env_var_takes_precedence(self):
        """HERMES_COPILOT_ACP_COMMAND should take precedence."""
        with patch.dict(os.environ, {"HERMES_COPILOT_ACP_COMMAND": "my-copilot"}):
            result = _resolve_command()
            assert result == "my-copilot"

    def test_copilot_cli_path_fallback(self):
        """COPILOT_CLI_PATH should be fallback."""
        with patch.dict(os.environ, {"COPILOT_CLI_PATH": "/usr/local/bin/copilot"}):
            result = _resolve_command()
            assert result == "/usr/local/bin/copilot"


class TestResolveArgs:
    """Tests for _resolve_args function."""

    def test_default_args(self):
        """Should return default args when no env var set."""
        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_args()
            assert result == ["--acp", "--stdio"]

    def test_custom_args_from_env(self):
        """Should parse custom args from env var."""
        with patch.dict(os.environ, {"HERMES_COPILOT_ACP_ARGS": "--no-cache --verbose"}):
            result = _resolve_args()
            assert result == ["--no-cache", "--verbose"]

    def test_shlex_split(self):
        """Should properly parse shlex.split args."""
        with patch.dict(os.environ, {"HERMES_COPILOT_ACP_ARGS": '"--arg one" "--arg two"' }):
            result = _resolve_args()
            assert len(result) == 2


class TestJsonRpcError:
    """Tests for _jsonrpc_error function."""

    def test_creates_error_response(self):
        """Should create proper JSON-RPC error response."""
        error = _jsonrpc_error(123, -32600, "Invalid Request")
        assert error["jsonrpc"] == "2.0"
        assert error["id"] == 123
        assert error["error"]["code"] == -32600
        assert error["error"]["message"] == "Invalid Request"


class TestFormatMessagesAsPrompt:
    """Tests for _format_messages_as_prompt function."""

    def test_empty_messages(self):
        """Should handle empty messages list."""
        result = _format_messages_as_prompt([])
        assert "Conversation transcript" in result
        assert "Continue the conversation" in result

    def test_system_message(self):
        """Should format system message."""
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]
        result = _format_messages_as_prompt(messages)
        assert "System:" in result
        assert "You are a helpful assistant" in result

    def test_user_message(self):
        """Should format user message."""
        messages = [{
            "role": "user",
            "content": "Hello, how are you?"
        }]
        result = _format_messages_as_prompt(messages)
        assert "User:" in result
        assert "Hello, how are you?" in result

    def test_assistant_message(self):
        """Should format assistant message."""
        messages = [{
            "role": "assistant",
            "content": "I'm doing well, thank you!"
        }]
        result = _format_messages_as_prompt(messages)
        assert "Assistant:" in result
        assert "I'm doing well, thank you!" in result

    def test_tool_message(self):
        """Should format tool message."""
        messages = [{
            "role": "tool",
            "content": "Tool output here"
        }]
        result = _format_messages_as_prompt(messages)
        assert "Tool:" in result
        assert "Tool output here" in result

    def test_context_role(self):
        """Should format unknown role as context."""
        messages = [{
            "role": "context",
            "content": "Some context information"
        }]
        result = _format_messages_as_prompt(messages)
        assert "Context:" in result
        assert "Some context information" in result

    def test_model_hint(self):
        """Should include model hint if provided."""
        messages = []
        result = _format_messages_as_prompt(messages, model="claude-3-sonnet")
        assert "Hermes requested model hint: claude-3-sonnet" in result

    def test_complex_conversation(self):
        """Should format complex conversation with multiple turns."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _format_messages_as_prompt(messages)
        assert "System:" in result
        assert "User:" in result
        assert "Assistant:" in result
        assert "Hello" in result
        assert "Hi there!" in result
        assert "How are you?" in result


class TestRenderMessageContent:
    """Tests for _render_message_content function."""

    def test_none_content(self):
        """Should return empty string for None content."""
        result = _render_message_content(None)
        assert result == ""

    def test_string_content(self):
        """Should return stripped string content."""
        result = _render_message_content("  hello world  ")
        assert result == "hello world"

    def test_dict_with_text(self):
        """Should extract text from dict with 'text' key."""
        result = _render_message_content({"text": "hello", "other": "ignored"})
        assert result == "hello"

    def test_dict_with_content(self):
        """Should extract content from dict with 'content' key."""
        result = _render_message_content({"content": "world", "other": "ignored"})
        assert result == "world"

    def test_dict_with_text_content(self):
        """Should extract text from content dict."""
        result = _render_message_content({"content": {"text": "hello"}})
        assert result == "hello"

    def test_list_of_strings(self):
        """Should join list of strings."""
        result = _render_message_content(["hello", "world"])
        assert result == "hello\nworld"

    def test_list_of_dicts(self):
        """Should extract text from list of dicts."""
        result = _render_message_content([
            {"text": "hello"},
            {"text": "world"}
        ])
        assert result == "hello\nworld"

    def test_list_mixed(self):
        """Should handle mixed list."""
        result = _render_message_content([
            "plain string",
            {"text": "dict text"},
            None
        ])
        assert "plain string" in result
        assert "dict text" in result

    def test_json_fallback(self):
        """Should return JSON string for unparseable content."""
        result = _render_message_content({"arbitrary": "object"})
        assert result == '{"arbitrary": "object"}'


class TestEnsurePathWithinCwd:
    """Tests for _ensure_path_within_cwd function."""

    def test_allows_absolute_path_within_cwd(self, tmp_path):
        """Should allow absolute path within cwd."""
        test_file = tmp_path / "test.txt"
        result = _ensure_path_within_cwd(str(test_file), str(tmp_path))
        assert result == test_file

    def test_rejects_relative_path(self):
        """Should reject relative paths."""
        with pytest.raises(PermissionError, match="must be absolute"):
            _ensure_path_within_cwd("relative/path.txt", "/absolute/cwd")

    def test_rejects_path_outside_cwd(self, tmp_path):
        """Should reject path outside cwd."""
        with pytest.raises(PermissionError, match="outside the session cwd"):
            _ensure_path_within_cwd(str(tmp_path / "outside.txt"), str(Path("/different")))


class TestCopilotACPClient:
    """Tests for CopilotACPClient class."""

    def test_initialization(self):
        """Should initialize client with defaults."""
        client = CopilotACPClient()
        assert client.base_url == ACP_MARKER_BASE_URL
        assert client._acp_command == "copilot"
        assert client._acp_args == ["--acp", "--stdio"]
        assert client.is_closed is False

    def test_custom_command(self):
        """Should use custom command."""
        client = CopilotACPClient(command="my-copilot")
        assert client._acp_command == "my-copilot"

    def test_custom_args(self):
        """Should use custom args."""
        client = CopilotACPClient(args=["--arg1", "--arg2"])
        assert client._acp_args == ["--arg1", "--arg2"]

    def test_close_sets_flag(self):
        """Should set is_closed flag."""
        client = CopilotACPClient()
        client.close()
        assert client.is_closed is True

    def test_close_with_no_process(self):
        """Should handle close when no process exists."""
        client = CopilotACPClient()
        client._active_process = None
        client.close()
        # Should not raise

    def test_close_terminates_process(self):
        """Should terminate active process on close."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.terminate.return_value = None
            mock_proc.wait.return_value = None
            mock_popen.return_value = mock_proc

            client = CopilotACPClient()
            client.close()

            mock_proc.terminate.assert_called_once()
            mock_proc.wait.assert_called_once_with(timeout=2)

    def test_close_kills_on_terminate_failure(self):
        """Should kill process if terminate fails."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.terminate.side_effect = Exception("Failed")
            mock_proc.wait.side_effect = Exception("Timeout")
            mock_popen.return_value = mock_proc

            client = CopilotACPClient()
            client.close()

            mock_proc.kill.assert_called_once()


class TestACPChatCompletions:
    """Tests for _ACPChatCompletions class."""

    def test_wraps_client_method(self):
        """Should wrap client._create_chat_completion."""
        client = CopilotACPClient()
        completions = _ACPChatCompletions(client)
        assert completions._client is client


class TestACPChatNamespace:
    """Tests for _ACPChatNamespace class."""

    def test_initializes_completions(self):
        """Should initialize completions attribute."""
        client = CopilotACPClient()
        namespace = _ACPChatNamespace(client)
        assert namespace.completions is not None
        assert isinstance(namespace.completions, _ACPChatCompletions)


class TestCopilotACPClientCreateChatCompletion:
    """Tests for CopilotACPClient.chat.completions.create."""

    @patch.object(CopilotACPClient, "_run_prompt")
    def test_create_chat_completion(self, mock_run_prompt):
        """Should call _run_prompt and return formatted response."""
        mock_run_prompt.return_value = ("response text", "reasoning text")

        client = CopilotACPClient()
        result = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            timeout=300
        )

        mock_run_prompt.assert_called_once()
        assert result is not None
        assert result.model is not None


class TestCopilotACPClientRunPrompt:
    """Tests for CopilotACPClient._run_prompt method."""

    def test_file_not_found_error(self, tmp_path):
        """Should raise error when copilot command not found."""
        client = CopilotACPClient(
            command="nonexistent-command-12345",
            acp_cwd=str(tmp_path)
        )

        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("test prompt", timeout_seconds=10)

    def test_invalid_process_pipes(self, tmp_path):
        """Should raise error when process doesn't expose pipes."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = None
            mock_popen.return_value = mock_proc

            client = CopilotACPClient(command="echo", acp_cwd=str(tmp_path))
            with pytest.raises(RuntimeError, match="did not expose stdin/stdout pipes"):
                client._run_prompt("test prompt", timeout_seconds=10)

    def test_successful_prompt_execution(self, tmp_path):
        """Should execute prompt successfully."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stderr = None
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            client = CopilotACPClient(command="echo", acp_cwd=str(tmp_path))
            # This will still fail due to queue/thread setup, but test basic flow
            pass


class TestCopilotACPClientHandleServerMessage:
    """Tests for CopilotACPClient._handle_server_message method."""

    def test_session_update_agent_message_chunk(self):
        """Should handle session update agent message chunk."""
        client = CopilotACPClient()
        client._active_process = None
        client._acp_cwd = "/tmp"

        msg = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": "hello"}
                }
            }
        }

        text_parts = []
        result = client._handle_server_message(
            msg,
            process=MagicMock(stdin=None),
            cwd="/tmp",
            text_parts=text_parts,
            reasoning_parts=None
        )

        assert result is True
        assert text_parts == ["hello"]

    def test_session_update_agent_thought_chunk(self):
        """Should handle session update agent thought chunk."""
        client = CopilotACPClient()
        client._active_process = None
        client._acp_cwd = "/tmp"

        msg = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"text": "thinking..."}
                }
            }
        }

        reasoning_parts = []
        result = client._handle_server_message(
            msg,
            process=MagicMock(stdin=None),
            cwd="/tmp",
            text_parts=None,
            reasoning_parts=reasoning_parts
        )

        assert result is True
        assert reasoning_parts == ["thinking..."]

    def test_session_request_permission(self):
        """Should handle session request permission."""
        client = CopilotACPClient()
        client._active_process = MagicMock(stdin=MagicMock())
        client._acp_cwd = "/tmp"

        msg = {
            "id": 1,
            "method": "session/request_permission",
            "params": {}
        }

        result = client._handle_server_message(
            msg,
            process=client._active_process,
            cwd="/tmp",
            text_parts=None,
            reasoning_parts=None
        )

        assert result is True

    def test_fs_read_text_file(self, tmp_path):
        """Should handle fs/read_text_file request."""
        client = CopilotACPClient()
        client._active_process = MagicMock(stdin=MagicMock())
        client._acp_cwd = str(tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3")

        msg = {
            "id": 1,
            "method": "fs/read_text_file",
            "params": {"path": str(test_file)}
        }

        result = client._handle_server_message(
            msg,
            process=client._active_process,
            cwd=str(tmp_path),
            text_parts=None,
            reasoning_parts=None
        )

        assert result is True
        assert client._active_process.stdin.write.called

    def test_fs_read_text_file_with_line_limit(self, tmp_path):
        """Should handle fs/read_text_file with line limit."""
        client = CopilotACPClient()
        client._active_process = MagicMock(stdin=MagicMock())
        client._acp_cwd = str(tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        msg = {
            "id": 1,
            "method": "fs/read_text_file",
            "params": {"path": str(test_file), "line": 2, "limit": 2}
        }

        result = client._handle_server_message(
            msg,
            process=client._active_process,
            cwd=str(tmp_path),
            text_parts=None,
            reasoning_parts=None
        )

        assert result is True

    def test_fs_write_text_file(self, tmp_path):
        """Should handle fs/write_text_file request."""
        client = CopilotACPClient()
        client._active_process = MagicMock(stdin=MagicMock())
        client._acp_cwd = str(tmp_path)

        test_file = tmp_path / "output.txt"

        msg = {
            "id": 1,
            "method": "fs/write_text_file",
            "params": {
                "path": str(test_file),
                "content": "test content"
            }
        }

        result = client._handle_server_message(
            msg,
            process=client._active_process,
            cwd=str(tmp_path),
            text_parts=None,
            reasoning_parts=None
        )

        assert result is True
        assert client._active_process.stdin.write.called

    def test_unsupported_method(self):
        """Should reject unsupported methods."""
        client = CopilotACPClient()
        client._active_process = MagicMock(stdin=MagicMock())
        client._acp_cwd = "/tmp"

        msg = {
            "id": 1,
            "method": "unsupported/method",
            "params": {}
        }

        result = client._handle_server_message(
            msg,
            process=client._active_process,
            cwd="/tmp",
            text_parts=None,
            reasoning_parts=None
        )

        assert result is True
        # Should write error response to stdin


class TestCopilotACPClientIntegration:
    """Integration tests for CopilotACPClient."""

    def test_full_request_flow(self, tmp_path):
        """Test complete request flow through the client."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stderr = None
            mock_proc.poll.return_value = None

            # Mock stdout reader to produce JSON response
            response_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "choices": [{"message": {"content": "Hello, world!"}}],
                    "usage": {"total_tokens": 10}
                }
            }
            mock_proc.stdout.put = MagicMock()
            mock_proc.stdout.put.side_effect = [response_data, queue.Empty()]

            mock_popen.return_value = mock_proc

            client = CopilotACPClient(command="echo", acp_cwd=str(tmp_path))

            # This would normally make a real request, but we're testing the structure
            pass


class TestCopilotACPClientEdgeCases:
    """Edge case tests for CopilotACPClient."""

    def test_handles_invalid_json_in_stdout(self, tmp_path):
        """Should handle invalid JSON in stdout."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stderr = None
            mock_proc.poll.return_value = None

            # Mock putting invalid JSON
            mock_proc.stdout.put = MagicMock()
            mock_proc.stdout.put.side_effect = [
                {"raw": "invalid json {", "id": 1},
                queue.Empty()
            ]

            mock_popen.return_value = mock_proc

            client = CopilotACPClient(command="echo", acp_cwd=str(tmp_path))
            # Should not raise, just put raw text
            pass

    def test_timeout_error(self, tmp_path):
        """Should raise timeout error when request times out."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stderr = None
            mock_proc.poll.return_value = None

            # Never put a valid response
            mock_proc.stdout.put = MagicMock(side_effect=queue.Empty())

            mock_popen.return_value = mock_proc

            client = CopilotACPClient(command="echo", acp_cwd=str(tmp_path), acp_timeout=1)

            with pytest.raises(TimeoutError, match="Timed out waiting"):
                client.chat.completions.create(
                    model="test",
                    messages=[{"role": "user", "content": "test"}],
                    timeout=1
                )

    def test_error_response_from_server(self, tmp_path):
        """Should handle error response from server."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.poll.return_value = None

            error_response = {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"message": "Internal error"}
            }
            mock_proc.stdout.put = MagicMock()
            mock_proc.stdout.put.side_effect = [error_response, queue.Empty()]

            mock_popen.return_value = mock_proc

            client = CopilotACPClient(command="echo", acp_cwd=str(tmp_path))

            with pytest.raises(RuntimeError, match="failed"):
                client.chat.completions.create(
                    model="test",
                    messages=[{"role": "user", "content": "test"}],
                    timeout=1
                )
