"""Integration tests that measure actual serialized context size
before and after /clear-context, using a mock model provider.

Unlike test_clear_context_integration.py (which only checks DB row counts),
these tests exercise the full message-assembly pipeline that produces the
api_messages payload sent to the LLM API, and verify that /clear-context
actually reduces the serialized byte size.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cli import HermesCLI
from hermes_state import SessionDB
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NO_TOOL_RESPONSE = SimpleNamespace(
    choices=[SimpleNamespace(
        message=SimpleNamespace(
            content="Done.",
            tool_calls=None,
            tool_call_id=None,
            function_call=None,
        ),
        finish_reason="stop",
        index=0,
    )],
    model="test-model",
)


def _make_cli_with_db(tmp_path):
    """Build a minimal HermesCLI stub wired to a real SessionDB.
    
    Includes conversation_history attribute which is required by
    _handle_clear_context() for before/after message counting.
    """
    db_path = tmp_path / "size_measurement.db"
    db = SessionDB(db_path=db_path)
    db.create_session("test-session", "cli")

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "test-session"
    cli_obj._session_db = db
    cli_obj.system_prompt = "You are a helpful assistant."
    cli_obj._app = None
    cli_obj.agent = None
    cli_obj.conversation_history = []  # Required by _handle_clear_context
    return cli_obj, db


def _populate_heavy_session(db, session_id):
    """Populate a session with realistic messages including LARGE tool outputs."""
    db.append_message(session_id, "user",
                      "Analyze all the Python files in this project.")

    db.append_message(session_id, "assistant",
                      "Searching for Python files...",
                      tool_calls=[{"id": "tc1"}])

    big_file_listing = "\n".join(
        f"# File {i}: {'a' * 80}\n" + "print('hello world')\n" * 20
        for i in range(30)
    )
    db.append_message(session_id, "tool", json.dumps({
        "stdout": big_file_listing,
        "exit_code": 0,
    }))

    db.append_message(session_id, "assistant",
                      "Found 30 files. Reading the important ones.",
                      tool_calls=[{"id": "tc2"}])

    for _i in range(5):
        big_content = "\n".join(
            f"line_{j}: {'x' * 100}" for j in range(200)
        )
        db.append_message(session_id, "tool", json.dumps({
            "stdout": big_content,
            "exit_code": 0,
        }))

    db.append_message(session_id, "assistant",
                      "I've read the files. Analyzing now.",
                      tool_calls=[{"id": "tc3"}])

    analysis_output = "\n".join(
        f"Analysis point {n}: {'=' * 120} detail {'=' * 120}"
        for n in range(50)
    )
    db.append_message(session_id, "tool", json.dumps({
        "stdout": analysis_output,
        "exit_code": 0,
    }))

    db.append_message(session_id, "assistant",
                      "Here is my summary:\n\n" + "Important finding " * 20)

    db.append_message(session_id, "user",
                      "Great. Now also check the test files.")

    db.append_message(session_id, "assistant",
                      "Running tests...",
                      tool_calls=[{"id": "tc4"}])

    test_output = "\n".join(
        f"test_function_{n}: PASSED ({'.' * 50} output details)"
        for n in range(40)
    )
    db.append_message(session_id, "tool", json.dumps({
        "stdout": test_output,
        "exit_code": 0,
    }))

    db.append_message(session_id, "assistant", "All tests pass.")


def _run_single_turn_and_capture(db, session_id, system_prompt, captured: list):
    """Build an AIAgent, run one conversation turn with a mocked API call.

    The mock captures the api_kwargs dict (which contains the fully-assembled
    api_messages list including system prompt, tool schemas, etc.) and
    returns a no-tool-call response so the loop exits after 1 iteration.

    Appends exactly one api_kwargs dict to the captured list.
    """
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    def fake_interruptible_call(self_, api_kwargs):
        """Bypass the real threaded API call — capture and return dummy response."""
        captured.append(dict(api_kwargs))  # snapshot the kwargs
        return _NO_TOOL_RESPONSE

    with patch("run_agent.get_tool_definitions", return_value=tool_defs), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch.object(AIAgent, "_interruptible_api_call", fake_interruptible_call):

        agent = AIAgent(
            api_key="test-key",
            model="qwen/qwen3.5",  # generic name so no auto-detection kicks in
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            api_mode="chat_completions",  # force OpenAI-style chat completions
        )
        agent._session_db = db
        agent.session_id = session_id
        agent.conversation_history = db.get_messages_as_conversation(session_id)
        # Disable DB flushing — we only want to measure, not mutate
        agent._flush_messages_to_session_db = lambda *a, **kw: None

        agent.run_conversation(
            user_message="Continue.",
            system_message=system_prompt,
            conversation_history=agent.conversation_history,
        )

    assert len(captured) == 1, f"Expected 1 API call, got {len(captured)}"
    return captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClearContextReducesPayloadSize:
    """Verify /clear-context measurably reduces the serialized API payload."""

    def test_clear_context_reduces_serialized_message_size(self, tmp_path, monkeypatch):
        """Heavy session with large tool outputs → clear-context → verify
        the serialized message payload decreased."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        _populate_heavy_session(db, sid)
        before_count = db.message_count(sid)
        assert before_count > 10, f"Expected many messages, got {before_count}"

        # Pre-clear measurement
        captured_pre = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_pre)
        pre_size = len(json.dumps(captured_pre[0]["messages"], ensure_ascii=False))

        # Run /clear-context
        with patch("cli._cprint"):
            cli_obj._handle_clear_context("/clear-context")

        after_count = db.message_count(sid)
        assert after_count < before_count, (
            f"Expected fewer after clear (was {before_count}, now {after_count})"
        )

        # Post-clear measurement
        captured_post = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_post)
        post_size = len(json.dumps(captured_post[0]["messages"], ensure_ascii=False))

        reduction_pct = (1 - post_size / pre_size) * 100
        assert post_size < pre_size, (
            f"Post-clear ({post_size:,}B) should be < pre-clear ({pre_size:,}B)"
        )
        # Tool outputs tend to be large — expect meaningful reduction
        assert reduction_pct > 10, (
            f"Expected >10% reduction, got {reduction_pct:.1f}%"
        )

    def test_clear_context_no_ephemeral_stays_same(self, tmp_path, monkeypatch):
        """If there is nothing to clear, context size should stay stable."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        # Only user + assistant, no tool outputs
        for i in range(5):
            db.append_message(sid, "user", f"Question {i}")
            db.append_message(sid, "assistant", f"Answer {i}")

        captured_pre = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_pre)
        pre_size = len(json.dumps(captured_pre[0]["messages"], ensure_ascii=False))

        with patch("cli._cprint"):
            cli_obj._handle_clear_context("/clear-context")

        captured_post = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_post)
        post_size = len(json.dumps(captured_post[0]["messages"], ensure_ascii=False))

        diff = abs(pre_size - post_size)
        assert diff < pre_size * 0.05, (
            f"No-op clear should not change size by >5% (pre={pre_size}, post={post_size})"
        )

    def test_recent_mode_reduces_size(self, tmp_path, monkeypatch):
        """/clear-context recent (prune last N) should reduce context size."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        for i in range(20):
            db.append_message(sid, "user", f"Q{i}: {'x' * 100}")
            db.append_message(sid, "assistant", f"A{i}: {'y' * 100}")

        captured_pre = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_pre)
        pre_size = len(json.dumps(captured_pre[0]["messages"], ensure_ascii=False))

        with patch("cli._cprint"):
            cli_obj._handle_clear_context("/clear-context recent")

        captured_post = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_post)
        post_size = len(json.dumps(captured_post[0]["messages"], ensure_ascii=False))

        assert post_size < pre_size, (
            f"Post-recent-clear ({post_size}) should be < pre-clear ({pre_size})"
        )


class TestClearContextAssemblyFormat:
    """Verify the mock captures correctly-assembled API payloads."""

    def test_messages_include_system_prompt(self, tmp_path, monkeypatch):
        """The assembled api_messages should contain the system prompt."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "user", "Hello")
        cli_obj.system_prompt = "You are TESTBOT, a testing assistant."

        captured = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured)

        messages = captured[0]["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles

        sys_msg = next(m for m in messages if m["role"] == "system")
        assert "TESTBOT" in str(sys_msg.get("content", ""))

    def test_large_tool_content_present_before_clear(self, tmp_path, monkeypatch):
        """Before clear-context, a large tool output should be in the payload."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        cli_obj, db = _make_cli_with_db(tmp_path)
        sid = cli_obj.session_id

        db.append_message(sid, "user", "Run this")
        db.append_message(sid, "assistant", "Running...", tool_calls=[{"id": "tc1"}])
        marker = "UNIQUE_" + "X" * 5000 + "_MARKER_END"
        db.append_message(sid, "tool", json.dumps({
            "stdout": marker,
            "exit_code": 0,
        }))
        db.append_message(sid, "assistant", "Done.")

        captured_pre = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_pre)
        all_text_pre = json.dumps(captured_pre[0]["messages"], ensure_ascii=False)
        pre_size = len(all_text_pre)

        assert "UNIQUE_" in all_text_pre, "Tool output should be present before clear"
        assert "_MARKER_END" in all_text_pre

        # Clear and re-measure
        with patch("cli._cprint"):
            cli_obj._handle_clear_context("/clear-context")

        captured_post = []
        _run_single_turn_and_capture(db, sid, cli_obj.system_prompt, captured_post)
        all_text_post = json.dumps(captured_post[0]["messages"], ensure_ascii=False)
        post_size = len(all_text_post)

        # The UNIQUE marker should NOT be in post-clear payload
        assert "UNIQUE_" not in all_text_post, \
            "Tool UNIQUE marker should be cleared from payload"
        assert "_MARKER_END" not in all_text_post, \
            "Tool _MARKER_END should be cleared from payload"
        assert post_size < pre_size, \
            f"Post-clear ({post_size}B) < pre-clear ({pre_size}B)"
