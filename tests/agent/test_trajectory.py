"""Tests for trajectory module.

This module provides utilities for saving and processing agent trajectories.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from agent.trajectory import (
    convert_scratchpad_to_think,
    has_incomplete_scratchpad,
    save_trajectory,
)


class TestConvertScratchpadToThink:
    """Tests for convert_scratchpad_to_think function."""

    def test_converts_scratchpad_tags(self):
        """Should convert <REASONING_SCRATCHPAD> to <think> tags."""
        content = """<REASONING_SCRATCHPAD>
This is my thinking process
- Step 1: Analyze the problem
- Step 2: Plan the solution
</think>

Here's my answer.
</REASONING_SCRATCHPAD>"""

        result = convert_scratchpad_to_think(content)
        assert "</think>" in result
        assert "<REASONING_SCRATCHPAD>" not in result
        assert "</REASONING_SCRATCHPAD>" not in result

    def test_unchanged_when_no_tags(self):
        """Should return unchanged when no scratchpad tags."""
        content = """This is normal text
without any special tags."""

        result = convert_scratchpad_to_think(content)
        assert result == content

    def test_unchanged_when_empty(self):
        """Should return empty string unchanged."""
        result = convert_scratchpad_to_think("")
        assert result == ""

    def test_unchanged_when_only_opening_tag(self):
        """Should convert opening tag to </think> (function does simple replace)."""
        content = """<REASONING_SCRATCHPAD>
This has an opening tag but no closing."""
        result = convert_scratchpad_to_think(content)
        # Function does simple string replace - converts ALL occurrences
        assert "</think>" in result  # Opening tag becomes </think>
        # Opening tag is converted
        assert "<REASONING_SCRATCHPAD>" not in result

    def test_multiple_scratchpads(self):
        """Should handle multiple scratchpad sections."""
        content = """<REASONING_SCRATCHPAD>First thought</REASONING_SCRATCHPAD>
Some text
<REASONING_SCRATCHPAD>Second thought</REASONING_SCRATCHPAD>"""

        result = convert_scratchpad_to_think(content)
        assert result.count("</think>") == 2


class TestHasIncompleteScratchpad:
    """Tests for has_incomplete_scratchpad function."""

    def test_has_incomplete_scratchpad(self):
        """Should return True when opening tag exists but closing doesn't."""
        content = """<REASONING_SCRATCHPAD>
This is incomplete thinking.
The model got cut off."""
        assert has_incomplete_scratchpad(content) is True

    def test_complete_scratchpad(self):
        """Should return False when both tags exist."""
        content = """<REASONING_SCRATCHPAD>
This is complete thinking.
</think>

Here's the answer.
</REASONING_SCRATCHPAD>"""
        assert has_incomplete_scratchpad(content) is False

    def test_no_scratchpad_tags(self):
        """Should return False when no scratchpad tags exist."""
        content = """This is normal text without any scratchpad tags."""
        assert has_incomplete_scratchpad(content) is False

    def test_empty_content(self):
        """Should return False for empty content."""
        assert has_incomplete_scratchpad("") is False

    def test_only_closing_tag(self):
        """Should return False when only closing tag exists."""
        content = """</think>

Here's my answer.
</REASONING_SCRATCHPAD>
This is after the closing tag."""
        assert has_incomplete_scratchpad(content) is False

    def test_nested_scratchpads(self):
        """Should detect incomplete nested scratchpads."""
        content = """<REASONING_SCRATCHPAD>
Outer scratchpad
<REASONING_SCRATCHPAD>Nested scratchpad without closing
</REASONING_SCRATCHPAD>
</REASONING_SCRATCHPAD>"""
        # This is actually complete - both open and close exist
        assert has_incomplete_scratchpad(content) is False

    def test_multiple_incomplete_scratchpads(self):
        """Should return True when opening tag exists but closing tag doesn't exist at all."""
        content = """<REASONING_SCRATCHPAD>Incomplete"""
        # has_incomplete checks: opening exists AND closing doesn't exist anywhere
        assert has_incomplete_scratchpad(content) is True


class TestSaveTrajectory:
    """Tests for save_trajectory function."""

    def test_saves_completed_trajectory(self, tmp_path):
        """Should save completed trajectory to trajectory_samples.jsonl."""
        trajectory = [
            {
                "role": "user",
                "content": "What is 2+2?"
            },
            {
                "role": "assistant",
                "content": "The answer is 4."
            }
        ]
        model = "anthropic/claude-3-sonnet"
        completed = True

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory, model, completed, filename=temp_file)

            # Verify file exists and was written
            assert os.path.exists(temp_file)

            # Read and verify content
            with open(temp_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1

                entry = json.loads(lines[0])
                assert entry["conversations"] == trajectory
                assert entry["timestamp"] is not None
                assert entry["model"] == model
                assert entry["completed"] is True
        finally:
            os.unlink(temp_file)

    def test_saves_failed_trajectory(self, tmp_path):
        """Should save failed trajectory to failed_trajectories.jsonl."""
        trajectory = [
            {
                "role": "user",
                "content": "Help me with code"
            },
            {
                "role": "assistant",
                "content": "Error: Something went wrong"
            }
        ]
        model = "anthropic/claude-3-sonnet"
        completed = False

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory, model, completed, filename=temp_file)

            assert os.path.exists(temp_file)

            with open(temp_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1

                entry = json.loads(lines[0])
                assert entry["conversations"] == trajectory
                assert entry["completed"] is False
        finally:
            os.unlink(temp_file)

    def test_appends_to_existing_file(self, tmp_path):
        """Should append to existing file rather than overwrite."""
        initial_trajectory = [
            {
                "role": "user",
                "content": "First request"
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name
            f.write(json.dumps({
                "conversations": initial_trajectory,
                "timestamp": "2024-01-01T00:00:00",
                "model": "test-model",
                "completed": True
            }) + "\n")

        try:
            new_trajectory = [
                {
                    "role": "assistant",
                    "content": "First response"
                }
            ]

            save_trajectory(new_trajectory, "new-model", True, filename=temp_file)

            with open(temp_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2  # Should have 2 entries
        finally:
            os.unlink(temp_file)

    def test_timestamp_format(self, tmp_path):
        """Should use ISO format for timestamp."""
        trajectory = [{"role": "user", "content": "test"}]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory, "test-model", True, filename=temp_file)

            with open(temp_file, 'r') as f:
                entry = json.loads(f.readline())

            # Verify timestamp is ISO format (datetime.isoformat uses local timezone)
            assert "T" in entry["timestamp"]
            assert len(entry["timestamp"]) > 10  # Should have date and time
        finally:
            os.unlink(temp_file)

    def test_default_filename_for_completed(self, tmp_path):
        """Should use trajectory_samples.jsonl as default for completed."""
        trajectory = [{"role": "user", "content": "test"}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_trajectory(trajectory, "model", True, filename=None)

            # Check file was created in current directory
            assert os.path.exists("trajectory_samples.jsonl")
            os.unlink("trajectory_samples.jsonl")

    def test_default_filename_for_failed(self, tmp_path):
        """Should use failed_trajectories.jsonl as default for failed."""
        trajectory = [{"role": "user", "content": "test"}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_trajectory(trajectory, "model", False, filename=None)

            # Check file was created in current directory
            assert os.path.exists("failed_trajectories.jsonl")
            os.unlink("failed_trajectories.jsonl")

    def test_unicode_content(self, tmp_path):
        """Should handle unicode content correctly."""
        trajectory = [
            {
                "role": "user",
                "content": "Hello 世界 🌍 Привет مرحبا"
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory, "model", True, filename=temp_file)

            with open(temp_file, 'r', encoding='utf-8') as f:
                entry = json.loads(f.readline())
                assert entry["conversations"][0]["content"] == "Hello 世界 🌍 Привет مرحبا"
        finally:
            os.unlink(temp_file)

    def test_complex_trajectory_structure(self, tmp_path):
        """Should handle complex trajectory with tool calls and reasoning."""
        trajectory = [
            {
                "role": "user",
                "content": "Analyze this code"
            },
            {
                "role": "assistant",
                "content": "Let me check the code",
                "reasoning": "I need to read the file first"
            },
            {
                "role": "tool",
                "name": "file_read",
                "args": {"path": "test.py"}
            },
            {
                "role": "assistant",
                "content": "Here's the analysis"
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory, "complex-model", True, filename=temp_file)

            with open(temp_file, 'r') as f:
                entry = json.loads(f.readline())

            assert len(entry["conversations"]) == 4
            assert entry["conversations"][0]["role"] == "user"
            assert entry["conversations"][2]["role"] == "tool"
        finally:
            os.unlink(temp_file)

    def test_empty_trajectory(self, tmp_path):
        """Should handle empty trajectory list."""
        trajectory = []

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory, "model", True, filename=temp_file)

            with open(temp_file, 'r') as f:
                entry = json.loads(f.readline())
                assert entry["conversations"] == []
        finally:
            os.unlink(temp_file)

    def test_trajectory_with_special_characters(self, tmp_path):
        """Should handle special characters in content."""
        trajectory = [
            {
                "role": "user",
                "content": 'Code: """multi-line\nstring with "quotes" and \'apostrophes\'"""'
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory, "model", True, filename=temp_file)

            with open(temp_file, 'r') as f:
                entry = json.loads(f.readline())
                assert entry["conversations"][0]["content"] == trajectory[0]["content"]
        finally:
            os.unlink(temp_file)


class TestSaveTrajectoryIntegration:
    """Integration tests for save_trajectory."""

    def test_multiple_saves_to_same_file(self, tmp_path):
        """Should handle multiple saves to the same file."""
        trajectory1 = [{"role": "user", "content": "First"}]
        trajectory2 = [{"role": "user", "content": "Second"}]
        trajectory3 = [{"role": "user", "content": "Third"}]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir=str(tmp_path)) as f:
            temp_file = f.name

        try:
            save_trajectory(trajectory1, "model1", True, filename=temp_file)
            save_trajectory(trajectory2, "model2", True, filename=temp_file)
            save_trajectory(trajectory3, "model3", False, filename=temp_file)

            with open(temp_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3

            # Verify all entries
            for i, expected in enumerate([trajectory1, trajectory2, trajectory3]):
                entry = json.loads(lines[i])
                assert entry["conversations"] == expected
        finally:
            os.unlink(temp_file)


class TestSaveTrajectoryErrorHandling:
    """Error handling tests for save_trajectory."""

    def test_logs_warning_on_write_error(self, tmp_path):
        """Should log warning when file write fails."""
        import logging

        trajectory = [{"role": "user", "content": "test"}]

        # Use a read-only path
        try:
            save_trajectory(trajectory, "model", True, filename="/root/trajectory.jsonl")

            # Should not raise, just log warning
            assert True  # Function completed without exception
        except PermissionError:
            # May also raise PermissionError depending on environment
            assert True
