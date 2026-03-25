"""Tests for context_references module.

This module handles @ context reference parsing and expansion for file, folder,
git, and URL references.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from agent.context_references import (
    ContextReference,
    ContextReferenceResult,
    REFERENCE_PATTERN,
    parse_context_references,
    preprocess_context_references,
    preprocess_context_references_async,
    _expand_reference,
    _expand_file_reference,
    _expand_folder_reference,
    _expand_git_reference,
    _fetch_url_content,
    _default_url_fetcher,
    _resolve_path,
    _ensure_reference_path_allowed,
    _strip_trailing_punctuation,
    _remove_reference_tokens,
    _is_binary_file,
    _build_folder_listing,
    _iter_visible_entries,
    _rg_files,
    _file_metadata,
    _code_fence_language,
)


class TestContextReferenceDataclass:
    """Tests for ContextReference dataclass."""

    def test_simple_reference(self):
        """Should create reference with basic fields."""
        ref = ContextReference(
            raw="@diff",
            kind="diff",
            target="",
            start=0,
            end=4
        )
        assert ref.raw == "@diff"
        assert ref.kind == "diff"
        assert ref.target == ""


class TestParseContextReferences:
    """Tests for parse_context_references function."""

    def test_empty_message(self):
        """Should return empty list for empty message."""
        result = parse_context_references("")
        assert result == []

    def test_simple_diff_reference(self):
        """Should parse simple @diff reference."""
        message = "Show me @diff please"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].kind == "diff"
        assert refs[0].raw == "@diff"
        assert refs[0].target == ""

    def test_staged_diff_reference(self):
        """Should parse @staged reference."""
        message = "Show @staged changes"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].kind == "staged"

    def test_file_reference(self):
        """Should parse @file:path references."""
        message = "Read @file:src/main.py"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].kind == "file"
        assert refs[0].target == "src/main.py"

    def test_file_reference_with_line_range(self):
        """Should parse @file:path:line-line references."""
        message = "Read @file:src/main.py:10-20"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].line_start == 10
        assert refs[0].line_end == 20

    def test_file_reference_with_single_line(self):
        """Should parse @file:path:line references."""
        message = "Read @file:src/main.py:42"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].line_start == 42
        assert refs[0].line_end == 42

    def test_git_reference(self):
        """Should parse @git references."""
        message = "Show @git history"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].kind == "git"
        assert refs[0].target == "1"  # Default count

    def test_git_reference_with_count(self):
        """Should parse @git:N references."""
        message = "Show @git:5 history"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].target == "5"

    def test_url_reference(self):
        """Should parse @url: references."""
        message = "Check @url:https://example.com/docs"
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].kind == "url"
        assert refs[0].target == "https://example.com/docs"

    def test_multiple_references(self):
        """Should parse multiple references in one message."""
        message = "Show @diff and @staged and @file:test.py"
        refs = parse_context_references(message)
        assert len(refs) == 3

    def test_trailing_punctuation_stripped(self):
        """Should strip trailing punctuation from URL."""
        message = "Check @url:https://example.com/docs."
        refs = parse_context_references(message)
        assert len(refs) == 1
        assert refs[0].target == "https://example.com/docs"


class TestStripTrailingPunctuation:
    """Tests for _strip_trailing_punctuation function."""

    def test_no_trailing_punctuation(self):
        """Should return unchanged if no trailing punctuation."""
        assert _strip_trailing_punctuation("hello world") == "hello world"

    def test_single_trailing_punctuation(self):
        """Should strip single trailing punctuation."""
        assert _strip_trailing_punctuation("hello.") == "hello"
        assert _strip_trailing_punctuation("hello,") == "hello"
        assert _strip_trailing_punctuation("hello!") == "hello"

    def test_multiple_trailing_punctuation(self):
        """Should strip multiple trailing punctuation."""
        assert _strip_trailing_punctuation("hello...") == "hello"
        assert _strip_trailing_punctuation("hello,,,") == "hello"

    def test_balanced_parentheses(self):
        """Should preserve balanced parentheses."""
        assert _strip_trailing_punctuation("hello (world)") == "hello (world)"

    def test_unbalanced_parentheses(self):
        """Should strip unbalanced parentheses."""
        assert _strip_trailing_punctuation("hello (world)") == "hello (world)"
        assert _strip_trailing_punctuation("hello world)") == "hello world"
        assert _strip_trailing_punctuation("hello world)") == "hello world"


class TestRemoveReferenceTokens:
    """Tests for _remove_reference_tokens function."""

    def test_remove_single_reference(self):
        """Should remove reference and join surrounding text."""
        refs = [ContextReference(
            raw="@file:test.py",
            kind="file",
            target="test.py",
            start=5,
            end=18
        )]
        result = _remove_reference_tokens("Read @file:test.py carefully", refs)
        assert result == "Read carefully"

    def test_remove_multiple_references(self):
        """Should remove multiple references."""
        refs = [
            ContextReference(
                raw="@diff",
                kind="diff",
                target="",
                start=0,
                end=5
            ),
            ContextReference(
                raw="@file:test.py",
                kind="file",
                target="test.py",
                start=6,
                end=19
            )
        ]
        result = _remove_reference_tokens("@diff @file:test.py show me", refs)
        assert result == "show me"

    def test_no_references(self):
        """Should return unchanged if no references."""
        result = _remove_reference_tokens("plain text", [])
        assert result == "plain text"


class TestResolvePath:
    """Tests for _resolve_path function."""

    def test_relative_path_resolved_from_cwd(self):
        """Should resolve relative paths from cwd."""
        result = _resolve_path(Path("/tmp"), "file.txt")
        assert str(result).startswith("/tmp/file.txt")

    def test_absolute_path_used_as_is(self):
        """Should use absolute paths as-is."""
        result = _resolve_path(Path("/tmp"), "/absolute/path/file.txt")
        assert str(result) == "/absolute/path/file.txt"

    def test_expands_home_directory(self):
        """Should expand ~ to home directory."""
        with patch("os.path.expanduser", return_value="/home/user/file.txt"):
            result = _resolve_path(Path("/tmp"), "~/file.txt")
            assert str(result) == "/home/user/file.txt"

    def test_raises_value_error_if_outside_allowed_root(self):
        """Should raise ValueError if path outside allowed root."""
        with pytest.raises(ValueError, match="outside the allowed workspace"):
            _resolve_path(
                Path("/allowed"),
                "/forbidden/path/file.txt",
                allowed_root=Path("/allowed")
            )


class TestEnsureReferencePathAllowed:
    """Tests for _ensure_reference_path_allowed function."""

    @patch("os.getenv")
    def test_allows_regular_file(self, mock_getenv):
        """Should allow regular files."""
        mock_getenv.return_value = "/tmp/.hermes"
        result = _ensure_reference_path_allowed(Path("/tmp/test.txt"))
        assert result is None

    @patch("os.getenv")
    def test_blocks_ssh_keys(self, mock_getenv):
        """Should block SSH key files."""
        mock_getenv.return_value = "/tmp/.hermes"
        with pytest.raises(ValueError, match="sensitive credential"):
            _ensure_reference_path_allowed(Path("/home/hermes/.ssh/id_rsa"))

    @patch("os.getenv")
    def test_blocks_sensitive_home_files(self, mock_getenv):
        """Should block sensitive home files."""
        mock_getenv.return_value = "/tmp/.hermes"
        with pytest.raises(ValueError, match="sensitive credential"):
            _ensure_reference_path_allowed(Path("/home/hermes/.bashrc"))

    @patch("os.getenv")
    def test_blocks_hermes_env(self, mock_getenv):
        """Should block ~/.hermes/.env."""
        mock_getenv.return_value = "/tmp/.hermes"
        with pytest.raises(ValueError, match="sensitive credential"):
            _ensure_reference_path_allowed(Path("/tmp/.hermes/.env"))


class TestIsBinaryFile:
    """Tests for _is_binary_file function."""

    def test_text_file_not_binary(self, tmp_path):
        """Text files should not be detected as binary."""
        file = tmp_path / "test.py"
        file.write_text("print('hello')")
        assert _is_binary_file(file) is False

    def test_binary_file_detected(self, tmp_path):
        """Binary files should be detected."""
        file = tmp_path / "image.bin"
        file.write_bytes(b"\x00\x01\x02\x03")
        assert _is_binary_file(file) is True

    def text_file_with_null_bytes(self, tmp_path):
        """Text files with null bytes in content should still be detected as text."""
        file = tmp_path / "weird.txt"
        file.write_text("hello\x00world")
        assert _is_binary_file(file) is False

    def test_common_text_extensions(self, tmp_path):
        """Common text extensions should not be detected as binary."""
        for ext in [".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts"]:
            file = tmp_path / f"test{ext}"
            file.write_text("content")
            assert _is_binary_file(file) is False


class TestExpandFileReference:
    """Tests for _expand_file_reference function."""

    def test_expands_valid_file(self, tmp_path):
        """Should expand content of valid file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')\nprint('world')")

        with patch("agent.context_references.estimate_tokens_rough", return_value=10):
            warning, content = _expand_file_reference(
                ContextReference(
                    raw="@file:test.py",
                    kind="file",
                    target="test.py",
                    start=0,
                    end=10
                ),
                tmp_path
            )

        assert warning is None
        assert "test.py" in content
        assert "print('hello')" in content

    def test_file_not_found(self, tmp_path):
        """Should return error if file not found."""
        ref = ContextReference(
            raw="@file:nonexistent.py",
            kind="file",
            target="nonexistent.py",
            start=0,
            end=15
        )
        warning, content = _expand_file_reference(ref, tmp_path)
        assert "file not found" in warning
        assert content is None

    def test_path_not_a_file(self, tmp_path):
        """Should return error if path is directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        ref = ContextReference(
            raw="@file:subdir",
            kind="file",
            target="subdir",
            start=0,
            end=7
        )
        warning, content = _expand_file_reference(ref, tmp_path)
        assert "not a file" in warning
        assert content is None

    def test_binary_file(self, tmp_path):
        """Should return error for binary files."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02")
        ref = ContextReference(
            raw="@file:binary.bin",
            kind="file",
            target="binary.bin",
            start=0,
            end=12
        )
        warning, content = _expand_file_reference(ref, tmp_path)
        assert "binary files are not supported" in warning
        assert content is None

    def test_expands_line_range(self, tmp_path):
        """Should expand only specified line range."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        with patch("agent.context_references.estimate_tokens_rough", return_value=5):
            warning, content = _expand_file_reference(
                ContextReference(
                    raw="@file:test.py:2-4",
                    kind="file",
                    target="test.py",
                    start=0,
                    end=12,
                    line_start=2,
                    line_end=4
                ),
                tmp_path
            )

        assert "line2" in content
        assert "line3" in content
        assert "line4" in content
        assert "line1" not in content
        assert "line5" not in content


class TestExpandFolderReference:
    """Tests for _expand_folder_reference function."""

    def test_expands_valid_folder(self, tmp_path):
        """Should expand content of valid folder."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (subdir / "file2.py").write_text("print(1)")

        with patch("agent.context_references.estimate_tokens_rough", return_value=20):
            warning, content = _expand_folder_reference(
                ContextReference(
                    raw="@folder:subdir",
                    kind="folder",
                    target="subdir",
                    start=0,
                    end=11
                ),
                tmp_path
            )

        assert warning is None
        assert "subdir" in content
        assert "file1.txt" in content
        assert "file2.py" in content

    def test_folder_not_found(self, tmp_path):
        """Should return error if folder not found."""
        ref = ContextReference(
            raw="@folder:nonexistent",
            kind="folder",
            target="nonexistent",
            start=0,
            end=15
        )
        warning, content = _expand_folder_reference(ref, tmp_path)
        assert "folder not found" in warning
        assert content is None

    def test_path_not_a_folder(self, tmp_path):
        """Should return error if path is file."""
        file = tmp_path / "test.py"
        file.write_text("content")
        ref = ContextReference(
            raw="@folder:test.py",
            kind="folder",
            target="test.py",
            start=0,
            end=12
        )
        warning, content = _expand_folder_reference(ref, tmp_path)
        assert "not a folder" in warning
        assert content is None


class TestExpandGitReference:
    """Tests for _expand_git_reference function."""

    @patch("subprocess.run")
    def test_expands_git_diff(self, mock_run, tmp_path):
        """Should expand git diff output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n-old\n+new\n",
            stderr=""
        )

        ref = ContextReference(
            raw="@diff",
            kind="diff",
            target="",
            start=0,
            end=4
        )
        warning, content = _expand_git_reference(ref, tmp_path, ["diff"], "git diff")
        assert warning is None
        assert "git diff" in content
        assert "diff --git" in content

    @patch("subprocess.run")
    def test_handles_git_error(self, mock_run):
        """Should handle git command failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository"
        )

        ref = ContextReference(
            raw="@diff",
            kind="diff",
            target="",
            start=0,
            end=4
        )
        warning, content = _expand_git_reference(ref, tmp_path, ["diff"], "git diff")
        assert "git command failed" in warning
        assert content is None

    def test_no_output_shows_placeholder(self, tmp_path):
        """Should show placeholder for empty output."""
        mock_run = MagicMock(return_value=MagicMock(
            returncode=0,
            stdout="",
            stderr=""
        ))
        with patch("subprocess.run", mock_run):
            ref = ContextReference(
                raw="@git",
                kind="git",
                target="1",
                start=0,
                end=3
            )
            warning, content = _expand_git_reference(ref, tmp_path, ["log", "-1", "-p"], "git log -1 -p")
            assert "(no output)" in content


class TestFetchUrlContent:
    """Tests for _fetch_url_content function."""

    @patch("agent.context_references._default_url_fetcher")
    def test_success_with_async_fetcher(self, mock_fetcher):
        """Should handle async fetcher."""
        async def mock_async_fetcher(url):
            return "async content"
        
        mock_fetcher.side_effect = mock_async_fetcher

        result = asyncio.run(_fetch_url_content("https://example.com"))
        assert result == "async content"

    @patch("agent.context_references._default_url_fetcher")
    def test_success_with_sync_fetcher(self, mock_fetcher):
        """Should handle sync fetcher."""
        mock_fetcher.return_value = "sync content"

        result = asyncio.run(_fetch_url_content("https://example.com"))
        assert result == "sync content"

    @patch("agent.context_references._default_url_fetcher")
    def test_empty_content(self, mock_fetcher):
        """Should handle empty content."""
        mock_fetcher.return_value = ""
        result = asyncio.run(_fetch_url_content("https://example.com"))
        assert result == ""


class TestBuildFolderListing:
    """Tests for _build_folder_listing function."""

    def test_lists_files_with_metadata(self, tmp_path):
        """Should list files with size or line count."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("line1\nline2\nline3")

        listing = _build_folder_listing(subdir, tmp_path, limit=100)
        assert "subdir/" in listing
        assert "file.txt" in listing
        assert "3 lines" in listing

    def test_ellipsis_for_many_files(self, tmp_path):
        """Should show ... for truncated listing."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        for i in range(200):
            (subdir / f"file{i}.txt").write_text(f"content {i}")

        listing = _build_folder_listing(subdir, tmp_path, limit=200)
        assert "..." in listing


class TestRgFiles:
    """Tests for _rg_files function."""

    @patch("subprocess.run")
    def test_success_with_ripgrep(self, mock_run):
        """Should succeed when ripgrep is available."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test.py\nreadme.md\n",
            stderr=""
        )
        result = _rg_files(Path("/tmp/test"), Path("/tmp"), limit=100)
        assert len(result) == 2

    @patch("subprocess.run")
    def test_fallback_when_ripgrep_not_found(self, mock_run):
        """Should return None when ripgrep not found."""
        mock_run.side_effect = FileNotFoundError()
        result = _rg_files(Path("/tmp/test"), Path("/tmp"), limit=100)
        assert result is None

    @patch("subprocess.run")
    def test_fallback_when_ripgrep_fails(self, mock_run):
        """Should return None when ripgrep fails."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="no match"
        )
        result = _rg_files(Path("/tmp/test"), Path("/tmp"), limit=100)
        assert result is None


class TestFileMetadata:
    """Tests for _file_metadata function."""

    def test_text_file_metadata(self, tmp_path):
        """Should return line count for text files."""
        file = tmp_path / "test.py"
        file.write_text("line1\nline2\nline3")
        metadata = _file_metadata(file)
        assert "3 lines" in metadata

    def test_binary_file_metadata(self, tmp_path):
        """Should return size for binary files."""
        file = tmp_path / "binary.bin"
        file.write_bytes(b"0123456789" * 100)
        metadata = _file_metadata(file)
        assert "bytes" in metadata


class TestCodeFenceLanguage:
    """Tests for _code_fence_language function."""

    def test_common_extensions(self, tmp_path):
        """Should return language for common extensions."""
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".jsx": "jsx",
            ".json": "json",
            ".md": "markdown",
            ".sh": "bash",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".toml": "toml",
        }
        for ext, lang in mapping.items():
            file = tmp_path / f"test{ext}"
            assert _code_fence_language(file) == lang

    def test_unknown_extension(self, tmp_path):
        """Should return empty string for unknown extensions."""
        file = tmp_path / "unknown.xyz"
        assert _code_fence_language(file) == ""


class TestPreprocessContextReferences:
    """Tests for preprocess_context_references function."""

    def test_no_references(self, tmp_path):
        """Should return unchanged message with no references."""
        result = preprocess_context_references(
            "plain message without any references",
            cwd=tmp_path,
            context_length=10000
        )
        assert result.message == "plain message without any references"
        assert result.warnings == []
        assert result.blocked is False

    def test_single_file_reference(self, tmp_path):
        """Should expand single file reference."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        result = preprocess_context_references(
            "Show @file:test.py",
            cwd=tmp_path,
            context_length=10000
        )
        assert result.references
        assert len(result.warnings) == 0
        assert result.blocked is False
        assert "test.py" in result.message

    def test_folder_reference(self, tmp_path):
        """Should expand folder reference."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        result = preprocess_context_references(
            "Show @folder:subdir",
            cwd=tmp_path,
            context_length=10000
        )
        assert result.references
        assert len(result.warnings) == 0
        assert result.blocked is False

    def test_hard_limit_exceeded(self, tmp_path):
        """Should block when injected tokens exceed hard limit."""
        # Create many files to exceed token limit
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        for i in range(50):
            (subdir / f"file{i}.txt").write_text(f"{'x' * 100} content {i}\n")

        result = preprocess_context_references(
            "Show @folder:subdir",
            cwd=tmp_path,
            context_length=500  # Small context length
        )
        # Should trigger warning or block depending on tokens injected
        assert result.references
        assert result.warnings or result.blocked


class TestAsyncPreprocessContextReferences:
    """Tests for preprocess_context_references_async function."""

    def test_async_execution(self, tmp_path):
        """Should execute async version correctly."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        result = asyncio.run(preprocess_context_references_async(
            "Show @file:test.py",
            cwd=tmp_path,
            context_length=10000
        ))
        assert result.references
        assert result.blocked is False


class TestIntegration:
    """Integration tests for context reference processing."""

    def test_full_workflow(self, tmp_path):
        """Test end-to-end reference processing."""
        # Create test files
        (tmp_path / "main.py").write_text("def main():\n    print('hello')")
        (tmp_path / "utils.py").write_text("def helper():\n    return 42")

        # Process message with multiple references
        result = preprocess_context_references(
            "Show @file:main.py and @file:utils.py",
            cwd=tmp_path,
            context_length=10000
        )

        assert len(result.references) == 2
        assert result.blocked is False
        assert "main.py" in result.message
        assert "utils.py" in result.message
