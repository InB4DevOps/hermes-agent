"""Tests for tools/file_operations.py — deny list, result dataclasses, helpers."""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tools.file_operations import (
    _is_write_denied,
    WRITE_DENIED_PATHS,
    WRITE_DENIED_PREFIXES,
    ReadResult,
    WriteResult,
    PatchResult,
    SearchResult,
    SearchMatch,
    LintResult,
    ShellFileOperations,
    BINARY_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MAX_LINE_LENGTH,
)


# =========================================================================
# Write deny list
# =========================================================================

class TestIsWriteDenied:
    def test_ssh_authorized_keys_denied(self):
        path = os.path.join(str(Path.home()), ".ssh", "authorized_keys")
        assert _is_write_denied(path) is True

    def test_ssh_id_rsa_denied(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_rsa")
        assert _is_write_denied(path) is True

    def test_netrc_denied(self):
        path = os.path.join(str(Path.home()), ".netrc")
        assert _is_write_denied(path) is True

    def test_aws_prefix_denied(self):
        path = os.path.join(str(Path.home()), ".aws", "credentials")
        assert _is_write_denied(path) is True

    def test_kube_prefix_denied(self):
        path = os.path.join(str(Path.home()), ".kube", "config")
        assert _is_write_denied(path) is True

    def test_normal_file_allowed(self, tmp_path):
        path = str(tmp_path / "safe_file.txt")
        assert _is_write_denied(path) is False

    def test_project_file_allowed(self):
        assert _is_write_denied("/tmp/project/main.py") is False

    def test_tilde_expansion(self):
        assert _is_write_denied("~/.ssh/authorized_keys") is True



# =========================================================================
# Result dataclasses
# =========================================================================

class TestReadResult:
    def test_to_dict_omits_defaults(self):
        r = ReadResult()
        d = r.to_dict()
        assert "error" not in d    # None omitted
        assert "similar_files" not in d  # empty list omitted

    def test_to_dict_preserves_empty_content(self):
        """Empty file should still have content key in the dict."""
        r = ReadResult(content="", total_lines=0, file_size=0)
        d = r.to_dict()
        assert "content" in d
        assert d["content"] == ""
        assert d["total_lines"] == 0
        assert d["file_size"] == 0

    def test_to_dict_includes_values(self):
        r = ReadResult(content="hello", total_lines=10, file_size=50, truncated=True)
        d = r.to_dict()
        assert d["content"] == "hello"
        assert d["total_lines"] == 10
        assert d["truncated"] is True

    def test_binary_fields(self):
        r = ReadResult(is_binary=True, is_image=True, mime_type="image/png")
        d = r.to_dict()
        assert d["is_binary"] is True
        assert d["is_image"] is True
        assert d["mime_type"] == "image/png"


class TestWriteResult:
    def test_to_dict_omits_none(self):
        r = WriteResult(bytes_written=100)
        d = r.to_dict()
        assert d["bytes_written"] == 100
        assert "error" not in d
        assert "warning" not in d

    def test_to_dict_includes_error(self):
        r = WriteResult(error="Permission denied")
        d = r.to_dict()
        assert d["error"] == "Permission denied"


class TestPatchResult:
    def test_to_dict_success(self):
        r = PatchResult(success=True, diff="--- a\n+++ b", files_modified=["a.py"])
        d = r.to_dict()
        assert d["success"] is True
        assert d["diff"] == "--- a\n+++ b"
        assert d["files_modified"] == ["a.py"]

    def test_to_dict_error(self):
        r = PatchResult(error="File not found")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "File not found"


class TestSearchResult:
    def test_to_dict_with_matches(self):
        m = SearchMatch(path="a.py", line_number=10, content="hello")
        r = SearchResult(matches=[m], total_count=1)
        d = r.to_dict()
        assert d["total_count"] == 1
        assert len(d["matches"]) == 1
        assert d["matches"][0]["path"] == "a.py"

    def test_to_dict_empty(self):
        r = SearchResult()
        d = r.to_dict()
        assert d["total_count"] == 0
        assert "matches" not in d

    def test_to_dict_files_mode(self):
        r = SearchResult(files=["a.py", "b.py"], total_count=2)
        d = r.to_dict()
        assert d["files"] == ["a.py", "b.py"]

    def test_to_dict_count_mode(self):
        r = SearchResult(counts={"a.py": 3, "b.py": 1}, total_count=4)
        d = r.to_dict()
        assert d["counts"]["a.py"] == 3

    def test_truncated_flag(self):
        r = SearchResult(total_count=100, truncated=True)
        d = r.to_dict()
        assert d["truncated"] is True


class TestLintResult:
    def test_skipped(self):
        r = LintResult(skipped=True, message="No linter for .md files")
        d = r.to_dict()
        assert d["status"] == "skipped"
        assert d["message"] == "No linter for .md files"

    def test_success(self):
        r = LintResult(success=True, output="")
        d = r.to_dict()
        assert d["status"] == "ok"

    def test_error(self):
        r = LintResult(success=False, output="SyntaxError line 5")
        d = r.to_dict()
        assert d["status"] == "error"
        assert "SyntaxError" in d["output"]


# =========================================================================
# ShellFileOperations helpers
# =========================================================================

@pytest.fixture()
def mock_env():
    """Create a mock terminal environment."""
    env = MagicMock()
    env.cwd = "/tmp/test"
    env.execute.return_value = {"output": "", "returncode": 0}
    return env


@pytest.fixture()
def file_ops(mock_env):
    return ShellFileOperations(mock_env)


class TestShellFileOpsHelpers:
    def test_escape_shell_arg_simple(self, file_ops):
        assert file_ops._escape_shell_arg("hello") == "'hello'"

    def test_escape_shell_arg_with_quotes(self, file_ops):
        result = file_ops._escape_shell_arg("it's")
        assert "'" in result
        # Should be safely escaped
        assert result.count("'") >= 4  # wrapping + escaping

    def test_is_likely_binary_by_extension(self, file_ops):
        assert file_ops._is_likely_binary("photo.png") is True
        assert file_ops._is_likely_binary("data.db") is True
        assert file_ops._is_likely_binary("code.py") is False
        assert file_ops._is_likely_binary("readme.md") is False

    def test_is_likely_binary_by_content(self, file_ops):
        # High ratio of non-printable chars -> binary
        binary_content = "\x00\x01\x02\x03" * 250
        assert file_ops._is_likely_binary("unknown", binary_content) is True

        # Normal text -> not binary
        assert file_ops._is_likely_binary("unknown", "Hello world\nLine 2\n") is False

    def test_is_image(self, file_ops):
        assert file_ops._is_image("photo.png") is True
        assert file_ops._is_image("pic.jpg") is True
        assert file_ops._is_image("icon.ico") is True
        assert file_ops._is_image("data.pdf") is False
        assert file_ops._is_image("code.py") is False

    def test_add_line_numbers(self, file_ops):
        content = "line one\nline two\nline three"
        result = file_ops._add_line_numbers(content)
        assert "     1|line one" in result
        assert "     2|line two" in result
        assert "     3|line three" in result

    def test_add_line_numbers_with_offset(self, file_ops):
        content = "continued\nmore"
        result = file_ops._add_line_numbers(content, start_line=50)
        assert "    50|continued" in result
        assert "    51|more" in result

    def test_add_line_numbers_truncates_long_lines(self, file_ops):
        long_line = "x" * (MAX_LINE_LENGTH + 100)
        result = file_ops._add_line_numbers(long_line)
        assert "[truncated]" in result

    def test_unified_diff(self, file_ops):
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        diff = file_ops._unified_diff(old, new, "test.py")
        assert "-line2" in diff
        assert "+changed" in diff
        assert "test.py" in diff

    def test_cwd_from_env(self, mock_env):
        mock_env.cwd = "/custom/path"
        ops = ShellFileOperations(mock_env)
        assert ops.cwd == "/custom/path"

    def test_cwd_fallback_to_slash(self):
        env = MagicMock(spec=[])  # no cwd attribute
        ops = ShellFileOperations(env)
        assert ops.cwd == "/"


class TestSearchPathValidation:
    """Test that search() returns an error for non-existent paths."""

    def test_search_nonexistent_path_returns_error(self, mock_env):
        """search() should return an error when the path doesn't exist."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "not_found", "returncode": 1}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/nonexistent/path")
        assert result.error is not None
        assert "not found" in result.error.lower() or "Path not found" in result.error

    def test_search_nonexistent_path_files_mode(self, mock_env):
        """search(target='files') should also return error for bad paths."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "not_found", "returncode": 1}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("*.py", path="/nonexistent/path", target="files")
        assert result.error is not None
        assert "not found" in result.error.lower() or "Path not found" in result.error

    def test_search_existing_path_proceeds(self, mock_env):
        """search() should proceed normally when the path exists."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "exists", "returncode": 0}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            # rg returns exit 1 (no matches) with empty output
            return {"output": "", "returncode": 1}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/existing/path")
        assert result.error is None
        assert result.total_count == 0  # No matches but no error

    def test_search_rg_error_exit_code(self, mock_env):
        """search() should report error when rg returns exit code 2."""
        call_count = {"n": 0}
        def side_effect(command, **kwargs):
            call_count["n"] += 1
            if "test -e" in command:
                return {"output": "exists", "returncode": 0}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            # rg returns exit 2 (error) with empty output
            return {"output": "", "returncode": 2}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/some/path")
        assert result.error is not None
        assert "search failed" in result.error.lower() or "Search error" in result.error


class TestShellFileOpsWriteDenied:
    def test_write_file_denied_path(self, file_ops):
        result = file_ops.write_file("~/.ssh/authorized_keys", "evil key")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_patch_replace_denied_path(self, file_ops):
        result = file_ops.patch_replace("~/.ssh/authorized_keys", "old", "new")
        assert result.error is not None
        assert "denied" in result.error.lower()


# =========================================================================
# read_file line counting fix (wc -l undercounting)
# =========================================================================

class TestReadFileLineCounting:
    """Test that read_file correctly counts lines for files with/without trailing newlines.
    
    Regression test for the wc -l bug where files without trailing newlines
    were undercounted by 1, causing incorrect truncation detection.
    """

    @pytest.fixture()
    def file_ops_with_real_exec(self, tmp_path):
        """Create ShellFileOperations that actually executes shell commands."""
        class RealExecMock:
            """Mock that executes real shell commands for testing."""
            def __init__(self, cwd):
                self.cwd = cwd
            
            def execute(self, command, cwd=None):
                import subprocess
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    cwd=cwd or self.cwd
                )
                return {
                    "output": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                    "returncode": result.returncode
                }
        
        return ShellFileOperations(RealExecMock(str(tmp_path)))

    def _create_test_file(self, tmp_path, lines, trailing_newline=True):
        """Helper to create a test file with specified line count."""
        content = "\n".join([f"Line {i}" for i in range(1, lines + 1)])
        if trailing_newline:
            content += "\n"
        filepath = tmp_path / "test_file.txt"
        filepath.write_text(content)
        return str(filepath)

    def test_read_file_with_trailing_newline(self, tmp_path, file_ops_with_real_exec):
        """File with trailing newline should report correct line count."""
        filepath = self._create_test_file(tmp_path, 10, trailing_newline=True)
        result = file_ops_with_real_exec.read_file(filepath, offset=1, limit=5)
        
        assert result.total_lines == 10
        assert result.truncated is True  # Lines 1-5 of 10, more available
        assert result.error is None

    def test_read_file_without_trailing_newline(self, tmp_path, file_ops_with_real_exec):
        """File without trailing newline should report correct line count (not undercount)."""
        filepath = self._create_test_file(tmp_path, 10, trailing_newline=False)
        result = file_ops_with_real_exec.read_file(filepath, offset=1, limit=5)
        
        # This was the bug: wc -l reported 9 instead of 10
        assert result.total_lines == 10, f"Expected 10 lines, got {result.total_lines}"
        assert result.truncated is True  # Lines 1-5 of 10, more available
        assert result.error is None

    def test_read_file_without_newline_end_of_file(self, tmp_path, file_ops_with_real_exec):
        """Reading end of file without trailing newline should not report truncated."""
        filepath = self._create_test_file(tmp_path, 10, trailing_newline=False)
        # Request lines 6-10 (the last 5 lines)
        result = file_ops_with_real_exec.read_file(filepath, offset=6, limit=5)
        
        assert result.total_lines == 10
        assert result.truncated is False  # No more lines available
        assert result.error is None

    def test_read_file_501_lines_without_newline(self, tmp_path, file_ops_with_real_exec):
        """Large file without trailing newline should report correct count."""
        filepath = self._create_test_file(tmp_path, 501, trailing_newline=False)
        result = file_ops_with_real_exec.read_file(filepath, offset=1, limit=500)
        
        assert result.total_lines == 501, f"Expected 501 lines, got {result.total_lines}"
        assert result.truncated is True  # Lines 1-500 of 501, more available
        assert result.error is None

    def test_read_file_single_line_without_newline(self, tmp_path, file_ops_with_real_exec):
        """Single line file without newline should report 1 line (not 0)."""
        filepath = self._create_test_file(tmp_path, 1, trailing_newline=False)
        result = file_ops_with_real_exec.read_file(filepath, offset=1, limit=1)
        
        # This was the edge case: wc -l reported 0 for single line without newline
        assert result.total_lines == 1, f"Expected 1 line, got {result.total_lines}"
        assert result.truncated is False  # Only 1 line exists
        assert result.error is None

    def test_read_file_pagination_across_boundary(self, tmp_path, file_ops_with_real_exec):
        """Reading exactly at the boundary of a file without trailing newline."""
        filepath = self._create_test_file(tmp_path, 501, trailing_newline=False)
        # Request lines 500-501
        result = file_ops_with_real_exec.read_file(filepath, offset=500, limit=2)
        
        assert result.total_lines == 501
        assert result.truncated is False  # Lines 500-501 are the last lines
        assert result.error is None
        # Should have content for both lines
        assert "500|Line 500" in result.content
        assert "501|Line 501" in result.content

    def test_off_by_one_bug_regression(self, tmp_path, file_ops_with_real_exec):
        """Regression test for the original off-by-one bug.
        
        This test demonstrates that the old implementation (using only wc -l)
        would have undercounted lines for files without trailing newlines.
        
        The bug: wc -l counts newline characters, not lines. A file with
        content "Line 1\nLine 2\nLine 3" (no trailing newline) has 3 lines
        but only 2 newline characters, so wc -l returns 2.
        
        The fix: Check if the file ends with a newline using _ends_with_newline(),
        and add 1 to the wc -l count if it doesn't.
        """
        # Create a file without trailing newline
        filepath = self._create_test_file(tmp_path, 10, trailing_newline=False)
        
        # Verify the file actually doesn't end with newline
        raw_content = Path(filepath).read_text()
        assert not raw_content.endswith("\n"), "Test file should not end with newline"
        
        # Verify wc -l would undercount (simulate old behavior)
        import subprocess
        wc_result = subprocess.run(
            f"wc -l < {filepath}", shell=True, capture_output=True, text=True
        )
        wc_count = int(wc_result.stdout.strip())
        assert wc_count == 9, f"wc -l should report 9 (one less than actual lines), got {wc_count}"
        
        # Verify the new implementation returns the correct count
        result = file_ops_with_real_exec.read_file(filepath, offset=1, limit=5)
        assert result.total_lines == 10, f"Expected 10 lines (with fix), got {result.total_lines}"
        assert result.error is None
        
        # Verify truncated flag is correct (lines 1-5 of 10)
        assert result.truncated is True, "Should be truncated (5 of 10 lines read)"
        
        # Verify we can read the last line correctly
        last_chunk = file_ops_with_real_exec.read_file(filepath, offset=10, limit=1)
        assert last_chunk.total_lines == 10
        assert last_chunk.truncated is False, "Should not be truncated when reading last line"
        assert "10|Line 10" in last_chunk.content, "Should contain the last line"


# =========================================================================
# _ends_with_newline helper tests (Task 6)
# =========================================================================

class TestEndsWithNewline:
    """Test the _ends_with_newline helper for various edge cases."""

    @pytest.fixture()
    def file_ops_with_real_exec(self, tmp_path):
        """Create ShellFileOperations that actually executes shell commands."""
        class RealExecMock:
            """Mock that executes real shell commands for testing."""
            def __init__(self, cwd):
                self.cwd = cwd
            
            def execute(self, command, cwd=None):
                import subprocess
                result = subprocess.run(
                    command, shell=True, capture_output=True,
                    cwd=cwd or self.cwd  # Note: no text=True for binary safety
                )
                # Decode stdout, handling binary data
                try:
                    stdout = result.stdout.decode('utf-8', errors='replace')
                except Exception:
                    stdout = result.stdout.decode('latin-1')  # Fallback
                return {
                    "output": stdout,
                    "stderr": result.stderr.decode('utf-8', errors='replace') if result.stderr else "",
                    "exit_code": result.returncode,
                    "returncode": result.returncode
                }
        
        return ShellFileOperations(RealExecMock(str(tmp_path)))

    def test_empty_file(self, tmp_path, file_ops_with_real_exec):
        """Empty file should return False (no newline)."""
        filepath = tmp_path / "empty.txt"
        filepath.write_text("")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is False, "Empty file does not end with newline"

    def test_file_ending_with_newline(self, tmp_path, file_ops_with_real_exec):
        """File ending with newline should return True."""
        filepath = tmp_path / "with_newline.txt"
        filepath.write_text("Line 1\nLine 2\nLine 3\n")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is True, "File ending with \\n should return True"

    def test_file_not_ending_with_newline(self, tmp_path, file_ops_with_real_exec):
        """File not ending with newline should return False."""
        filepath = tmp_path / "no_newline.txt"
        filepath.write_text("Line 1\nLine 2\nLine 3")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is False, "File not ending with \\n should return False"

    def test_single_character_with_newline(self, tmp_path, file_ops_with_real_exec):
        """Single character file with newline should return True."""
        filepath = tmp_path / "single_with_nl.txt"
        filepath.write_text("a\n")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is True, "Single char with \\n should return True"

    def test_single_character_without_newline(self, tmp_path, file_ops_with_real_exec):
        """Single character file without newline should return False."""
        filepath = tmp_path / "single_no_nl.txt"
        filepath.write_text("a")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is False, "Single char without \\n should return False"

    def test_only_newline(self, tmp_path, file_ops_with_real_exec):
        """File containing only a newline should return True."""
        filepath = tmp_path / "only_newline.txt"
        filepath.write_text("\n")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is True, "File with only \\n should return True"

    def test_path_with_spaces(self, tmp_path, file_ops_with_real_exec):
        """File with spaces in path should be handled correctly."""
        # Create a subdirectory with spaces
        subdir = tmp_path / "dir with spaces"
        subdir.mkdir()
        filepath = subdir / "file with spaces.txt"
        filepath.write_text("content\n")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is True, "File with spaces in path ending with \\n should return True"

    def test_path_with_special_characters(self, tmp_path, file_ops_with_real_exec):
        """File with special characters in path should be handled correctly."""
        # Create a subdirectory with special characters
        subdir = tmp_path / "dir-special!@#"
        subdir.mkdir()
        filepath = subdir / "file@special.txt"
        filepath.write_text("content")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is False, "File with special chars in path not ending with \\n should return False"

    def test_multiline_file_with_newline(self, tmp_path, file_ops_with_real_exec):
        """Multi-line file ending with newline should return True."""
        filepath = tmp_path / "multiline.txt"
        filepath.write_text("\n".join([f"Line {i}" for i in range(1, 101)]) + "\n")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is True, "100-line file ending with \\n should return True"

    def test_multiline_file_without_newline(self, tmp_path, file_ops_with_real_exec):
        """Multi-line file not ending with newline should return False."""
        filepath = tmp_path / "multiline_no_nl.txt"
        filepath.write_text("\n".join([f"Line {i}" for i in range(1, 101)]))
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is False, "100-line file not ending with \\n should return False"

    def test_binary_like_content_with_newline(self, tmp_path, file_ops_with_real_exec):
        """File with binary-like content ending with newline."""
        filepath = tmp_path / "binary_like.bin"
        filepath.write_bytes(b"\x00\x01\x02\x03\n")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is True, "Binary file ending with \\n should return True"

    def test_unicode_content_with_newline(self, tmp_path, file_ops_with_real_exec):
        """File with unicode content ending with newline."""
        filepath = tmp_path / "unicode.txt"
        filepath.write_text("Hello 世界！\n🎉🚀\n")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is True, "Unicode file ending with \\n should return True"

    def test_unicode_content_without_newline(self, tmp_path, file_ops_with_real_exec):
        """File with unicode content not ending with newline."""
        filepath = tmp_path / "unicode_no_nl.txt"
        filepath.write_text("Hello 世界！\n🎉🚀")
        result = file_ops_with_real_exec._ends_with_newline(str(filepath))
        assert result is False, "Unicode file not ending with \\n should return False"


# =========================================================================
# Threshold constants and investigation mode tests (Tasks 4 & 5)
# =========================================================================

class TestThresholdConstants:
    """Test threshold constants and their environment variable overrides."""

    def test_default_block_threshold(self):
        """Default block threshold should be 4."""
        from tools.file_tools import DEFAULT_BLOCK_THRESHOLD
        assert DEFAULT_BLOCK_THRESHOLD == 4

    def test_default_warn_threshold(self):
        """Default warn threshold should be 3."""
        from tools.file_tools import DEFAULT_WARN_THRESHOLD
        assert DEFAULT_WARN_THRESHOLD == 3

    def test_investigation_block_threshold(self):
        """Investigation mode block threshold should be 2."""
        from tools.file_tools import INVESTIGATION_BLOCK_THRESHOLD
        assert INVESTIGATION_BLOCK_THRESHOLD == 2

    def test_investigation_warn_threshold(self):
        """Investigation mode warn threshold should be 2."""
        from tools.file_tools import INVESTIGATION_WARN_THRESHOLD
        assert INVESTIGATION_WARN_THRESHOLD == 2


class TestInvestigationMode:
    """Test per-session investigation mode functionality."""

    def test_set_investigation_mode_enabled(self):
        """Setting investigation mode should enable it for the task."""
        from tools.file_tools import set_investigation_mode, get_investigation_mode, clear_read_tracker
        
        task_id = "test_task_enabled"
        set_investigation_mode(task_id, enabled=True)
        assert get_investigation_mode(task_id) is True
        
        clear_read_tracker(task_id)

    def test_set_investigation_mode_disabled(self):
        """Disabling investigation mode should set it to False."""
        from tools.file_tools import set_investigation_mode, get_investigation_mode, clear_read_tracker
        
        task_id = "test_task_disabled"
        set_investigation_mode(task_id, enabled=False)
        assert get_investigation_mode(task_id) is False
        
        clear_read_tracker(task_id)

    def test_get_investigation_mode_default(self):
        """Getting investigation mode for non-existent task should return False."""
        from tools.file_tools import get_investigation_mode
        
        task_id = "non_existent_task_12345"
        assert get_investigation_mode(task_id) is False

    def test_investigation_mode_per_session_isolation(self):
        """Investigation mode should be isolated per task_id."""
        from tools.file_tools import set_investigation_mode, get_investigation_mode, clear_read_tracker
        
        task_id_1 = "session_one"
        task_id_2 = "session_two"
        
        set_investigation_mode(task_id_1, enabled=True)
        set_investigation_mode(task_id_2, enabled=False)
        
        assert get_investigation_mode(task_id_1) is True
        assert get_investigation_mode(task_id_2) is False
        
        clear_read_tracker(task_id_1)
        clear_read_tracker(task_id_2)

    def test_investigation_mode_toggle(self):
        """Investigation mode should be toggleable."""
        from tools.file_tools import set_investigation_mode, get_investigation_mode, clear_read_tracker
        
        task_id = "toggle_test"
        
        set_investigation_mode(task_id, enabled=True)
        assert get_investigation_mode(task_id) is True
        
        set_investigation_mode(task_id, enabled=False)
        assert get_investigation_mode(task_id) is False
        
        set_investigation_mode(task_id, enabled=True)
        assert get_investigation_mode(task_id) is True
        
        clear_read_tracker(task_id)

    def test_investigation_mode_initializes_tracker(self):
        """Setting investigation mode should initialize the tracker entry."""
        from tools.file_tools import set_investigation_mode, _read_tracker, _read_tracker_lock, clear_read_tracker
        
        task_id = "init_tracker_test"
        
        # Ensure task doesn't exist yet
        with _read_tracker_lock:
            assert task_id not in _read_tracker
        
        set_investigation_mode(task_id, enabled=True)
        
        # Now it should exist with proper structure
        with _read_tracker_lock:
            assert task_id in _read_tracker
            assert _read_tracker[task_id]["investigation_mode"] is True
            assert _read_tracker[task_id]["last_key"] is None
            assert _read_tracker[task_id]["consecutive"] == 0
            assert isinstance(_read_tracker[task_id]["read_history"], set)
        
        clear_read_tracker(task_id)
