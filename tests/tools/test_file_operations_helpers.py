"""Tests for ShellFileOperations helper methods.

Tests for:
- _ends_with_newline() helper (Task 6)
- _escape_shell_arg() for safe shell escaping (Task 9)
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Import file_operations directly without triggering tools/__init__.py
# This avoids importing optional dependencies like firecrawl
_sys_path = sys.path.copy()
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in _sys_path:
    _sys_path.insert(0, str(_project_root))

# Load the module directly
import importlib.util
_file_ops_path = _project_root / "tools" / "file_operations.py"
_file_ops_spec = importlib.util.spec_from_file_location("file_operations", _file_ops_path)
_file_ops_module = importlib.util.module_from_spec(_file_ops_spec)
_file_ops_spec.loader.exec_module(_file_ops_module)

ShellFileOperations = _file_ops_module.ShellFileOperations


# ============================================================================
# Test _ends_with_newline helper (Task 6)
# ============================================================================

class TestEndsWithNewline:
    """Comprehensive tests for the _ends_with_newline helper method.
    
    Note: The optimized version checks content in-memory rather than
    spawning a subprocess, so tests pass content strings directly.
    """
    
    def setup_method(self):
        """Create a mock terminal environment for each test."""
        self.mock_env = MagicMock()
        self.file_ops = ShellFileOperations(self.mock_env, cwd="/test")
    
    def test_empty_content(self):
        """Empty content should return False (no newline)."""
        result = self.file_ops._ends_with_newline("")
        assert result is False
    
    def test_content_ending_with_newline(self):
        """Content ending with newline should return True."""
        result = self.file_ops._ends_with_newline("hello world\n")
        assert result is True
    
    def test_content_not_ending_with_newline(self):
        """Content not ending with newline should return False."""
        result = self.file_ops._ends_with_newline("hello world")
        assert result is False
    
    def test_single_newline(self):
        """Single newline character should return True."""
        result = self.file_ops._ends_with_newline("\n")
        assert result is True
    
    def test_single_character_no_newline(self):
        """Single character without newline should return False."""
        result = self.file_ops._ends_with_newline("x")
        assert result is False
    
    def test_multiline_ending_with_newline(self):
        """Multiline content ending with newline should return True."""
        content = "line1\nline2\nline3\n"
        result = self.file_ops._ends_with_newline(content)
        assert result is True
    
    def test_multiline_not_ending_with_newline(self):
        """Multiline content not ending with newline should return False."""
        content = "line1\nline2\nline3"
        result = self.file_ops._ends_with_newline(content)
        assert result is False
    
    def test_only_newlines(self):
        """Content with only newlines should return True."""
        result = self.file_ops._ends_with_newline("\n\n\n")
        assert result is True
    
    def test_unicode_content_with_newline(self):
        """Unicode content ending with newline should return True."""
        result = self.file_ops._ends_with_newline("hello 世界\n")
        assert result is True
    
    def test_unicode_content_without_newline(self):
        """Unicode content without newline should return False."""
        result = self.file_ops._ends_with_newline("hello 世界")
        assert result is False
    
    def test_special_characters_with_newline(self):
        """Content with special characters ending with newline should return True."""
        result = self.file_ops._ends_with_newline("test $PATH && echo 'done'\n")
        assert result is True
    
    def test_special_characters_without_newline(self):
        """Content with special characters without newline should return False."""
        result = self.file_ops._ends_with_newline("test $PATH && echo 'done'")
        assert result is False
    
    def test_binary_like_content_with_newline(self):
        """Binary-like content ending with newline should return True."""
        result = self.file_ops._ends_with_newline("\x00\x01\x02\n")
        assert result is True
    
    def test_binary_like_content_without_newline(self):
        """Binary-like content without newline should return False."""
        result = self.file_ops._ends_with_newline("\x00\x01\x02")
        assert result is False


# ============================================================================
# Test _escape_shell_arg helper (Task 9)
# ============================================================================

class TestEscapeShellArg:
    """Tests for safe shell argument escaping."""
    
    def setup_method(self):
        """Create a mock terminal environment for each test."""
        self.mock_env = MagicMock()
        self.file_ops = ShellFileOperations(self.mock_env, cwd="/test")
    
    def test_simple_string(self):
        """Simple string without special characters."""
        result = self.file_ops._escape_shell_arg("hello")
        assert result == "'hello'"
    
    def test_string_with_spaces(self):
        """String with spaces should be quoted."""
        result = self.file_ops._escape_shell_arg("hello world")
        assert result == "'hello world'"
    
    def test_string_with_single_quotes(self):
        """String with single quotes should be properly escaped."""
        result = self.file_ops._escape_shell_arg("hello'world")
        # Should use the pattern: 'hello'\'\'\'world'
        assert "hello" in result and "world" in result
        # Verify it's valid shell by checking the pattern
        assert result.count("'") >= 3  # At least opening, closing, and escape quotes
    
    def test_malicious_path_with_command_substitution(self):
        """Malicious path with command substitution should be safely quoted."""
        malicious = "/tmp/$(rm -rf /)"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # The $ should be inside single quotes, preventing command substitution
        assert "$" in result
        # Single quotes prevent shell expansion
        assert result.startswith("'") and result.endswith("'")
    
    def test_malicious_path_with_backticks(self):
        """Malicious path with backticks should be safely quoted."""
        malicious = "/tmp/`rm -rf /`"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # Backticks should be inside single quotes
        assert "`" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_malicious_path_with_semicolon(self):
        """Malicious path with semicolon should be safely quoted."""
        malicious = "/tmp/file; rm -rf /"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # Semicolon should be inside single quotes
        assert ";" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_malicious_path_with_pipe(self):
        """Malicious path with pipe should be safely quoted."""
        malicious = "/tmp/file | cat /etc/passwd"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # Pipe should be inside single quotes
        assert "|" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_malicious_path_with_redirect(self):
        """Malicious path with redirect should be safely quoted."""
        malicious = "/tmp/file > /etc/passwd"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # Redirect should be inside single quotes
        assert ">" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_path_with_newline(self):
        """Path with newline character should be safely quoted."""
        malicious = "/tmp/file\nname.txt"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # Newline should be inside single quotes (literal)
        assert "\n" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_path_with_dollar_sign(self):
        """Path with dollar sign (variable expansion) should be safely quoted."""
        malicious = "/tmp/$HOME/malicious"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # Dollar sign should be inside single quotes (literal)
        assert "$HOME" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_path_with_ampersand(self):
        """Path with ampersand (background operator) should be safely quoted."""
        malicious = "/tmp/file & rm -rf /"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # Ampersand should be inside single quotes
        assert "&" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_empty_string(self):
        """Empty string should return empty quotes."""
        result = self.file_ops._escape_shell_arg("")
        assert result == "''"
    
    def test_only_quotes(self):
        """String containing only single quotes."""
        result = self.file_ops._escape_shell_arg("'''")
        # Should properly escape multiple quotes
        assert len(result) > 2  # More than just empty quotes
    
    def test_complex_malicious_path(self):
        """Complex malicious path with multiple attack vectors."""
        malicious = "/tmp/$(cat /etc/passwd); rm -rf / & `whoami` > /dev/null"
        result = self.file_ops._escape_shell_arg(malicious)
        
        # All special characters should be inside single quotes
        assert "$" in result
        assert ";" in result
        assert "&" in result
        assert "`" in result
        assert ">" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_path_with_backslash(self):
        """Path with backslash should be handled correctly."""
        path = "/tmp/file\\name.txt"
        result = self.file_ops._escape_shell_arg(path)
        
        # Backslash should be preserved inside single quotes
        assert "\\" in result
        assert result.startswith("'") and result.endswith("'")
    
    def test_unicode_path(self):
        """Unicode path should be properly quoted."""
        path = "/tmp/日本語/file.txt"
        result = self.file_ops._escape_shell_arg(path)
        
        assert "日本語" in result
        assert result.startswith("'") and result.endswith("'")


# ============================================================================
# Integration tests for shell command safety
# ============================================================================

class TestShellCommandSafety:
    """Integration tests verifying shell commands are properly escaped."""
    
    def setup_method(self):
        """Create a mock terminal environment."""
        self.mock_env = MagicMock()
        self.file_ops = ShellFileOperations(self.mock_env, cwd="/test")
    
    def test_read_file_uses_escaped_path(self, tmp_path):
        """Verify read_file properly escapes the path in shell command."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")
        
        # Mock execute to capture the command
        captured_command = []
        def capture_execute(cmd, **kwargs):
            captured_command.append(cmd)
            return {"output": "1|content\n", "returncode": 0}
        
        self.mock_env.execute.side_effect = capture_execute
        
        result = self.file_ops.read_file(str(file_path))
        
        # Verify a command was executed
        assert len(captured_command) > 0
        # The path should appear in the command, properly quoted
        cmd = captured_command[0]
        assert str(file_path) in cmd or self.file_ops._escape_shell_arg(str(file_path)) in cmd
    
    def test_malicious_path_in_ends_with_newline(self):
        """Verify _ends_with_newline handles malicious-looking content safely.
        
        The optimized version checks content in-memory rather than spawning
        a subprocess, so malicious path strings are simply treated as content
        to check for trailing newlines - no shell escaping needed.
        """
        malicious_content = "/tmp/$(rm -rf /)"
        
        # Should not raise an exception - just treats it as plain content
        result = self.file_ops._ends_with_newline(malicious_content)
        
        # No subprocess should be called (optimized version works in-memory)
        assert self.mock_env.execute.call_count == 0
        
        # Result should be False (no trailing newline)
        assert result is False
        
        # With trailing newline, should return True
        result_with_newline = self.file_ops._ends_with_newline(malicious_content + "\n")
        assert result_with_newline is True
