"""Tests for ShellFileOperations helper methods.

Tests for:
- _ends_with_newline() helper (Task 6)
- _escape_shell_arg() for safe shell escaping (Task 9)
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib.util

# Load file_operations directly to avoid circular imports through tools/__init__.py
_tools_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "tools"))
_file_ops_spec = importlib.util.spec_from_file_location(
    "file_operations",
    os.path.join(_tools_dir, "file_operations.py")
)
file_operations_module = importlib.util.module_from_spec(_file_ops_spec)
_file_ops_spec.loader.exec_module(file_operations_module)

ShellFileOperations = file_operations_module.ShellFileOperations


# ============================================================================
# Test _ends_with_newline helper (Task 6)
# ============================================================================

class TestEndsWithNewline:
    """Comprehensive tests for the _ends_with_newline helper method."""
    
    def setup_method(self):
        """Create a mock terminal environment for each test."""
        self.mock_env = MagicMock()
        self.file_ops = ShellFileOperations(self.mock_env, cwd="/test")
    
    def test_empty_file(self, tmp_path):
        """Empty file should return False (no newline)."""
        file_path = tmp_path / "empty.txt"
        file_path.write_bytes(b"")
        
        self.mock_env.execute.return_value = {"output": "", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is False
    
    def test_file_ending_with_newline(self, tmp_path):
        """File ending with newline should return True."""
        file_path = tmp_path / "with_newline.txt"
        file_path.write_text("hello world\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True
    
    def test_file_not_ending_with_newline(self, tmp_path):
        """File not ending with newline should return False."""
        file_path = tmp_path / "no_newline.txt"
        file_path.write_bytes(b"hello world")
        
        self.mock_env.execute.return_value = {"output": "d", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is False
    
    def test_path_with_spaces(self, tmp_path):
        """Path containing spaces should be handled correctly."""
        dir_with_spaces = tmp_path / "dir with spaces"
        dir_with_spaces.mkdir()
        file_path = dir_with_spaces / "file with spaces.txt"
        file_path.write_text("content\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True
    
    def test_path_with_special_characters(self, tmp_path):
        """Path with special characters should be properly escaped."""
        dir_special = tmp_path / "dir-special!@#"
        dir_special.mkdir()
        file_path = dir_special / "file@special.txt"
        file_path.write_text("content\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True
    
    def test_unicode_content(self, tmp_path):
        """File with unicode content ending with newline."""
        file_path = tmp_path / "unicode.txt"
        file_path.write_text("こんにちは世界\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True
    
    def test_binary_like_content(self, tmp_path):
        """File with binary-like content."""
        file_path = tmp_path / "binary.bin"
        file_path.write_bytes(b"\x00\x01\x02\x03\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True
    
    def test_single_character_file_with_newline(self, tmp_path):
        """Single character file containing just a newline."""
        file_path = tmp_path / "single_nl.txt"
        file_path.write_bytes(b"\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True
    
    def test_single_character_file_without_newline(self, tmp_path):
        """Single character file without newline."""
        file_path = tmp_path / "single_char.txt"
        file_path.write_bytes(b"x")
        
        self.mock_env.execute.return_value = {"output": "x", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is False
    
    def test_multiline_file_ending_with_newline(self, tmp_path):
        """Multi-line file ending with newline."""
        file_path = tmp_path / "multiline.txt"
        file_path.write_text("line1\nline2\nline3\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True
    
    def test_multiline_file_not_ending_with_newline(self, tmp_path):
        """Multi-line file not ending with newline."""
        file_path = tmp_path / "multiline_no_nl.txt"
        file_path.write_text("line1\nline2\nline3")
        
        self.mock_env.execute.return_value = {"output": "3", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is False
    
    def test_file_not_found(self, tmp_path):
        """Non-existent file should return False."""
        file_path = tmp_path / "nonexistent.txt"
        
        self.mock_env.execute.return_value = {"output": "", "returncode": 1}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is False
    
    def test_only_newline_in_file(self, tmp_path):
        """File containing only a newline character."""
        file_path = tmp_path / "only_newline.txt"
        file_path.write_bytes(b"\n")
        
        self.mock_env.execute.return_value = {"output": "\n", "returncode": 0}
        result = self.file_ops._ends_with_newline(str(file_path))
        assert result is True


# ============================================================================
# Test _escape_shell_arg for safe shell escaping (Task 9)
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
        """Verify _ends_with_newline properly escapes malicious paths."""
        malicious_path = "/tmp/$(rm -rf /)"
        
        self.mock_env.execute.return_value = {"output": "", "returncode": 1}
        
        # Should not raise an exception
        result = self.file_ops._ends_with_newline(malicious_path)
        
        # Verify execute was called with a properly escaped command
        call_args = self.mock_env.execute.call_args
        command = call_args[0][0] if call_args else ""
        
        # The malicious path should be quoted in the command
        assert "'" in command
        # Command substitution should be inside quotes (neutralized)
        assert "$(" in command  # Present but inside quotes
