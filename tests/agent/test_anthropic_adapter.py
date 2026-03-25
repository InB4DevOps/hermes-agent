"""Tests for anthropic_adapter module.

This module handles Anthropic API authentication, token resolution, and credential management.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.anthropic_adapter import (
    _supports_adaptive_thinking,
    _is_oauth_token,
    _detect_claude_code_version,
    _CLAUDE_CODE_VERSION_FALLBACK,
    build_anthropic_client,
    read_claude_code_credentials,
    read_claude_managed_key,
    is_claude_code_token_valid,
    _refresh_oauth_token,
    _write_claude_code_credentials,
    _resolve_claude_code_token_from_credentials,
    _prefer_refreshable_claude_code_token,
    get_anthropic_token_source,
    resolve_anthropic_token,
    run_oauth_setup_token,
    run_hermes_oauth_login,
)


class TestSupportsAdaptiveThinking:
    """Tests for _supports_adaptive_thinking function."""

    def test_supports_claude_4_6_models(self):
        """Should return True for Claude 4.6 models."""
        assert _supports_adaptive_thinking("anthropic/claude-4-6") is True
        assert _supports_adaptive_thinking("anthropic/claude-4.6") is True
        assert _supports_adaptive_thinking("anthropic/claude-opus-4.6") is True

    def test_does_not_support_old_models(self):
        """Should return False for older models."""
        assert _supports_adaptive_thinking("anthropic/claude-3-5-sonnet") is False
        assert _supports_adaptive_thinking("anthropic/claude-3-opus") is False
        assert _supports_adaptive_thinking("anthropic/claude-2.0") is False

    def test_supports_any_claude_4_variant(self):
        """Should return True for any model containing 4-6 or 4.6."""
        assert _supports_adaptive_thinking("anthropic/claude-4.0") is False
        assert _supports_adaptive_thinking("custom/claude-4.5") is False


class TestIsOauthToken:
    """Tests for _is_oauth_token function."""

    def test_regular_api_key(self):
        """Regular API keys should return False."""
        assert _is_oauth_token("sk-ant-api1234567890") is False
        assert _is_oauth_token("sk-ant-apikey") is False

    def test_oauth_setup_token(self):
        """OAuth setup tokens should return True."""
        assert _is_oauth_token("sk-ant-oat1234567890") is True
        assert _is_oauth_token("sk-ant-oat-token") is True

    def test_empty_token(self):
        """Empty token should return False."""
        assert _is_oauth_token("") is False
        assert _is_oauth_token(None) is False


class TestDetectClaudeCodeVersion:
    """Tests for _detect_claude_code_version function."""

    @patch("subprocess.run")
    def test_detects_version_from_claude_command(self, mock_run):
        """Should detect version when 'claude' command is available."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="2.1.74 (Claude Code)",
            stderr=""
        )
        result = _detect_claude_code_version()
        assert result == "2.1.74"
        mock_run.assert_called_once_with(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

    @patch("subprocess.run")
    def test_detects_version_from_claude_code_command(self, mock_run):
        """Should detect version when 'claude-code' command is available."""
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="not found"),  # 'claude' not found
            MagicMock(
                returncode=0,
                stdout="2.2.0",
                stderr=""
            )  # 'claude-code' found
        ]
        result = _detect_claude_code_version()
        assert result == "2.2.0"

    @patch("subprocess.run")
    def test_fallback_on_failure(self, mock_run):
        """Should return fallback version on all failures."""
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="not found"),
            Exception("command not found"),
        ]
        result = _detect_claude_code_version()
        assert result == _CLAUDE_CODE_VERSION_FALLBACK

    @patch("subprocess.run")
    def test_ignores_non_numeric_versions(self, mock_run):
        """Should ignore versions that don't start with a digit."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="beta-rc1 (Claude Code)",
            stderr=""
        )
        result = _detect_claude_code_version()
        assert result == _CLAUDE_CODE_VERSION_FALLBACK


class TestReadClaudeCodeCredentials:
    """Tests for read_claude_code_credentials function."""

    def test_reads_valid_credentials_file(self, tmp_path):
        """Should read valid credentials from file."""
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        # Claude Code credentials are nested under "claudeAiOauth" key
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "test-access-token",
                "refreshToken": "test-refresh-token",
                "expiresAt": int(time.time() * 1000) + 3600000
            }
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = read_claude_code_credentials()
            assert result is not None
            assert result["accessToken"] == "test-access-token"
            assert result["source"] == "claude_code_credentials_file"

    def test_returns_none_if_file_not_exists(self, tmp_path):
        """Should return None if credentials file doesn't exist."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = read_claude_code_credentials()
            assert result is None

    def test_returns_none_if_file_invalid_json(self, tmp_path):
        """Should return None if file contains invalid JSON."""
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text("not valid json {")

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = read_claude_code_credentials()
            assert result is None

    def test_returns_none_if_no_access_token(self, tmp_path):
        """Should return None if no access token in credentials."""
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "refreshToken": "test-refresh-token",
            "expiresAt": int(time.time() * 1000) + 3600000
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = read_claude_code_credentials()
            assert result is None


class TestReadClaudeManagedKey:
    """Tests for read_claude_managed_key function."""

    def test_reads_valid_managed_key(self, tmp_path):
        """Should read managed key from ~/.claude.json."""
        claude_file = tmp_path / ".claude.json"
        claude_file.write_text(json.dumps({
            "primaryApiKey": "managed-api-key-123"
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = read_claude_managed_key()
            assert result == "managed-api-key-123"

    def test_returns_none_if_file_not_exists(self, tmp_path):
        """Should return None if file doesn't exist."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = read_claude_managed_key()
            assert result is None

    def test_returns_none_if_no_key(self, tmp_path):
        """Should return None if no primaryApiKey."""
        claude_file = tmp_path / ".claude.json"
        claude_file.write_text(json.dumps({
            "someOtherKey": "value"
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = read_claude_managed_key()
            assert result is None


class TestIsClaudeCodeTokenValid:
    """Tests for is_claude_code_token_valid function."""

    def test_valid_token_with_expiration(self):
        """Token should be valid if not expired."""
        creds = {
            "accessToken": "test-token",
            "expiresAt": int(time.time() * 1000) + 3600000  # 1 hour from now
        }
        assert is_claude_code_token_valid(creds) is True

    def test_expired_token(self):
        """Token should be invalid if expired."""
        creds = {
            "accessToken": "test-token",
            "expiresAt": int(time.time() * 1000) - 3600000  # 1 hour ago
        }
        assert is_claude_code_token_valid(creds) is False

    def test_no_expiry_valid_if_token_present(self):
        """Token without expiry should be valid if present."""
        creds = {
            "accessToken": "test-token",
            "refreshToken": "test-refresh"
        }
        assert is_claude_code_token_valid(creds) is True

    def test_no_expiry_invalid_if_no_token(self):
        """Token without expiry and no access token should be invalid."""
        creds = {
            "refreshToken": "test-refresh"
        }
        assert is_claude_code_token_valid(creds) is False

    def test_no_expiry_invalid_if_no_token_or_expiry(self):
        """Empty creds should be invalid."""
        creds = {}
        assert is_claude_code_token_valid(creds) is False


class TestPreferRefreshableClaudeCodeToken:
    """Tests for _prefer_refreshable_claude_code_token function."""

    def test_prefers_refreshable_over_env_token(self):
        """Should prefer refreshable credentials over static env token."""
        creds = {
            "accessToken": "refreshable-token-from-file",
            "refreshToken": "refresh-token",
            "source": "claude_code_credentials_file"
        }
        env_token = "static-env-token"
        result = _prefer_refreshable_claude_code_token(env_token, creds)
        assert result == "refreshable-token-from-file"

    def test_returns_none_if_env_token_is_regular_api_key(self):
        """Should return None if env token is a regular API key."""
        creds = {
            "accessToken": "file-token",
            "refreshToken": "refresh-token"
        }
        env_token = "sk-ant-api123"
        result = _prefer_refreshable_claude_code_token(env_token, creds)
        assert result is None

    def test_returns_none_if_no_refresh_token(self):
        """Should return None if no refresh token available."""
        creds = {
            "accessToken": "file-token"
        }
        env_token = "oauth-token-from-env"
        result = _prefer_refreshable_claude_code_token(env_token, creds)
        assert result is None

    def test_returns_none_if_not_oauth_token(self):
        """Should return None if env token is not OAuth."""
        creds = {
            "accessToken": "file-token",
            "refreshToken": "refresh-token"
        }
        env_token="sk-ant-api-regular-key"
        result = _prefer_refreshable_claude_code_token(env_token, creds)
        assert result is None

    def test_returns_none_if_not_dict(self):
        """Should return None if creds is not a dict."""
        env_token = "oauth-token"
        creds = "not-a-dict"
        result = _prefer_refreshable_claude_code_token(env_token, creds)
        assert result is None


class TestGetAnthropicTokenSource:
    """Tests for get_anthropic_token_source function."""

    def test_source_from_env_token(self, tmp_path):
        """Should identify token from ANTHROPIC_TOKEN env var."""
        with patch.dict(os.environ, {"ANTHROPIC_TOKEN": "test-token"}):
            result = get_anthropic_token_source("test-token")
            assert result == "anthropic_token_env"

    def test_source_from_claude_code_oauth_env(self, tmp_path):
        """Should identify token from CLAUDE_CODE_OAUTH_TOKEN env var."""
        with patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "test-token"}):
            result = get_anthropic_token_source("test-token")
            assert result == "claude_code_oauth_token_env"

    def test_source_from_claude_credentials(self, tmp_path):
        """Should identify token from Claude Code credentials file."""
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        # Claude Code credentials are nested under "claudeAiOauth" key
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "test-token",
                "refreshToken": "refresh-test-token",
                "expiresAt": 9999999999
            }
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = get_anthropic_token_source("test-token")
            assert "claude_code" in result

    def test_source_from_claude_managed_key(self, tmp_path):
        """Should identify token from Claude managed key."""
        claude_file = tmp_path / ".claude.json"
        claude_file.write_text(json.dumps({
            "primaryApiKey": "test-token"
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = get_anthropic_token_source("test-token")
            assert result == "claude_json_primary_api_key"

    def test_source_from_api_key_env(self, tmp_path):
        """Should identify token from ANTHROPIC_API_KEY env var."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-token"}):
            result = get_anthropic_token_source("test-token")
            assert result == "anthropic_api_key_env"

    def test_source_none(self):
        """Should return 'none' for empty token."""
        result = get_anthropic_token_source(None)
        assert result == "none"


class TestResolveAnthropicToken:
    """Tests for resolve_anthropic_token function."""

    @patch.dict(os.environ, {"ANTHROPIC_TOKEN": "env-token"}, clear=False)
    @patch("os.getenv")
    @patch("agent.anthropic_adapter.read_claude_code_credentials")
    def test_priority_env_token(self, mock_read_creds, mock_getenv):
        """Should return env token when set."""
        mock_getenv.side_effect = lambda x, default: "env-token" if x == "ANTHROPIC_TOKEN" else default
        mock_read_creds.return_value = None
        result = resolve_anthropic_token()
        assert result == "env-token"

    @patch.dict(os.environ, clear=True)
    @patch("os.getenv")
    @patch("agent.anthropic_adapter.read_claude_code_credentials")
    def test_priority_claude_credentials(self, mock_read_creds, mock_getenv):
        """Should return token from Claude Code credentials."""
        mock_getenv.return_value = ""
        mock_read_creds.return_value = {
            "accessToken": "creds-token",
            "expiresAt": int(time.time() * 1000) + 3600000
        }
        result = resolve_anthropic_token()
        assert result == "creds-token"


class TestWriteClaudeCodeCredentials:
    """Tests for _write_claude_code_credentials function."""

    def test_writes_credentials_with_correct_permissions(self, tmp_path):
        """Should write credentials and set 600 permissions."""
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)

        with patch("pathlib.Path.home", return_value=tmp_path):
            _write_claude_code_credentials(
                "new-access-token",
                "new-refresh-token",
                int(time.time() * 1000) + 3600000
            )

        assert cred_file.exists()
        assert cred_file.stat().st_mode & 0o777 == 0o600

        data = json.loads(cred_file.read_text())
        assert data["claudeAiOauth"]["accessToken"] == "new-access-token"
        assert data["claudeAiOauth"]["refreshToken"] == "new-refresh-token"

    def test_preserves_existing_fields(self, tmp_path):
        """Should preserve existing fields in credentials file."""
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "someOtherField": "should-preserve",
            "anotherField": 123
        }))

        _write_claude_code_credentials(
            "new-access-token",
            "new-refresh-token",
            int(time.time() * 1000) + 3600000
        )

        data = json.loads(cred_file.read_text())
        assert data["someOtherField"] == "should-preserve"
        assert data["anotherField"] == 123


class TestRefreshOauthToken:
    """Tests for _refresh_oauth_token function."""

    @patch("urllib.request.urlopen")
    @patch("agent.anthropic_adapter._write_claude_code_credentials")
    def test_successful_refresh(self, mock_write, mock_urlopen):
        """Should successfully refresh token and update file."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = _refresh_oauth_token({
            "accessToken": "old-token",
            "refreshToken": "old-refresh-token"
        })

        assert result == "new-access-token"
        mock_write.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_refresh_without_refresh_token(self, mock_urlopen):
        """Should return None without refreshing if no refresh token."""
        result = _refresh_oauth_token({
            "accessToken": "old-token"
        })
        assert result is None
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_refresh_on_failure(self, mock_urlopen):
        """Should return None if refresh fails."""
        mock_urlopen.side_effect = Exception("Network error")
        result = _refresh_oauth_token({
            "accessToken": "old-token",
            "refreshToken": "old-refresh-token"
        })
        assert result is None
