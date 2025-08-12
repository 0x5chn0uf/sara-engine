"""Tests for CLI command error handling and behavior."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli import init_cmd, search_cmd


class TestCLIErrorHandling:
    """Test CLI commands handle errors gracefully."""

    def test_init_cmd_permission_error(self, capsys):
        """Test init command handles permission errors gracefully."""
        args = Mock()

        with patch("sara.cli.init_cmd.init_database") as mock_init:
            mock_init.side_effect = PermissionError("Permission denied")

            with pytest.raises(SystemExit) as exc_info:
                init_cmd.cmd_init(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "❌ Permission denied" in captured.out

    def test_init_cmd_keyboard_interrupt(self, capsys):
        """Test init command handles keyboard interrupt gracefully."""
        args = Mock()

        with patch("sara.cli.init_cmd.init_database") as mock_init:
            mock_init.side_effect = KeyboardInterrupt()

            with pytest.raises(SystemExit) as exc_info:
                init_cmd.cmd_init(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "🛑 Initialization cancelled by user" in captured.out

    def test_init_cmd_generic_error(self, capsys):
        """Test init command handles generic errors gracefully."""
        args = Mock()

        with patch("sara.cli.init_cmd.init_database") as mock_init:
            mock_init.side_effect = RuntimeError("Database error")

            with pytest.raises(SystemExit) as exc_info:
                init_cmd.cmd_init(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "❌ Initialization failed: Database error" in captured.out

    def test_search_cmd_import_error(self, capsys):
        """Test search command handles import errors gracefully."""
        args = Mock()
        args.query = "test query"
        args.limit = 10

        with patch("sara.cli.search_cmd.Memory") as mock_memory:
            mock_memory.side_effect = ImportError("sentence-transformers not found")

            with pytest.raises(SystemExit) as exc_info:
                search_cmd.cmd_search(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert (
                "❌ Missing dependency: sentence-transformers not found" in captured.out
            )

    def test_search_cmd_keyboard_interrupt(self, capsys):
        """Test search command handles keyboard interrupt gracefully."""
        args = Mock()
        args.query = "test query"
        args.limit = 10

        with patch("sara.cli.search_cmd.Memory") as mock_memory:
            mock_memory.side_effect = KeyboardInterrupt()

            with pytest.raises(SystemExit) as exc_info:
                search_cmd.cmd_search(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "🛑 Search cancelled by user" in captured.out

    def test_search_cmd_generic_error(self, capsys):
        """Test search command handles generic errors gracefully."""
        args = Mock()
        args.query = "test query"
        args.limit = 10

        with patch("sara.cli.search_cmd.Memory") as mock_memory:
            mock_memory.side_effect = RuntimeError("Database connection failed")

            with pytest.raises(SystemExit) as exc_info:
                search_cmd.cmd_search(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "❌ Search failed: Database connection failed" in captured.out

    def test_search_cmd_success_no_results(self, capsys):
        """Test search command handles no results gracefully."""
        args = Mock()
        args.query = "nonexistent query"
        args.limit = 10

        mock_memory_instance = Mock()
        mock_memory_instance.search.return_value = []

        with patch("sara.cli.search_cmd.Memory") as mock_memory:
            mock_memory.return_value = mock_memory_instance

            search_cmd.cmd_search(args)

            captured = capsys.readouterr()
            assert "🔍 Searching for: 'nonexistent query'" in captured.out
            assert "❌ No results found" in captured.out

    def test_search_cmd_success_with_results(self, capsys):
        """Test search command displays results correctly."""
        args = Mock()
        args.query = "test query"
        args.limit = 10

        # Mock search result
        mock_result = Mock()
        mock_result.task_id = "123"
        mock_result.title = "Test Task"
        mock_result.filepath = "/test/path.md"
        mock_result.kind = Mock()
        mock_result.kind.value = "archive"
        mock_result.score = 0.85
        mock_result.excerpt = "This is a test excerpt"

        mock_memory_instance = Mock()
        mock_memory_instance.search.return_value = [mock_result]

        with patch("sara.cli.search_cmd.Memory") as mock_memory:
            mock_memory.return_value = mock_memory_instance

            search_cmd.cmd_search(args)

            captured = capsys.readouterr()
            assert "🔍 Searching for: 'test query'" in captured.out
            assert "✅ Found 1 results:" in captured.out
            assert "[123] Test Task" in captured.out
            assert "📁 /test/path.md" in captured.out
            assert "🏷️ archive" in captured.out
            assert "📊 Score: 0.850" in captured.out
            assert "📝 This is a test excerpt" in captured.out


class TestCLIArgumentHandling:
    """Test CLI argument validation and edge cases."""

    def test_init_cmd_no_args_success(self):
        """Test init command works with minimal arguments."""
        args = Mock()

        with patch("sara.cli.init_cmd.init_database") as mock_init:
            mock_init.return_value = None

            # Should not raise any exception
            init_cmd.cmd_init(args)
            mock_init.assert_called_once()

    def test_search_cmd_empty_results_handling(self):
        """Test search handles empty results without errors."""
        args = Mock()
        args.query = ""
        args.limit = 0

        mock_memory_instance = Mock()
        mock_memory_instance.search.return_value = []

        with patch("sara.cli.search_cmd.Memory") as mock_memory:
            mock_memory.return_value = mock_memory_instance

            # Should handle empty query/limit gracefully
            search_cmd.cmd_search(args)
            mock_memory_instance.search.assert_called_once_with(query="", k=0)
