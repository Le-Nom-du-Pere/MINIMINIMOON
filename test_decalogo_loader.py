import logging
from unittest.mock import patch

from decalogo_loader import (
    DECALOGO_INDUSTRIAL_TEMPLATE,
    get_decalogo_industrial,
    load_decalogo_industrial,
)


class TestDecalogoLoader:
    @staticmethod
    def test_load_decalogo_no_target_path(caplog):
        """Test loading without target path returns template and logs appropriately."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        result = load_decalogo_industrial(None)

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert (
            "loaded from in-memory template (no target path specified)" in caplog.text
        )

    @staticmethod
    def test_load_decalogo_successful_file_write(tmp_path, caplog):
        """Test successful file write and atomic rename."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        target_path = tmp_path / "decalogo.txt"

        result = load_decalogo_industrial(str(target_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert target_path.exists()
        assert (
            target_path.read_text(encoding="utf-8")
            == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        )
        assert f"loaded and written to file: {target_path}" in caplog.text

    @staticmethod
    def test_load_decalogo_permission_error_fallback(caplog):
        """Test fallback to in-memory template on permission error."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = PermissionError("Access denied")

            result = load_decalogo_industrial("/restricted/path/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "PermissionError" in caplog.text or "Access denied" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    @staticmethod
    def test_load_decalogo_io_error_fallback(caplog):
        """Test fallback to in-memory template on I/O error."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = IOError("Disk full")

            result = load_decalogo_industrial("/tmp/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    @staticmethod
    def test_load_decalogo_os_error_fallback(caplog):
        """Test fallback to in-memory template on OS error."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = OSError("No space left on device")

            result = load_decalogo_industrial("/tmp/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    @staticmethod
    def test_load_decalogo_unexpected_error_fallback(caplog):
        """Test fallback to in-memory template on unexpected error."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.side_effect = ValueError("Unexpected error")

            result = load_decalogo_industrial("/tmp/decalogo.txt")

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Unexpected error writing DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    @staticmethod
    def test_load_decalogo_rename_failure(tmp_path, caplog):
        """Test fallback when atomic rename fails."""
        target_path = tmp_path / "decalogo.txt"

        caplog.set_level(logging.INFO, logger="decalogo_loader")

        with patch("decalogo_loader.os.replace") as mock_replace:
            mock_replace.side_effect = PermissionError("Cannot rename file")

            result = load_decalogo_industrial(str(target_path))

            assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
            assert "Failed to write DECALOGO_INDUSTRIAL" in caplog.text
            assert "loaded from in-memory fallback template" in caplog.text

    @staticmethod
    def test_get_decalogo_convenience_function(tmp_path):
        """Test convenience function with caching."""
        cache_path = tmp_path / "cached_decalogo.txt"

        result = get_decalogo_industrial(str(cache_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert cache_path.exists()
        assert (
            cache_path.read_text(encoding="utf-8")
            == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        )

    @staticmethod
    def test_get_decalogo_no_cache(caplog):
        """Test convenience function without caching."""
        caplog.set_level(logging.INFO, logger="decalogo_loader")
        result = get_decalogo_industrial(None)

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert "no target path specified" in caplog.text

    @staticmethod
    def test_temp_file_cleanup_on_error(tmp_path):
        """Test that temporary files are cleaned up when rename fails."""
        target_path = tmp_path / "decalogo.txt"

        with patch("decalogo_loader.os.replace") as mock_replace:
            mock_replace.side_effect = PermissionError("Cannot rename")

            load_decalogo_industrial(str(target_path))

            # Check that no temporary files are left behind
            temp_files = list(tmp_path.glob(".*_tmp_*"))
            assert len(temp_files) == 0

    @staticmethod
    def test_directory_creation(tmp_path):
        """Test that parent directories are created when they don't exist."""
        nested_path = tmp_path / "nested" / "dirs" / "decalogo.txt"

        result = load_decalogo_industrial(str(nested_path))

        assert result == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert (
            nested_path.read_text(encoding="utf-8")
            == DECALOGO_INDUSTRIAL_TEMPLATE.strip()
        )
