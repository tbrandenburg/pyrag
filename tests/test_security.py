"""Security tests for path validation."""

import pytest

from pyrag.utils import PathValidationError, get_supported_files, validate_path_security


class TestPathValidation:
    """Test path security validation."""

    def test_rejects_file_access_when_no_allowed_paths(self, monkeypatch):
        """Local files should be rejected when ALLOWED_BASE_PATHS is empty."""
        monkeypatch.setattr("pyrag.utils.ALLOWED_BASE_PATHS", [])
        with pytest.raises(PathValidationError, match="Local file access is disabled"):
            validate_path_security("/etc/passwd")

    def test_rejects_path_traversal(self, tmp_path, monkeypatch):
        """Path traversal attempts should be rejected."""
        monkeypatch.setattr("pyrag.utils.ALLOWED_BASE_PATHS", [str(tmp_path)])
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        with pytest.raises(PathValidationError, match="not within allowed"):
            validate_path_security(str(safe_dir.parent / ".." / "etc" / "passwd"))

    def test_rejects_file_scheme_urls(self):
        """file:// URLs should be rejected."""
        with pytest.raises(PathValidationError, match="scheme 'file' is not allowed"):
            validate_path_security("file:///etc/passwd")

    def test_rejects_internal_network_urls(self):
        """Internal network URLs should be rejected."""
        with pytest.raises(PathValidationError, match="cloud metadata endpoint"):
            validate_path_security("http://169.254.169.254/latest/meta-data/")

    def test_accepts_http_https_urls(self):
        """http:// and https:// URLs should be accepted."""
        validate_path_security("https://example.com/document.pdf")
        validate_path_security("http://example.com/document.pdf")

    def test_accepts_paths_within_allowed_base(self, tmp_path, monkeypatch):
        """Paths within allowed base directories should be accepted."""
        monkeypatch.setattr("pyrag.utils.ALLOWED_BASE_PATHS", [str(tmp_path)])
        test_file = tmp_path / "test.pdf"
        test_file.touch()
        validate_path_security(str(test_file))


class TestGetSupportedFilesIntegration:
    """Test get_supported_files with security validation."""

    def test_url_allowed_by_default(self):
        """URLs should work without ALLOWED_BASE_PATHS configured."""
        result = get_supported_files("https://example.com/test.pdf")
        assert result == ["https://example.com/test.pdf"]

    def test_local_file_rejected_without_config(self, monkeypatch):
        """Local files should be rejected without ALLOWED_BASE_PATHS."""
        monkeypatch.setattr("pyrag.utils.ALLOWED_BASE_PATHS", [])
        with pytest.raises(PathValidationError):
            get_supported_files("/etc/passwd")
