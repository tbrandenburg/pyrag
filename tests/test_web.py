"""Tests for web interface error handling."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pyrag.web import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestIndexEndpointErrorHandling:
    """Test error handling in GET / endpoint."""

    def test_returns_sanitized_error_on_exception(self, client):
        """Test that GET / returns sanitized error message, not raw exception."""
        with patch("pyrag.web.discover_documents") as mock_discover:
            mock_discover.side_effect = ConnectionError(
                "HTTPSConnectionPool(host='localhost', port=5432): Connection refused"
            )

            response = client.get("/")

            assert response.status_code == 200
            assert "internal" not in response.text.lower()
            assert "connection" not in response.text.lower()
            assert "localhost" not in response.text.lower()
            assert "error" in response.text.lower() or "unable" in response.text.lower()


class TestIndexDocumentEndpointErrorHandling:
    """Test error handling in POST /index endpoint."""

    def test_returns_sanitized_error_for_value_error(self, client):
        """Test that POST /index returns sanitized error for invalid input."""
        from unittest.mock import MagicMock

        mock_rag = MagicMock()
        mock_rag.index.side_effect = ValueError("Unsupported filetype: .exe")
        client.app.state.rag_cache = {"test": mock_rag}

        response = client.post("/index", json={"path": "malicious.exe", "collection_name": "test"})

        assert response.status_code == 400
        assert "internal" not in response.json()["detail"].lower()
        assert "memory address" not in response.json()["detail"].lower()

    def test_returns_sanitized_error_for_connection_error(self, client):
        """Test that POST /index returns sanitized error for network failures."""
        from unittest.mock import MagicMock

        mock_rag = MagicMock()
        mock_rag.index.side_effect = ConnectionError(
            "HTTPSConnectionPool(host='example.com', port=443): Max retries exceeded "
            "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at "
            "0x7f1234567890>: Failed to establish a new connection'))"
        )
        client.app.state.rag_cache = {"test": mock_rag}

        response = client.post(
            "/index",
            json={"path": "https://example.com/unreachable.pdf", "collection_name": "test"},
        )

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "0x7f" not in detail
        assert "HTTPSConnectionPool" not in detail
        assert "urllib3" not in detail
        assert "internal error" in detail.lower()


class TestStatsEndpointErrorHandling:
    """Test error handling in GET /stats endpoint."""

    def test_returns_sanitized_error_on_exception(self, client):
        """Test that GET /stats returns sanitized error for internal errors."""
        from unittest.mock import MagicMock

        from pyrag.web import DEFAULT_COLLECTION_NAME

        mock_rag = MagicMock()
        mock_rag.discover.side_effect = RuntimeError(
            "HTTPSConnectionPool(host='db', port=19530): Connection refused to Milvus at 0x12345"
        )
        client.app.state.rag_cache = {DEFAULT_COLLECTION_NAME: mock_rag}

        response = client.get("/stats")

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "0x12345" not in detail
        assert "HTTPSConnectionPool" not in detail
        assert "internal error" in detail.lower()


class TestDiscoverDocumentsErrorHandling:
    """Test error handling in discover_documents function."""

    def test_raises_sanitized_http_exception(self):
        """Test that discover_documents raises sanitized HTTPException."""
        from unittest.mock import MagicMock

        from pyrag.web import discover_documents

        mock_rag = MagicMock()
        mock_rag.discover.side_effect = ConnectionError(
            "Connection pool at 0xABC123 to redis://localhost:6379 failed"
        )

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            discover_documents("test_collection", mock_rag)

        assert exc_info.value.status_code == 500
        detail = str(exc_info.value.detail)
        assert "0xABC123" not in detail
        assert "redis" not in detail.lower()
        assert "connection pool" not in detail.lower()
        assert "internal error" in detail.lower()
