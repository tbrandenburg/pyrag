# Investigation: Internal Python exceptions leaked to users via POST /index error responses

**Issue**: #17 (https://github.com/tbrandenburg/pyrag/issues/17)
**Type**: BUG
**Investigated**: 2026-03-27T00:00:00Z

### Assessment

| Metric     | Value                         | Reasoning                                                                                                    |
| ---------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Severity   | HIGH                          | Information disclosure vulnerability exposing internal library details, memory addresses, and network topology |
| Complexity | LOW                           | Only 1-2 files affected (web.py), isolated change with minimal risk, clear fix location                     |
| Confidence | HIGH                          | Clear root cause identified at specific lines 165, 176, and 210 in web.py; evidence from source code       |

---

## Problem Statement

When `POST /index` fails (network error, unsupported format, invalid URL), the raw Python exception message—including full stack traces, internal object memory addresses, and library internals—is returned to the client. This exposes sensitive infrastructure details that aid attacker reconnaissance.

---

## Analysis

### Root Cause / Change Rationale

The root cause is twofold:

1. **Lines 175-176 in `web.py`**: The `index_document` endpoint catches all exceptions and passes `str(e)` directly to `HTTPException`, which exposes urllib3/requests internal details (memory addresses, connection pools, etc.)

2. **Line 165 in `web.py`**: The `index` endpoint returns raw exception strings directly in HTML via f-string, leaking exceptions to browsers without sanitization

### Evidence Chain

WHY: Client receives internal exception details
↓ BECAUSE: `str(e)` is passed directly to HTTP responses
Evidence: `src/pyrag/web.py:176` - `raise HTTPException(status_code=500, detail=str(e)) from e`

↓ BECAUSE: No sanitization of exception messages before returning to client
Evidence: `src/pyrag/web.py:175` - `except Exception as e:` (catches all exceptions)

↓ ROOT CAUSE: Raw exception output includes library internals (urllib3 connection pools, memory addresses)
Evidence: `src/pyrag/web.py:176` - `str(e)` on urllib3 exceptions exposes `HTTPSConnectionPool(host='...', port=443): Max retries exceeded...`

### Affected Files

| File                      | Lines | Action | Description                                       |
| ------------------------- | ----- | ------ | ------------------------------------------------- |
| `src/pyrag/web.py`        | 1-221 | UPDATE | Add logging, sanitize error messages              |
| `tests/test_web.py`       | NEW   | CREATE | Add tests for error handling                      |

### Integration Points

- `POST /index` endpoint (lines 168-176) - directly affected
- `GET /` endpoint (lines 141-165) - directly affected
- `discover_documents()` function (lines 179-210) - raises HTTPException with raw error
- Client UI receives these error responses

### Git History

- **Introduced**: `c38fcd4` - "feat: Added agentic solution team :)"
- **Last modified**: `c38fcd4` - same commit (single introduction)
- **Implication**: Original implementation did not consider error sanitization

---

## Implementation Plan

### Step 1: Add logging import and logger instance to web.py

**File**: `src/pyrag/web.py`
**Lines**: 1-22
**Action**: UPDATE

**Current code:**

```python
"""Web interface for PyRAG using FastAPI."""

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .rag import RAG
```

**Required change:**

```python
"""Web interface for PyRAG using FastAPI."""

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .rag import RAG

logger = logging.getLogger(__name__)
```

**Why**: Enable server-side logging of full exceptions while returning sanitized messages to clients.

---

### Step 2: Fix error handling in POST /index endpoint

**File**: `src/pyrag/web.py`
**Lines**: 168-176
**Action**: UPDATE

**Current code:**

```python
@app.post("/index")
def index_document(request: IndexRequest):
    """Index a document path or URL and return completion status."""
    try:
        rag = RAG(collection_name=request.collection_name)
        rag.index(request.path)
        return {"status": "finished"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Required change:**

```python
@app.post("/index")
def index_document(request: IndexRequest):
    """Index a document path or URL and return completion status."""
    try:
        rag = RAG(collection_name=request.collection_name)
        rag.index(request.path)
        return {"status": "finished"}
    except ValueError as e:
        logger.warning("Invalid input for document indexing: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid input provided. Please check the file path or URL.")
    except Exception as e:
        logger.exception("Error during document indexing: %s", str(e))
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request. Please try again later.")
```

**Why**: Specific handling for user input errors (400) vs internal errors (500), with full logging server-side and sanitized messages to clients.

---

### Step 3: Fix error handling in GET / endpoint

**File**: `src/pyrag/web.py`
**Lines**: 141-165
**Action**: UPDATE

**Current code:**

```python
@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    collection_name: str = Query(DEFAULT_COLLECTION_NAME, description="Milvus collection name"),
):
    """Root endpoint showing discovery of all indexed documents."""
    try:
        data = discover_documents(collection_name)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "collection_name": data.collection_name,
                "total_sources": data.total_sources,
                "total_chunks": data.total_chunks,
                "content_types": data.content_types,
                "sources": data.sources,
                "filenames": data.filenames,
                "total_content_length": data.total_content_length,
                "source_groups": data.source_groups,
            },
        )
    except Exception as e:
        # Fallback to error template or simple HTML
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
```

**Required change:**

```python
@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    collection_name: str = Query(DEFAULT_COLLECTION_NAME, description="Milvus collection name"),
):
    """Root endpoint showing discovery of all indexed documents."""
    try:
        data = discover_documents(collection_name)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "collection_name": data.collection_name,
                "total_sources": data.total_sources,
                "total_chunks": data.total_chunks,
                "content_types": data.content_types,
                "sources": data.sources,
                "filenames": data.filenames,
                "total_content_length": data.total_content_length,
                "source_groups": data.source_groups,
            },
        )
    except Exception as e:
        logger.exception("Error loading index page: %s", str(e))
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "collection_name": collection_name,
                "total_sources": 0,
                "total_chunks": 0,
                "content_types": {},
                "sources": [],
                "filenames": [],
                "total_content_length": 0,
                "source_groups": [],
                "error": "Unable to load documents. Please try again later.",
            },
        )
```

**Why**: Return proper HTML template with sanitized error message instead of raw f-string, ensuring consistent UI and no information leakage.

---

### Step 4: Fix error handling in discover_documents function

**File**: `src/pyrag/web.py`
**Lines**: 179-210
**Action**: UPDATE

**Current code:**

```python
def discover_documents(collection_name: str) -> DiscoveryResponse:
    """Discover and analyze all indexed documents in the collection."""
    try:
        rag = RAG(collection_name=collection_name)
        documents = rag.discover()

        if not documents:
            return DiscoveryResponse(
                collection_name=collection_name,
                total_sources=0,
                total_chunks=0,
                content_types={},
                sources=[],
                filenames=[],
                total_content_length=0,
                source_groups=[],
            )

        analysis = _analyze_documents(documents)

        return DiscoveryResponse(
            collection_name=collection_name,
            total_sources=analysis["total_sources"],
            total_chunks=analysis["total_chunks"],
            content_types=analysis["content_types"],
            sources=analysis["sources"],
            filenames=analysis["filenames"],
            total_content_length=analysis["total_content_length"],
            source_groups=analysis["source_groups"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Required change:**

```python
def discover_documents(collection_name: str) -> DiscoveryResponse:
    """Discover and analyze all indexed documents in the collection."""
    try:
        rag = RAG(collection_name=collection_name)
        documents = rag.discover()

        if not documents:
            return DiscoveryResponse(
                collection_name=collection_name,
                total_sources=0,
                total_chunks=0,
                content_types={},
                sources=[],
                filenames=[],
                total_content_length=0,
                source_groups=[],
            )

        analysis = _analyze_documents(documents)

        return DiscoveryResponse(
            collection_name=collection_name,
            total_sources=analysis["total_sources"],
            total_chunks=analysis["total_chunks"],
            content_types=analysis["content_types"],
            sources=analysis["sources"],
            filenames=analysis["filenames"],
            total_content_length=analysis["total_content_length"],
            source_groups=analysis["source_groups"],
        )
    except Exception as e:
        logger.exception("Error discovering documents: %s", str(e))
        raise HTTPException(status_code=500, detail="An internal error occurred while discovering documents.") from e
```

**Why**: Consistent sanitized error handling across all endpoints with server-side logging.

---

### Step 5: Add error handling to index.html template

**File**: `src/pyrag/web.py/templates/index.html`
**Lines**: NEW or existing
**Action**: UPDATE (if error variable used)

**Note**: The template should display the `error` variable if present. If not already in the template, add:

```html
{% if error %}
<div class="error-banner">{{ error }}</div>
{% endif %}
```

---

### Step 6: Add Tests

**File**: `tests/test_web.py`
**Action**: CREATE

**Test cases to add:**

```python
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
            mock_discover.side_effect = ConnectionError("HTTPSConnectionPool(host='localhost', port=5432): Connection refused")

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
        with patch("pyrag.rag.RAG") as mock_rag:
            mock_rag.return_value.index.side_effect = ValueError("Unsupported filetype: .exe")

            response = client.post(
                "/index",
                json={"path": "malicious.exe", "collection_name": "test"}
            )

            assert response.status_code == 400
            assert "internal" not in response.json()["detail"].lower()
            assert "memory address" not in response.json()["detail"].lower()

    def test_returns_sanitized_error_for_connection_error(self, client):
        """Test that POST /index returns sanitized error for network failures."""
        with patch("pyrag.rag.RAG") as mock_rag:
            mock_rag.return_value.index.side_effect = ConnectionError(
                "HTTPSConnectionPool(host='example.com', port=443): Max retries exceeded "
                "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at "
                "0x7f1234567890>: Failed to establish a new connection'))"
            )

            response = client.post(
                "/index",
                json={"path": "https://example.com/unreachable.pdf", "collection_name": "test"}
            )

            assert response.status_code == 500
            detail = response.json()["detail"]
            assert "0x7f" not in detail
            assert "HTTPSConnectionPool" not in detail
            assert "urllib3" not in detail
            assert "internal error" in detail.lower()
```

---

## Patterns to Follow

**From codebase - mirror these exactly:**

```python
# SOURCE: src/pyrag/rag.py (similar error handling approach)
# Pattern for catching specific exception types first

try:
    rag = RAG(collection_name=collection_name)
    rag.index(request.path)
    return {"status": "finished"}
except ValueError as e:
    # User input errors - return 400
    raise HTTPException(status_code=400, detail="User-friendly message")
except Exception as e:
    # Internal errors - return 500
    logger.exception("Context: %s", str(e))
    raise HTTPException(status_code=500, detail="Generic internal error message")
```

---

## Edge Cases & Risks

| Risk/Edge Case                        | Mitigation                                              |
| ------------------------------------- | ------------------------------------------------------- |
| Template may not have error variable  | Check and add error handling to template if needed      |
| Existing clients expect specific text | Maintain "An internal error occurred" phrasing for 500s  |
| Logging not configured                | Python logging defaults to WARNING level if not configured |

---

## Validation

### Automated Checks

```bash
# Run the new tests
python -m pytest tests/test_web.py -v

# Run existing tests to ensure no regression
python -m pytest tests/ -v

# Type checking (if available)
python -m mypy src/pyrag/web.py --ignore-missing-imports

# Linting
ruff check src/pyrag/web.py
```

### Manual Verification

1. Start the web server: `python -m pyrag.web`
2. Submit an invalid URL that triggers a network error
3. Verify the response shows "An internal error occurred..." instead of raw exception
4. Submit an invalid file type
5. Verify the response shows "Invalid input provided..." for 400 errors

---

## Scope Boundaries

**IN SCOPE:**

- Sanitizing error messages in `POST /index`
- Sanitizing error messages in `GET /`
- Sanitizing error messages in `discover_documents()`
- Adding server-side logging for debugging
- Adding tests for new error handling

**OUT OF SCOPE (do not touch):**

- CLI error handling (different audience, less severe)
- Storage.py silent exception handling (separate issue)
- Changes to RAG class exception handling
- Database connection pooling details

---

## Metadata

- **Investigated by**: GHAR
- **Timestamp**: 2026-03-27T00:00:00Z
- **Artifact**: `.ghar/issues/issue-17.md`
