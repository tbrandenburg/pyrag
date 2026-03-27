# Investigation: RAG model reloaded on every HTTP request — ~2.5s cold start per request

**Issue**: #18 (https://github.com/tbrandenburg/pyrag/issues/18)
**Type**: BUG
**Investigated**: 2026-03-27T00:00:00Z

### Assessment

| Metric     | Value                         | Reasoning                                                                                              |
| ---------- | ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| Severity   | HIGH                          | Major performance issue causing ~2.5s latency on every request, makes app unusable under any load     |
| Complexity | LOW                           | Single file (web.py), clear pattern to mirror from mcp.py, no architectural changes required         |
| Confidence | HIGH                          | Clear root cause identified: RAG() instantiation in route handlers, proven by server logs and code   |

---

## Problem Statement

Every HTTP request to the FastAPI web server creates a new `RAG` instance, which loads the `SentenceTransformer` model (~2.5 seconds). This makes the web interface unusable under any load as each request incurs the full cold-start penalty.

---

## Implementation Plan

### Step 1: Add lifespan handler and import contextmanager

**File**: `src/pyrag/web.py`
**Lines**: 1-26
**Action**: UPDATE

Added `asynccontextmanager` import and lifespan handler to initialize/cleanup RAG cache.

### Step 2: Update index_document endpoint to use cached RAG

**File**: `src/pyrag/web.py`
**Lines**: 168-176
**Action**: UPDATE

Replaced direct `RAG()` call with `_get_rag()` using cached instance.

### Step 3: Update discover_documents to accept rag parameter

**File**: `src/pyrag/web.py`
**Lines**: 179-210
**Action**: UPDATE

Changed function signature to accept rag parameter instead of creating new instance.

### Step 4: Update index endpoint to use cached RAG

**File**: `src/pyrag/web.py`
**Lines**: 141-165
**Action**: UPDATE

Get cached RAG instance and pass to discover_documents.

---

## Validation

### Automated Checks

```bash
make lint
make test
```

### Manual Verification

1. Start the web server with `make run-web`
2. Make multiple requests to GET `/` and POST `/index`
3. Verify "Load pretrained SentenceTransformer" appears only ONCE in logs (at startup)
4. Verify no "Failed to initialize AsyncMilvusClient" warnings after startup

---

## Metadata

- **Investigated by**: GHAR
- **Timestamp**: 2026-03-27T00:00:00Z
- **Implementation**: gh-actions (OpenCode)
