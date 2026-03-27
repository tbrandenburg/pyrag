# Investigation: Header stats not refreshed after indexing a document

**Issue**: #20 (https://github.com/tbrandenburg/pyrag/issues/20)
**Type**: BUG
**Investigated**: 2026-03-27T00:00:00Z

### Assessment

| Metric     | Value   | Reasoning                                                                                             |
| ---------- | ------- | ----------------------------------------------------------------------------------------------------- |
| Severity   | MEDIUM  | Feature partially broken - indexing works but UI stats don't update; workaround is page reload        |
| Complexity | LOW     | Only 2 files affected (templates/index.html, web.py); isolated change with low risk                  |
| Confidence | HIGH    | Clear root cause identified in indexSource() function; evidence from code inspection confirms issue  |

---

## Problem Statement

After successfully indexing a document via the web UI, the header statistics cards (SOURCES, CHUNKS, TEXT CHUNKS, TABLE CHUNKS) and the "N Indexed Source" section heading are not updated. Users must manually reload the page to see current counts.

---

## Analysis

### Root Cause

The `indexSource()` JavaScript function in `templates/index.html` only updates the individual card status to "Finished" after successful indexing. It does not fetch updated statistics from the server or update the header stats DOM elements.

### Evidence Chain

**WHY**: Header stats don't update after indexing
**↓ BECAUSE**: `indexSource()` at `templates/index.html:427-462` only calls `updateCardStatus(card, 'Finished')`
**↓ Evidence**: Line 456 - `updateCardStatus(card, 'Finished');` is the last line before function ends

**↓ ROOT CAUSE**: Stats are server-side rendered via Jinja2 (lines 158-178) and there is no JavaScript mechanism to fetch or update them dynamically after indexing completes

### Affected Files

| File                               | Lines   | Action | Description                                              |
| ---------------------------------- | ------- | ------ | -------------------------------------------------------- |
| `src/pyrag/templates/index.html`   | 158-178 | UPDATE | Add `id` attributes to stat elements for JS targeting    |
| `src/pyrag/templates/index.html`   | 234-240 | UPDATE | Add `id` to section heading for JS targeting             |
| `src/pyrag/templates/index.html`   | 427-462 | UPDATE | Modify `indexSource()` to fetch and update stats        |
| `src/pyrag/web.py`                 | 168-176 | UPDATE | Add new `/stats` GET endpoint returning JSON stats      |

### Integration Points

- `indexSource()` calls `POST /index` to index documents (line 430)
- `discover_documents()` function in `web.py:179-210` provides stats data
- Stats are initially rendered in template via Jinja2 from `discover_documents()` return value

---

## Implementation Plan

### Step 1: Add `/stats` endpoint to web.py

**File**: `src/pyrag/web.py`
**Lines**: 168-176 (after existing `/index` endpoint)
**Action**: UPDATE

**Required change:**

Add a new GET endpoint after line 176:

```python
@app.get("/stats")
def get_stats(collection_name: str = Query(DEFAULT_COLLECTION_NAME, description="Milvus collection name")):
    """Return indexing statistics as JSON for frontend updates."""
    try:
        data = discover_documents(collection_name)
        return {
            "total_sources": data.total_sources,
            "total_chunks": data.total_chunks,
            "content_types": data.content_types,
            "source_count": len(data.source_groups),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Why**: Provides a lightweight JSON endpoint for the frontend to fetch updated stats without reloading the page.

---

### Step 2: Add `id` attributes to stat elements in index.html

**File**: `src/pyrag/templates/index.html`
**Lines**: 158-178
**Action**: UPDATE

**Current code:**

```html
<div class="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 brutalist-block text-c-foreground">
    <div class="text-center">
        <p class="text-3xl font-bold text-c-accent">{{ total_sources }}</p>
        <p class="text-xs text-c-subtle mt-1">SOURCES</p>
    </div>

    <div class="text-center">
        <p class="text-3xl font-bold text-c-foreground">{{ total_chunks }}</p>
        <p class="text-xs text-c-subtle mt-1">CHUNKS</p>
    </div>

    <div class="text-center">
        <p class="text-3xl font-bold text-c-foreground">{{ content_types.get('text', 0) }}</p>
        <p class="text-xs text-c-subtle mt-1">TEXT CHUNKS</p>
    </div>
    
    <div class="text-center">
        <p class="text-3xl font-bold text-c-accent">{{ content_types.get('table', 0) }}</p>
        <p class="text-xs text-c-subtle mt-1">TABLE CHUNKS</p>
    </div>
</div>
```

**Required change:**

```html
<div class="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 brutalist-block text-c-foreground">
    <div class="text-center">
        <p id="stat-sources" class="text-3xl font-bold text-c-accent">{{ total_sources }}</p>
        <p class="text-xs text-c-subtle mt-1">SOURCES</p>
    </div>

    <div class="text-center">
        <p id="stat-chunks" class="text-3xl font-bold text-c-foreground">{{ total_chunks }}</p>
        <p class="text-xs text-c-subtle mt-1">CHUNKS</p>
    </div>

    <div class="text-center">
        <p id="stat-text-chunks" class="text-3xl font-bold text-c-foreground">{{ content_types.get('text', 0) }}</p>
        <p class="text-xs text-c-subtle mt-1">TEXT CHUNKS</p>
    </div>
    
    <div class="text-center">
        <p id="stat-table-chunks" class="text-3xl font-bold text-c-accent">{{ content_types.get('table', 0) }}</p>
        <p class="text-xs text-c-subtle mt-1">TABLE CHUNKS</p>
    </div>
</div>
```

**Why**: `id` attributes enable JavaScript to target these elements for dynamic updates.

---

### Step 3: Add `id` to section heading in index.html

**File**: `src/pyrag/templates/index.html`
**Lines**: 234-240
**Action**: UPDATE

**Current code:**

```html
<h2 class="text-2xl font-semibold text-c-foreground mb-6 text-center">
    {% if source_groups %}
        {{ source_groups | length }} Indexed Source{{ 's' if source_groups | length != 1 else '' }}
    {% else %}
        No Indexed Sources Yet
    {% endif %}
</h2>
```

**Required change:**

```html
<h2 id="indexed-sources-heading" class="text-2xl font-semibold text-c-foreground mb-6 text-center">
    {% if source_groups %}
        {{ source_groups | length }} Indexed Source{{ 's' if source_groups | length != 1 else '' }}
    {% else %}
        No Indexed Sources Yet
    {% endif %}
</h2>
```

**Why**: `id` attribute enables JavaScript to update the heading text after indexing.

---

### Step 4: Update `indexSource()` function to refresh stats

**File**: `src/pyrag/templates/index.html`
**Lines**: 427-462
**Action**: UPDATE

**Current code:**

```javascript
async function indexSource(source, card) {
    updateCardStatus(card, 'In Progress');
    try {
        const response = await fetch('/index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                path: source,
                collection_name: collectionName,
            }),
        });

        if (!response.ok) {
            let errorMessage = 'Indexing failed';
            try {
                const errorJson = await response.json();
                if (errorJson?.detail) {
                    errorMessage = errorJson.detail;
                }
            } catch (_) {
                const errorText = await response.text();
                if (errorText) errorMessage = errorText;
            }

            throw new Error(errorMessage);
        }

        updateCardStatus(card, 'Finished');
    } catch (error) {
        console.error('Indexing failed', error);
        updateCardStatus(card, 'Failed', error.message);
        showModal('Indexing Failed', `Failed to index <code>${source}</code>.<br>${error.message}`);
    }
}
```

**Required change:**

```javascript
async function indexSource(source, card) {
    updateCardStatus(card, 'In Progress');
    try {
        const response = await fetch('/index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                path: source,
                collection_name: collectionName,
            }),
        });

        if (!response.ok) {
            let errorMessage = 'Indexing failed';
            try {
                const errorJson = await response.json();
                if (errorJson?.detail) {
                    errorMessage = errorJson.detail;
                }
            } catch (_) {
                const errorText = await response.text();
                if (errorText) errorMessage = errorText;
            }

            throw new Error(errorMessage);
        }

        updateCardStatus(card, 'Finished');
        await refreshStats();
    } catch (error) {
        console.error('Indexing failed', error);
        updateCardStatus(card, 'Failed', error.message);
        showModal('Indexing Failed', `Failed to index <code>${source}</code>.<br>${error.message}`);
    }
}

async function refreshStats() {
    try {
        const params = new URLSearchParams({ collection_name: collectionName });
        const response = await fetch(`/stats?${params}`);
        if (!response.ok) return;
        const stats = await response.json();
        
        document.getElementById('stat-sources').textContent = stats.total_sources;
        document.getElementById('stat-chunks').textContent = stats.total_chunks;
        document.getElementById('stat-text-chunks').textContent = stats.content_types.text || 0;
        document.getElementById('stat-table-chunks').textContent = stats.content_types.table || 0;
        
        const heading = document.getElementById('indexed-sources-heading');
        if (stats.source_count === 0) {
            heading.textContent = 'No Indexed Sources Yet';
        } else {
            heading.textContent = `${stats.source_count} Indexed Source${stats.source_count !== 1 ? 's' : ''}`;
        }
    } catch (error) {
        console.error('Failed to refresh stats', error);
    }
}
```

**Why**: After successful indexing, fetch updated stats from the new `/stats` endpoint and update all stat elements and the section heading.

---

## Patterns to Follow

**From web.py - existing endpoint pattern:**

```python
# SOURCE: web.py:168-176
# Pattern for new /stats endpoint
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

**From index.html - existing updateCardStatus pattern:**

```javascript
// SOURCE: index.html:373-400
// Pattern for DOM updates
function updateCardStatus(card, statusText, message = null) {
    const statusLabel = card.querySelector('[data-role="status-label"]');
    // ... DOM manipulation pattern
}
```

---

## Edge Cases & Risks

| Risk/Edge Case           | Mitigation                                                                 |
| ------------------------ | -------------------------------------------------------------------------- |
| `/stats` endpoint fails   | `refreshStats()` catches errors silently, stats remain unchanged          |
| `content_types` missing  | Use `|| 0` fallback for text/table counts in `refreshStats()`              |
| Empty collection          | Heading shows "No Indexed Sources Yet" - handled by fallback logic        |
| Race condition            | User indexes multiple sources rapidly - each call refreshes stats after     |

---

## Validation

### Automated Checks

```bash
# Run existing tests to ensure no regressions
make test

# Run linting to check code quality
make lint
```

### Manual Verification

1. Open web UI and note initial stats (e.g., 1 source, 100 chunks)
2. Submit a new document URL for indexing
3. Observe the card shows "Finished" status
4. **Verify**: Header stats update without page reload
5. **Verify**: "N Indexed Sources" heading updates correctly
6. Submit another document and verify stats continue to update

---

## Scope Boundaries

**IN SCOPE:**

- Adding `/stats` GET endpoint to `web.py`
- Adding `id` attributes to stat elements in `index.html`
- Adding `id` to section heading in `index.html`
- Modifying `indexSource()` to call `refreshStats()` after success
- Creating `refreshStats()` function to update DOM

**OUT OF SCOPE (do not touch):**

- `discover_documents()` function - already exists and works correctly
- `updateCardStatus()` function - works as designed
- Any backend RAG indexing logic
- Other pages or endpoints beyond `/` and `/stats`

---

## Metadata

- **Investigated by**: GHAR
- **Timestamp**: 2026-03-27T00:00:00Z
- **Artifact**: `.ghar/issues/issue-20.md`

---

## Implementation Log

- **Branch**: `fix/issue-20-stats-refresh`
- **Commit**: `ac1f8a0` - Fix: Header stats not refreshed after indexing (#20)
- **Status**: Completed
- **Date**: 2026-03-27
