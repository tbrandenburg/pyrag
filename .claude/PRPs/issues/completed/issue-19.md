# Issue #19 — Filter/exclude-paths UI field is non-functional

**Issue URL:** https://github.com/tbrandenburg/pyrag/issues/19  
**Type:** Bug  
**Priority:** Medium  
**Status:** Ready for implementation  
**Investigated:** 2026-04-06

---

## Problem Statement

The web UI presents a "Filter" input field (placeholder: `e.g., *.md, temp/*, /ignore-me`) that visually suggests users can exclude files/paths from indexing. However, the value entered is never read by JavaScript, never sent to the server, and never passed to the indexing pipeline. The field is completely inert — a broken promise in the UI.

---

## Root Cause Analysis

The bug spans all four layers of the stack. No single layer has partial support; the entire feature is missing end-to-end.

### Layer 1 — HTML (`src/pyrag/templates/index.html`)

The input exists at lines 222-225:
```html
<input type="text" id="exclude-filter"
    class="w-full bg-transparent text-c-foreground text-base focus:outline-none"
    placeholder="e.g., *.md, temp/*, /ignore-me"
>
```

The `indexDocuments()` function (lines 495-508) reads only `doc-source`:
```js
async function indexDocuments() {
    const sources = document.getElementById('doc-source').value.trim();
    // exclude-filter value is never read
    ...
    for (const source of sourceArray) {
        const card = addDocumentCard(source, 'In Progress');
        await indexSource(source, card);  // no exclude_patterns passed
    }
}
```

The `indexSource()` function (lines 434-469) sends only `path` and `collection_name`:
```js
body: JSON.stringify({
    path: source,
    collection_name: collectionName,
    // exclude_patterns missing
}),
```

### Layer 2 — API model (`src/pyrag/web.py`)

`IndexRequest` at line 68-71 has no `exclude_patterns` field:
```python
class IndexRequest(BaseModel):
    path: str
    collection_name: str = DEFAULT_COLLECTION_NAME
    # exclude_patterns missing
```

The endpoint at line 213 passes no patterns to `rag.index()`:
```python
rag.index(request.path)
```

### Layer 3 — RAG pipeline (`src/pyrag/rag.py`)

`RAG.index()` at line 135-138 passes no patterns to `get_supported_files()`:
```python
def index(self, input_path: str):
    load_dotenv()
    file_paths = get_supported_files(input_path)  # no exclude_patterns
```

### Layer 4 — Utils (`src/pyrag/utils.py`)

`get_supported_files()` at line 74 has no `exclude_patterns` parameter:
```python
def get_supported_files(path_or_url: str):
    ...
    if p.is_dir():
        files = [
            str(f) for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
```

---

## Implementation Plan

### Step 1 — `src/pyrag/utils.py`

Add `exclude_patterns: list[str] | None = None` parameter to `get_supported_files()`.  
Apply pattern matching using `fnmatch` against the relative path string when iterating directory files.

```python
import fnmatch

def get_supported_files(path_or_url: str, exclude_patterns: list[str] | None = None):
    ...
    if p.is_dir():
        files = []
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if exclude_patterns:
                rel = str(f.relative_to(p))
                if any(fnmatch.fnmatch(rel, pat) for pat in exclude_patterns):
                    continue
            files.append(str(f))
        if not files:
            raise ValueError(f"No supported files found under directory: {p}")
        return files
```

Pattern semantics: each pattern in `exclude_patterns` is matched against the path **relative to the root directory** using `fnmatch`. Examples:
- `*.md` — excludes all markdown files in any subdir
- `temp/*` — excludes everything under `temp/`
- `README.md` — excludes a specific file by name

### Step 2 — `src/pyrag/rag.py`

Add `exclude_patterns: list[str] | None = None` to `RAG.index()` and forward it:

```python
def index(self, input_path: str, exclude_patterns: list[str] | None = None):
    load_dotenv()
    file_paths = get_supported_files(input_path, exclude_patterns=exclude_patterns)
    ...
```

### Step 3 — `src/pyrag/web.py`

Add `exclude_patterns` field to `IndexRequest` (optional, defaults to empty list):

```python
class IndexRequest(BaseModel):
    path: str
    collection_name: str = DEFAULT_COLLECTION_NAME
    exclude_patterns: list[str] = []
```

Forward to `rag.index()` in the endpoint:

```python
rag.index(request.path, exclude_patterns=request.exclude_patterns or None)
```

### Step 4 — `src/pyrag/templates/index.html`

In `indexDocuments()`, read the filter field and parse comma-separated patterns:

```js
async function indexDocuments() {
    const sources = document.getElementById('doc-source').value.trim();
    const filterValue = document.getElementById('exclude-filter').value.trim();
    const excludePatterns = filterValue
        ? filterValue.split(',').map(p => p.trim()).filter(p => p !== '')
        : [];
    ...
    for (const source of sourceArray) {
        const card = addDocumentCard(source, 'In Progress');
        await indexSource(source, card, excludePatterns);
    }
}
```

In `indexSource()`, accept and send `excludePatterns`:

```js
async function indexSource(source, card, excludePatterns = []) {
    ...
    body: JSON.stringify({
        path: source,
        collection_name: collectionName,
        exclude_patterns: excludePatterns,
    }),
```

### Step 5 — `tests/test_unit.py`

Add tests for `get_supported_files()` with `exclude_patterns`:
- Excludes files matching a simple glob pattern (`*.md`)
- Excludes files in a subdirectory (`subdir/*`)
- Does NOT exclude when patterns don't match
- Empty/None patterns → no filtering

Add test for `RAG.index()` forwarding `exclude_patterns` to `get_supported_files`.

### Step 6 — `tests/test_web.py`

Add tests for the `/index` endpoint:
- `exclude_patterns` field accepted and forwarded to `rag.index()`
- `exclude_patterns` defaults to empty list when not provided
- Patterns are forwarded correctly (mock `rag.index` and assert call args)

---

## Files to Modify

| File | Change |
|------|--------|
| `src/pyrag/utils.py` | Add `exclude_patterns` param to `get_supported_files()`, apply `fnmatch` filtering |
| `src/pyrag/rag.py` | Add `exclude_patterns` param to `RAG.index()`, forward to `get_supported_files()` |
| `src/pyrag/web.py` | Add `exclude_patterns: list[str] = []` to `IndexRequest`, forward to `rag.index()` |
| `src/pyrag/templates/index.html` | Read `#exclude-filter`, parse CSV, pass to `indexSource()`, include in POST body |
| `tests/test_unit.py` | Add unit tests for `get_supported_files` filtering and `RAG.index` forwarding |
| `tests/test_web.py` | Add web endpoint tests for `exclude_patterns` field |

---

## Validation

After implementation, run:

```bash
make qa
```

All 23 existing tests must continue to pass. New tests for `exclude_patterns` must pass.

---

## Branch Name

`fix/issue-19-exclude-patterns-filter`

---

## Linked Issue

Closes #19
