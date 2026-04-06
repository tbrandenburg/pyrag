# Investigation: Security: 38 critical/high vulnerabilities found in dependencies

**Issue**: #25 (https://github.com/tbrandenburg/pyrag/issues/25)
**Type**: CHORE
**Investigated**: 2026-04-06T08:00:00Z

### Assessment

| Metric     | Value  | Reasoning                                                                                                           |
| ---------- | ------ | ------------------------------------------------------------------------------------------------------------------- |
| Priority   | HIGH   | Multiple CRITICAL CVEs (cryptography, urllib3, pyjwt, pygments) remain unresolved despite partial lockfile updates |
| Complexity | LOW    | Two-file change: update version constraints in `pyproject.toml` and regenerate `uv.lock` via `uv lock --upgrade`   |
| Confidence | HIGH   | Current locked versions are known; fix versions are documented; root cause is stale constraints + partial lockfile  |

---

## Problem Statement

A `pip-audit` security scan identified 38 known vulnerabilities across 15 packages. Despite a previous partial attempt (the `uv.lock` file reflects some updated transitive dependencies), the `pyproject.toml` version constraints for direct dependencies (`jinja2`, `requests`) are still too loose, and several transitive packages remain at vulnerable versions: `urllib3=2.5.0` (needs 2.6.3), `pyjwt=2.10.1` (needs 2.12.0), `pygments=2.19.2` (needs 2.20.0), and `cryptography=46.0.3` (needs 46.0.6). The `torch<2.3.0` constraint pins torch to 2.2.2 which has known CVEs but updating it is out of scope due to compatibility risk.

---

## Analysis

### Change Rationale

The fix requires two actions:
1. Raise minimum version constraints in `pyproject.toml` for direct dependencies with known safe versions.
2. Run `uv lock --upgrade` to regenerate `uv.lock` with all transitive dependencies at their minimum safe versions.

The previous automation (GHAR) ran `uv lock --upgrade` but did NOT commit the result, so the lockfile still reflects old pinned versions in the repository.

### Evidence Chain

WHY: `pip-audit` reports 38 vulnerabilities
↓ BECAUSE: `uv.lock` pins packages at vulnerable versions
Evidence: `uv.lock` - `urllib3 version = "2.5.0"` (CVE-2025-66418, CVE-2025-66471, CVE-2026-21441 need 2.6.3)
Evidence: `uv.lock` - `pyjwt version = "2.10.1"` (CVE-2026-32597 needs 2.12.0)
Evidence: `uv.lock` - `pygments version = "2.19.2"` (CVE-2026-4539 needs 2.20.0)
Evidence: `uv.lock` - `cryptography version = "46.0.3"` (CVE-2026-34073 needs 46.0.6)

↓ BECAUSE: `pyproject.toml` constraints allow vulnerable versions
Evidence: `pyproject.toml:36` - `"jinja2>=3.1.2"` (allows vulnerable 3.1.2; should be >=3.1.6)
Evidence: `pyproject.toml:47` - `"requests>=2.31.0"` (allows vulnerable 2.31.0; should be >=2.33.0)

↓ ROOT CAUSE: `pyproject.toml` minimum version constraints are stale and `uv.lock` has not been regenerated with `--upgrade` and committed
Evidence: `pyproject.toml:26` - `"torch>=1.13.0,<2.3.0"` (torch 2.2.2 has CVEs, but changing this constraint is out of scope due to compatibility risk)

### Affected Files

| File             | Lines  | Action | Description                                                          |
| ---------------- | ------ | ------ | -------------------------------------------------------------------- |
| `pyproject.toml` | 36, 47 | UPDATE | Raise `jinja2>=3.1.6` and `requests>=2.33.0` (dev)                  |
| `uv.lock`        | ALL    | UPDATE | Regenerate via `uv lock --upgrade` to pull safe transitive versions  |

### Integration Points

- `pyproject.toml` is the source of truth for dependency constraints
- `uv.lock` is the pinned lockfile derived from `pyproject.toml`
- `.github/workflows/test.yml` runs `uv sync --group dev` to install from the lockfile
- No source code files import the affected packages directly (they are transitive or test-only)

### Git History

- **pyproject.toml last modified**: `4691054` - "Added targets for web and mcp"
- **uv.lock last modified**: `4691054` - same commit (initial project setup with web/mcp)
- **Implication**: Version constraints have not been updated since initial commit; this is a long-standing issue, not a regression

---

## Implementation Plan

### Step 1: Update direct dependency minimum versions in pyproject.toml

**File**: `pyproject.toml`
**Lines**: 36, 47
**Action**: UPDATE

**Current code (`pyproject.toml:36`):**

```toml
    "jinja2>=3.1.2",
```

**Required change:**

```toml
    "jinja2>=3.1.6",
```

**Why**: Raises minimum to version that fixes 5 CVEs (CVE-2024-22195, CVE-2024-34064, CVE-2024-56201, CVE-2024-56326, CVE-2025-27516).

---

**Current code (`pyproject.toml:47`):**

```toml
    "requests>=2.31.0",
```

**Required change:**

```toml
    "requests>=2.33.0",
```

**Why**: Raises minimum to version that fixes 3 CVEs (CVE-2024-35195, CVE-2024-47081, CVE-2026-25645). This is a dev dependency, but still should be updated.

---

### Step 2: Regenerate uv.lock with upgraded transitive dependencies

**Action**: Run `uv lock --upgrade` to regenerate the lockfile

```bash
uv lock --upgrade
```

**Why**: Forces uv to resolve all transitive dependencies to their latest compatible versions, which will pick up:
- `urllib3>=2.6.3` (fixes CVE-2025-66418, CVE-2025-66471, CVE-2026-21441)
- `pyjwt>=2.12.0` (fixes CVE-2026-32597)
- `pygments>=2.20.0` (fixes CVE-2026-4539)
- `cryptography>=46.0.6` (fixes CVE-2026-26007, CVE-2026-34073)

**Note on torch**: The constraint `torch>=1.13.0,<2.3.0` will limit torch to 2.2.2 which has reported CVEs. Do NOT change this constraint - torch 2.3+ may have breaking API changes. This is a known accepted risk, documented in scope boundaries below.

---

### Step 3: Sync and verify

**Action**: Install updated packages and verify

```bash
uv sync --group dev
```

**Then verify key package versions:**

```bash
uv pip show urllib3 | grep Version
uv pip show pyjwt | grep Version
uv pip show pygments | grep Version
uv pip show cryptography | grep Version
```

Expected outputs:
- urllib3: >= 2.6.3
- pyjwt: >= 2.12.0
- pygments: >= 2.20.0
- cryptography: >= 46.0.6

---

### Step 4: Run tests to ensure no regressions

**Action**: Run the full QA suite

```bash
make qa
```

**Why**: Confirms all tests pass with the updated dependency versions.

---

## Patterns to Follow

**From project - existing pyproject.toml constraint style:**

```toml
# SOURCE: pyproject.toml:15-38
# Pattern: use >= minimum with no upper bound (unless compatibility requires it)
dependencies = [
    "langchain>=1.0.8",
    "langchain-core>=1.0.6",
    ...
    "jinja2>=3.1.2",    # <-- this is what we update
]
```

**Upper bound exception pattern (keep torch as-is):**

```toml
# SOURCE: pyproject.toml:26
# torch has explicit upper bound for compatibility - do NOT change
"torch>=1.13.0,<2.3.0",
```

---

## Edge Cases & Risks

| Risk/Edge Case                                     | Mitigation                                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `torch<2.3.0` keeps torch at 2.2.2 with CVEs      | Accepted risk; updating torch may break API compatibility; leave for separate issue        |
| `uv lock --upgrade` may pull breaking dep changes  | Run `make qa` to detect regressions; revert specific packages if tests fail                |
| Some CVEs may be in sys packages (pip, setuptools) | pip-audit scans system python; project venv packages are what matters, not system ones     |
| Daily security scan will re-report until fixed     | Merging this PR will update the lockfile and silence the scanner for resolved CVEs         |

---

## Validation

### Automated Checks

```bash
# Update constraints and regenerate lockfile
uv lock --upgrade

# Install updated packages
uv sync --group dev

# Verify key versions (expect: urllib3>=2.6.3, pyjwt>=2.12.0, pygments>=2.20.0, cryptography>=46.0.6)
uv pip show urllib3 pyjwt pygments cryptography | grep -E "^(Name|Version)"

# Run full QA (format + lint + test)
make qa
```

### Manual Verification

1. Run `uv pip show urllib3` - verify Version >= 2.6.3
2. Run `uv pip show pyjwt` - verify Version >= 2.12.0
3. Run `uv pip show pygments` - verify Version >= 2.20.0
4. Run `uv pip show cryptography` - verify Version >= 46.0.6
5. Run `pip-audit` (if available) - verify reduced vulnerability count (torch CVEs may remain)

---

## Scope Boundaries

**IN SCOPE:**

- Update `jinja2>=3.1.2` to `jinja2>=3.1.6` in `pyproject.toml`
- Update `requests>=2.31.0` to `requests>=2.33.0` in dev dependencies
- Run `uv lock --upgrade` to regenerate `uv.lock`
- Verify all tests pass after update

**OUT OF SCOPE (do not touch):**

- `torch<2.3.0` constraint (compatibility risk; torch 2.2.2 CVEs are accepted for now)
- Any source code in `src/pyrag/`
- System-level packages (pip, setuptools, wheel) - those are not project dependencies
- Any other `pyproject.toml` settings (ruff, pytest config, etc.)

---

## Metadata

- **Investigated by**: Claude (OpenCode)
- **Timestamp**: 2026-04-06T08:00:00Z
- **Artifact**: `.claude/PRPs/issues/issue-25.md`
