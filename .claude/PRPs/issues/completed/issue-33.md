# Investigation: 11 critical/high vulnerabilities found in dependencies

**Issue**: #33 (https://github.com/tbrandenburg/pyrag/issues/33)
**Type**: CHORE
**Investigated**: 2026-04-22T00:00:00Z

### Assessment

| Metric     | Value                         | Reasoning                                                                |
| ---------- | ----------------------------- | ------------------------------------------------------------------------ |
| Priority   | LOW                           | Fixes already applied via #32; lock file has patched versions; no active vulnerability |
| Complexity | LOW                           | No code changes needed; just verification that lock file is current     |
| Confidence | HIGH                          | Clear evidence in uv.lock, git history shows PR #32 merged before issue created |

---

## Problem Statement

A pip-audit security scan reported 11 critical/high vulnerabilities in project dependencies. However, investigation reveals these vulnerabilities have already been addressed via issue #31 fix (merged as PR #32) which updated the lock file with patched versions.

---

## Analysis

### Root Cause / Change Rationale

The security scan in issue #33 appears to have been run after the security fixes from #31 were already merged. The uv.lock file already contains all patched versions:

### Evidence Chain

**Issue #33 created**: 2026-04-21T06:58:50Z
**PR #32 merged**: 2026-04-20T11:34:31Z

↓ BECAUSE: Issue #33 was created AFTER the security fix
Evidence: Git history shows PR #32 merged 1 day before issue #33

### Current Package Versions (from uv.lock)

| Package    | Issue Reported Version | Fixed Version | Lock File Version | Status |
|------------|------------------------|---------------|-------------------|--------|
| cryptography | 41.0.7               | 46.0.5+       | 46.0.7            | ✅ FIXED |
| jinja2     | 3.1.2                  | 3.1.6         | 3.1.6             | ✅ FIXED |
| pyopenssl  | 23.2.0                 | 26.0.0        | NOT IN LOCK      | ✅ N/A (not a direct dep) |
| requests   | 2.31.0                 | 2.32.0        | 2.33.1            | ✅ FIXED |
| urllib3    | 2.0.7                  | 2.6.3         | 2.6.3             | ✅ FIXED |
| twisted    | 24.3.0                 | 24.7.0rc1     | NOT IN LOCK      | ✅ N/A (not a direct dep) |
| setuptools | 68.1.2                 | 70.0.0        | 80.10.2 (<81)    | ✅ FIXED |

### Affected Files

| File            | Lines | Action | Description    |
| --------------- | ----- | ------ | -------------- |
| `pyproject.toml` | 62-68 | NO CHANGE | Constraint dependencies already correct |
| `uv.lock`       | -    | VERIFIED | All patched versions present |

### Integration Points

- Security constraint: `setuptools<81` (pyproject.toml:63) - enforced to maintain milvus-lite compatibility
- All direct dependencies already use patched versions via transitive dependencies

### Git History

- **Issue #31 Security Fix merged**: PR #32 - 2026-04-20 - "Merge pull request #32 from tbrandenburg/fix/issue-31-security-vulnerabilities"
- **Issue #33 created**: 2026-04-21 - After the fix was already merged

---

## Implementation Plan

### Step 1: Verify No Vulnerabilities Remain

**File**: N/A (verification only)
**Action**: VERIFY

Run pip-audit to confirm no vulnerabilities:

```bash
pip-audit
```

Expected: No output (no vulnerabilities found)

---

### Step 2: Update Lock File (Optional - Already Current)

**File**: `uv.lock`
**Action**: VERIFY (already current)

If any vulnerabilities were found, regenerate the lock file:

```bash
uv lock
```

---

## Edge Cases & Risks

| Risk/Edge Case | Mitigation      |
| -------------- | --------------- |
| pip-audit run on stale pip cache | Use `uv pip list` to verify actual installed versions |
| Constraint conflicts with security updates | The setuptools constraint `<81` is for milvus-lite compatibility; 80.x is still patched |

---

## Validation

### Automated Checks

```bash
# Verify lock file is current
uv lock --check

# Run pip-audit (should show no vulnerabilities)
pip-audit
```

---

## Scope Boundaries

**IN SCOPE:**

- Verify no active vulnerabilities remain

**OUT OF SCOPE (do not touch):**

- No code changes required
- Dependencies are already patched

---

## Metadata

- **Investigated by**: GHAR
- **Timestamp**: 2026-04-22T00:00:00Z
- **Artifact**: `.ghar/issues/issue-33.md`
