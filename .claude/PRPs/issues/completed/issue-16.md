# Issue #16 - Investigation Complete

**Issue**: #16 - [SECURITY] Local file read / SSRF via POST /index — no path validation
**Type**: BUG
**Severity**: CRITICAL
**Status**: Fixed

## Files Modified

| File | Action |
|------|--------|
| `src/pyrag/config.py` | Added ALLOWED_URL_SCHEMES, ALLOWED_BASE_PATHS |
| `src/pyrag/utils.py` | Added PathValidationError, validate_path_security() |
| `src/pyrag/web.py` | Handle PathValidationError with 400 |
| `src/pyrag/mcp.py` | Handle PathValidationError in add_doc |
| `tests/test_security.py` | Security test suite (8 tests) |
| `tests/test_system.py` | Updated test to use URL |

## PR

- PR: https://github.com/tbrandenburg/pyrag/pull/24
- Branch: fix/issue-16-path-validation

## Implementation Notes

- Default-deny for local file access (empty ALLOWED_BASE_PATHS)
- Only http/https URL schemes allowed
- Cloud metadata endpoints blocked (169.254.169.254, metadata.google.internal)
- Path traversal prevented via Path.resolve() + relative_to()
