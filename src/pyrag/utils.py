"""Utility functions for PyRAG."""

import fnmatch
from pathlib import Path
from urllib.parse import urlparse

from .config import ALLOWED_BASE_PATHS, ALLOWED_URL_SCHEMES, SUPPORTED_EXTENSIONS


class PathValidationError(ValueError):
    """Raised when a path fails security validation."""


def looks_like_url(s: str) -> bool:
    """Check if input is a valid URL with scheme and netloc."""
    try:
        r = urlparse(s)
        return all([r.scheme, r.netloc])
    except Exception:
        return False


def has_url_scheme(s: str) -> bool:
    """Check if input has any URL scheme (e.g., file://, http://)."""
    try:
        r = urlparse(s)
        return bool(r.scheme)
    except Exception:
        return False


def validate_path_security(path_or_url: str) -> None:
    """
    Validate that a path or URL meets security requirements.

    Raises PathValidationError if the path is not allowed.
    """
    if has_url_scheme(path_or_url):
        parsed = urlparse(path_or_url)
        if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
            raise PathValidationError(
                f"URL scheme '{parsed.scheme}' is not allowed. "
                f"Allowed schemes: {', '.join(sorted(ALLOWED_URL_SCHEMES))}"
            )
        if parsed.netloc in {"169.254.169.254", "metadata.google.internal"}:
            raise PathValidationError(
                f"URL targets cloud metadata endpoint '{parsed.netloc}' which is not allowed"
            )
        return

    if not ALLOWED_BASE_PATHS:
        raise PathValidationError(
            "Local file access is disabled. Set ALLOWED_BASE_PATHS in configuration to enable."
        )

    p = Path(path_or_url).resolve()

    allowed = False
    for base in ALLOWED_BASE_PATHS:
        base_path = Path(base).resolve()
        try:
            p.relative_to(base_path)
            allowed = True
            break
        except ValueError:
            continue

    if not allowed:
        raise PathValidationError(
            f"Path '{path_or_url}' is not within allowed directories. "
            f"Allowed directories: {', '.join(ALLOWED_BASE_PATHS)}"
        )


def get_supported_files(path_or_url: str, exclude_patterns: list[str] | None = None):
    """
    Returns a list of valid input files:
    - URL → [URL] (if scheme is allowed)
    - File → [file] (if within allowed base paths)
    - Directory → recursively collect supported file types (if within allowed base paths)

    Args:
        path_or_url: Path to file/directory or URL to index.
        exclude_patterns: Optional list of fnmatch patterns matched against paths
            relative to the root directory. E.g. ``["*.md", "temp/*"]``.
            Note: ``*`` matches any character including ``/``, so ``temp/*``
            excludes all files under ``temp/`` at any depth.

    Raises:
        PathValidationError: If path/URL fails security validation
    """
    validate_path_security(path_or_url)

    if looks_like_url(path_or_url):
        return [path_or_url]

    p = Path(path_or_url)

    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [str(p)]
        else:
            raise ValueError(f"Unsupported filetype for Docling: {p.suffix}")

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

    raise ValueError(f"Invalid path or URL: {path_or_url}")


def clip_text(text: str, threshold: int = 150) -> str:
    """Clip text to a maximum length with ellipsis."""
    return f"{text[:threshold]}..." if len(text) > threshold else text
