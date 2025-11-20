"""Utility functions for PyRAG."""

from pathlib import Path
from urllib.parse import urlparse

from .config import SUPPORTED_EXTENSIONS


def looks_like_url(s: str) -> bool:
    """Check if input is a valid URL."""
    try:
        r = urlparse(s)
        return all([r.scheme, r.netloc])
    except Exception:
        return False


def get_supported_files(path_or_url: str):
    """
    Returns a list of valid input files:
    - URL → [URL]
    - File → [file]
    - Directory → recursively collect supported file types
    """
    # URL
    if looks_like_url(path_or_url):
        return [path_or_url]

    p = Path(path_or_url)

    # Single file
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [str(p)]
        else:
            raise ValueError(f"Unsupported filetype for Docling: {p.suffix}")

    # Directory — recursively walk
    if p.is_dir():
        files = [
            str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            raise ValueError(f"No supported files found under directory: {p}")
        return files

    raise ValueError(f"Invalid path or URL: {path_or_url}")


def clip_text(text: str, threshold: int = 150) -> str:
    """Clip text to a maximum length with ellipsis."""
    return f"{text[:threshold]}..." if len(text) > threshold else text
