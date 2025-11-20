"""PyRAG - Docling-powered modular RAG CLI."""

from .config import SUPPORTED_EXTENSIONS
from .utils import clip_text, get_supported_files, looks_like_url

__version__ = "0.1.0"
__all__ = [
    "SUPPORTED_EXTENSIONS",
    "looks_like_url",
    "get_supported_files",
    "clip_text",
]
