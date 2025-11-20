"""Configuration constants for PyRAG."""

from langchain_docling.loader import ExportType

# Supported Docling file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".xhtml",
    ".md",
    ".adoc",
    ".txt",
    ".csv",
    ".tsv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".xml",
    ".vtt",
    ".srt",
}

# Default configuration
DEFAULT_EXPORT_TYPE = ExportType.DOC_CHUNKS
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "docling_retriever_demo"
DEFAULT_TOP_K = 5
