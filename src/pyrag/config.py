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
DEFAULT_COLLECTION_NAME = "rag"
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 128  # Conservative limit well under 256 token model limit
DEFAULT_OVERLAP_TOKENS = 10  # Minimal overlap to avoid token buildup
DEFAULT_MILVUS_URI = "milvus_storage/docling.db"

# Ensemble retriever configuration
DEFAULT_VECTOR_WEIGHT = 0.7  # 70% vector retrieval weight
DEFAULT_BM25_WEIGHT = 0.3  # 30% BM25 retrieval weight
DEFAULT_RANK_FUSION_CONSTANT = 60  # Default rank fusion constant
