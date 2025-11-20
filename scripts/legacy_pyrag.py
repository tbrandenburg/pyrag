#!/usr/bin/env python3
"""
Docling + LangChain + Milvus (Retriever-Only)
- Accepts file, URL, or directory
- Recursively collects supported Docling file types
- Loads or creates a persistent Milvus vector store
- Appends new documents if store exists
- Uses all-MiniLM-L6-v2 for embeddings
- Performs retrieval without any LLM
"""

import os
import sys
import json
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from tempfile import mkdtemp

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus


# ---------------------------------------------------------
# Supported Docling file extensions
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html", ".htm",
    ".xhtml",
    ".md",
    ".adoc",
    ".txt",
    ".csv", ".tsv",
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff",
    ".bmp", ".webp",
    ".xml",
    ".vtt", ".srt",
}


def looks_like_url(s: str) -> bool:
    """Check if input is a valid URL."""
    try:
        r = urlparse(s)
        return all([r.scheme, r.netloc])
    except:
        return False


# ---------------------------------------------------------
# Collect supported input files (URL, file, or directory)
# ---------------------------------------------------------

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
            str(f)
            for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            raise ValueError(f"No supported files found under directory: {p}")
        return files

    raise ValueError(f"Invalid path or URL: {path_or_url}")


def clip_text(text, threshold=150):
    return f"{text[:threshold]}..." if len(text) > threshold else text


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():
    load_dotenv()

    if len(sys.argv) < 1:
        print("Usage: python rag_retriever.py <file|url|directory>")
        sys.exit(1)

    INPUT = sys.argv[1]

    # -----------------------------------------------------
    # Collect supported files
    # -----------------------------------------------------
    print("=" * 80)
    print("Docling RAG — Retriever Only — Load/Create Milvus + Append")
    print("=" * 80)

    print(f"\n1) Scanning for supported files under: {INPUT}")

    file_paths = get_supported_files(INPUT)

    print(f"   Found {len(file_paths)} supported files:")
    for f in file_paths:
        print(f"     • {f}")

    # -----------------------------------------------------
    # Docling Loader
    # -----------------------------------------------------
    EXPORT_TYPE = ExportType.DOC_CHUNKS
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    print("\n2) Loading documents using Docling...")

    loader = DoclingLoader(
        file_path=file_paths,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL),
    )

    docs = loader.load()
    print(f"   Loaded {len(docs)} document chunks")

    # -----------------------------------------------------
    # Chunking (Docling chunks are already good)
    # -----------------------------------------------------
    splits = docs
    print(f"\n3) Using {len(splits)} Docling-generated chunks")

    # -----------------------------------------------------
    # Embeddings
    # -----------------------------------------------------
    print(f"\n4) Initializing embeddings: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # -----------------------------------------------------
    # Milvus Load OR Create + Append
    # -----------------------------------------------------
    COLLECTION_NAME = "docling_retriever_demo"

    STORAGE_DIR = Path("milvus_storage")
    STORAGE_DIR.mkdir(exist_ok=True)
    MILVUS_URI = str(STORAGE_DIR / "docling.db")

    print(f"\n5) Loading/creating Milvus collection: {COLLECTION_NAME}")

    try:
        # Try opening existing Milvus store
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
        )
        print("   ✔ Existing Milvus collection loaded")
        print("   → Appending new documents...")
        vectorstore.add_documents(splits)

    except Exception as e:
        print("   ✘ No existing collection found.")
        print("   → Creating a new Milvus collection")
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            index_params={"index_type": "FLAT"},
            drop_old=False,
        )

    print("   ✔ Vectorstore ready")

    # -----------------------------------------------------
    # Retriever (No LLM)
    # -----------------------------------------------------
    TOP_K = 5
    print(f"\n6) Initializing retriever (TOP_K={TOP_K})")
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Query
    QUERY = "Which are the main AI models in Docling?"

    print("\n7) Searching retrieved chunks...")
    print(f"   Query: {QUERY}")
    print("-" * 80)

    results = retriever.get_relevant_documents(QUERY)

    print(f"\nTop {TOP_K} Retrieved Chunks:\n")
    for i, d in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(clip_text(d.page_content, 350))
        m = {k: v for k, v in d.metadata.items() if k != "pk"}
        print("Metadata:", m)
        print()

    print("=" * 80)
    print("RAG retrieval (no LLM) completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
