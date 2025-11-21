# PyRAG

A Docling-powered modular RAG (Retrieval-Augmented Generation) CLI tool for document processing and intelligent search.

## Features

- **Multiple input formats**: Supports files, URLs, and directories
- **Docling integration**: Advanced document parsing and chunking
- **Vector storage**: Persistent Milvus vector database
- **Flexible retrieval**: Configurable top-k search
- **CLI interface**: Easy-to-use command-line tool

## Supported File Types

PyRAG supports a wide range of document formats through Docling:

- **Documents**: PDF, DOCX, PPTX, XLSX, HTML, XHTML, Markdown, AsciiDoc, TXT
- **Data files**: CSV, TSV, XML
- **Images**: PNG, JPG, JPEG, TIF, TIFF, BMP, WEBP
- **Subtitles**: VTT, SRT

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/tbrandenburg/pyrag.git
cd pyrag
uv sync

# 2. Test the installation  
uv run python scripts/verify_installation.py

# 3. Search your documents
uv run pyrag --add /path/to/your/documents --query "What is this about?"
uv run pyrag --add https://arxiv.org/pdf/2408.09869 --query "Which are the main AI models in Docling?"
```

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/tbrandenburg/pyrag.git
cd pyrag

# Install the project and dependencies
uv sync

# Alternatively, install with development dependencies
uv sync --group dev
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/tbrandenburg/pyrag.git
cd pyrag

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Basic usage - index documents
uv run pyrag --add /path/to/documents

# Index and search documents with a custom query
uv run pyrag --add /path/to/documents --query "What are the main features?"

# Index and search a URL with custom query
uv run pyrag --add https://arxiv.org/pdf/2408.09869 --query "Which are the main AI models in Docling?"

# Search existing indexed documents only
uv run pyrag --query "AI models"

# Specify number of results
uv run pyrag --add /path/to/documents --query "AI models" --top-k 10

# Use custom collection name
uv run pyrag --add /path/to/documents --collection my_collection

# Get help
uv run pyrag --help
```

### Python Module

```bash
# Run as a module (legacy support)
uv run python -m pyrag --add /path/to/documents
```

### Python API

```python
from pyrag.rag import RAG

# Initialize the pipeline
pipeline = RAG(top_k=5)

# Process documents and search
results = pipeline.run("/path/to/documents", "What are the key concepts?")
```

## Configuration

PyRAG uses sensible defaults but can be customized:

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Collection name**: `docling_retriever_demo`
- **Top-k results**: `5`
- **Storage directory**: `milvus_storage/`

## Development

### Using Make (Recommended)

```bash
# Install project with dependencies
make install

# Run all quality assurance tasks (lint + format + test)
make qa

# Individual tasks
make lint      # Run ruff linter
make format    # Format code with ruff  
make test      # Run pytest test suite

# Build and run
make build     # Build the package
make run       # Run CLI (shows help)
make run ARGS="--add /path/to/docs --query 'your question'"

# Maintenance
make clean     # Clean build artifacts
make verify    # Verify installation
```

### Manual Commands

```bash
# Install project with development dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Format and lint code with ruff
uv run ruff format src/ tests/
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Verify installation
uv run python scripts/verify_installation.py
```

### Running Tests

```bash
# Using Make
make test

# Or manually
uv run pytest tests/
```

## Project Structure

```
pyrag/
├── src/pyrag/              # Main package source
│   ├── __init__.py         # Package initialization
│   ├── __main__.py         # Module entry point  
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration constants
│   ├── pipeline.py         # Main RAG pipeline
│   └── utils.py            # Utility functions
├── tests/                  # Test suite
├── scripts/                # Utility scripts
│   ├── examples            # Another example
│   └── legacy_pyrag.py     # Original script
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## Architecture

PyRAG is built with a modular architecture:

- `pyrag.pipeline`: Core RAG pipeline implementation
- `pyrag.utils`: Utility functions for file handling and text processing
- `pyrag.config`: Configuration constants and defaults
- `pyrag.cli`: Command-line interface using Typer

## Requirements

- **Python**: 3.12
- **uv**: Modern Python package manager (recommended)
- **Platform**: macOS, Linux, Windows

## Dependencies

- **Docling**: Document parsing and chunking
- **LangChain**: Document processing and vector store integration  
- **Milvus**: Vector database for semantic search
- **HuggingFace**: Embedding models
- **Typer**: CLI framework
- **Rich**: Enhanced terminal output
- **PyTorch**: ML framework (CPU version)
- **NumPy**: Numerical computing (v1.x for compatibility)

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Contributions are welcome! Please feel free to submit a Pull Request.
