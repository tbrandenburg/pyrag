"""Tests for RAG class public methods."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from pyrag.rag import RAG


@pytest.fixture
def temp_milvus_uri():
    """Create a temporary Milvus URI for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield f"{temp_dir}/test_milvus.db"


@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        # Create minimal PDF content
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
            b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n"
            b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>\nendobj\n"
            b"4 0 obj\n<</Length 44>>stream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello World) Tj\n"
            b"ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n0000000189 00000 n \n"
            b"trailer\n<</Size 5/Root 1 0 R>>\nstartxref\n283\n%%EOF"
        )
        temp_file.write(pdf_content)
        yield temp_file.name
    Path(temp_file.name).unlink(missing_ok=True)


def test_rag_load_method(sample_pdf):
    """Test RAG.load() method with document loading and chunking."""
    rag = RAG(milvus_uri="./test_milvus.db")

    # Mock the DoclingLoader to return test documents
    test_docs = [
        Document(
            page_content="Sample document content about AI models", metadata={"source": sample_pdf}
        ),
        Document(
            page_content="Another chunk about machine learning", metadata={"source": sample_pdf}
        ),
    ]

    with patch("pyrag.rag.DoclingLoader") as mock_loader:
        mock_instance = Mock()
        mock_instance.load.return_value = test_docs
        mock_loader.return_value = mock_instance

        documents = rag.load([sample_pdf])

        assert len(documents) == 2
        assert documents[0].page_content == "Sample document content about AI models"
        assert documents[1].page_content == "Another chunk about machine learning"
        mock_loader.assert_called_once()


def test_rag_chunk_method():
    """Test RAG.chunk() method (passthrough functionality)."""
    rag = RAG(milvus_uri="./test_milvus.db")

    documents = [
        Document(page_content="Test document 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test document 2", metadata={"source": "test2.pdf"}),
    ]

    chunked_docs = rag.chunk(documents)

    assert chunked_docs == documents
    assert len(chunked_docs) == 2


def test_rag_embed_method():
    """Test RAG.embed() method (embedding initialization)."""
    rag = RAG(milvus_uri="./test_milvus.db")

    chunks = [
        Document(page_content="Test chunk 1", metadata={"source": "test.pdf"}),
        Document(page_content="Test chunk 2", metadata={"source": "test.pdf"}),
    ]

    embedded_chunks = rag.embed(chunks)

    assert embedded_chunks == chunks
    assert rag.embeddings is not None
    assert hasattr(rag.embeddings, "embed_query")


def test_rag_store_method(temp_milvus_uri):
    """Test RAG.store() method with vectorstore creation and document storage."""
    rag = RAG(milvus_uri=temp_milvus_uri, collection_name="test_collection")

    documents = [
        Document(page_content="AI models are powerful tools", metadata={"source": "ai_doc.pdf"}),
        Document(page_content="Machine learning algorithms", metadata={"source": "ml_doc.pdf"}),
    ]

    # Store documents
    rag.store(documents)

    assert rag.vectorstore is not None
    assert rag.indexed_documents is not None
    assert len(rag.indexed_documents) == 2
    assert rag.indexed_documents[0].page_content == "AI models are powerful tools"


def test_rag_retrieve_method(temp_milvus_uri):
    """Test RAG.retrieve() method with hybrid retriever setup."""
    rag = RAG(milvus_uri=temp_milvus_uri, collection_name="test_collection", top_k=3)

    # First store some documents
    documents = [
        Document(
            page_content="AI models for natural language processing", metadata={"source": "nlp.pdf"}
        ),
        Document(page_content="Deep learning neural networks", metadata={"source": "dl.pdf"}),
        Document(page_content="Machine learning fundamentals", metadata={"source": "ml.pdf"}),
    ]

    rag.store(documents)
    rag.retrieve()

    assert rag.retriever is not None
    assert hasattr(rag, "bm25_retriever")
    assert rag.bm25_retriever is not None
    assert rag.bm25_retriever.k == 3


def test_rag_search_method(temp_milvus_uri):
    """Test RAG.search() method with query execution."""
    rag = RAG(milvus_uri=temp_milvus_uri, collection_name="test_collection", top_k=2)

    # Setup complete pipeline
    documents = [
        Document(
            page_content="Artificial intelligence and machine learning",
            metadata={"source": "ai.pdf"},
        ),
        Document(
            page_content="Natural language processing techniques", metadata={"source": "nlp.pdf"}
        ),
        Document(page_content="Computer vision algorithms", metadata={"source": "cv.pdf"}),
    ]

    rag.store(documents)
    rag.retrieve()

    results = rag.search("artificial intelligence")

    assert isinstance(results, list)
    assert len(results) <= 2  # Respects top_k
    assert all(isinstance(doc, Document) for doc in results)


def test_rag_index_method(sample_pdf, temp_milvus_uri):
    """Test RAG.index() method with end-to-end document indexing."""
    rag = RAG(milvus_uri=temp_milvus_uri, collection_name="test_collection")

    # Mock get_supported_files to return our test PDF
    with (
        patch("pyrag.rag.get_supported_files") as mock_get_files,
        patch("pyrag.rag.DoclingLoader") as mock_loader,
    ):
        mock_get_files.return_value = [sample_pdf]

        # Mock DoclingLoader to return test documents
        test_docs = [
            Document(page_content="Indexed document content", metadata={"source": sample_pdf})
        ]
        mock_instance = Mock()
        mock_instance.load.return_value = test_docs
        mock_loader.return_value = mock_instance

        # Run indexing
        rag.index(sample_pdf)

        assert rag.vectorstore is not None
        assert rag.indexed_documents is not None
        assert len(rag.indexed_documents) == 1
        mock_get_files.assert_called_once_with(sample_pdf)


def test_rag_query_method(temp_milvus_uri):
    """Test RAG.query() method with complete query pipeline."""
    rag = RAG(milvus_uri=temp_milvus_uri, collection_name="test_collection", top_k=2)

    # Setup documents
    documents = [
        Document(
            page_content="Python programming language features", metadata={"source": "python.pdf"}
        ),
        Document(page_content="JavaScript web development", metadata={"source": "js.pdf"}),
        Document(page_content="Machine learning with Python", metadata={"source": "ml_python.pdf"}),
    ]

    rag.store(documents)

    # Query should automatically setup retriever
    results = rag.query("Python programming")

    assert rag.retriever is not None
    assert isinstance(results, list)
    assert len(results) <= 2  # Respects top_k
    assert all(isinstance(doc, Document) for doc in results)
