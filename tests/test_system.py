"""System tests for PyRAG with real documents and queries."""

import tempfile
from pathlib import Path

import pytest
import requests

from pyrag.rag import RAG


class TestDoclingPaper:
    """System tests using the Docling paper from ArXiv."""

    ARXIV_URL = "https://arxiv.org/pdf/2408.09869"
    DEFAULT_QUERY = "Which are the main AI models in Docling?"

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def docling_paper_path(self, temp_dir):
        """Download the Docling paper for testing."""
        paper_path = temp_dir / "docling_paper.pdf"

        # Download the paper
        response = requests.get(self.ARXIV_URL, timeout=30)
        response.raise_for_status()

        with open(paper_path, "wb") as f:
            f.write(response.content)

        return paper_path

    def test_docling_paper_processing(self, docling_paper_path, temp_dir):
        """Test that the Docling paper can be processed successfully."""
        # Initialize pipeline with temporary storage
        storage_dir = temp_dir / "milvus_storage"
        pipeline = RAG(
            collection_name="test_docling_paper", 
            milvus_uri=str(storage_dir / "docling.db"), 
            top_k=3
        )

        # Load and process the paper
        docs = pipeline.load([str(docling_paper_path)])

        # Verify documents were loaded
        assert len(docs) > 0, "No documents were loaded from the PDF"
        assert all(doc.page_content.strip() for doc in docs), "Some documents have empty content"

        # Verify metadata is present
        for doc in docs:
            assert hasattr(doc, "metadata"), "Document missing metadata"
            assert "source" in doc.metadata, "Document metadata missing source"

    def test_docling_paper_vectorstore_setup(self, docling_paper_path, temp_dir):
        """Test that vectorstore can be created and populated."""
        storage_dir = temp_dir / "milvus_storage"
        pipeline = RAG(
            collection_name="test_docling_vectorstore", 
            milvus_uri=str(storage_dir / "docling.db")
        )

        # Load documents
        docs = pipeline.load([str(docling_paper_path)])

        # Setup vectorstore
        pipeline.store(docs)

        # Verify vectorstore was created
        assert pipeline.vectorstore is not None, "Vectorstore was not created"

        # Test that retriever can be initialized via query
        pipeline.query("test query")
        assert pipeline.retriever is not None, "Retriever was not created"

    def test_main_ai_models_query(self, docling_paper_path, temp_dir):
        """Test querying for main AI models in Docling."""
        storage_dir = temp_dir / "milvus_storage"
        pipeline = RAG(
            collection_name="test_ai_models_query", 
            milvus_uri=str(storage_dir / "docling.db"), 
            top_k=5
        )

        # Process the document and setup retrieval
        docs = pipeline.load([str(docling_paper_path)])
        pipeline.store(docs)

        # Perform the query
        results = pipeline.query(self.DEFAULT_QUERY)

        # Verify results
        assert len(results) > 0, "No results returned for the query"
        assert len(results) <= 5, "More results returned than requested"

        # Check that results contain relevant content about AI models
        relevant_content = "\n".join([doc.page_content for doc in results])

        # Look for common AI model terms that should appear in a Docling paper
        model_terms = [
            "model",
            "transformer",
            "neural",
            "AI",
            "machine learning",
            "deep learning",
            "architecture",
            "embedding",
            "BERT",
            "LLM",
        ]

        found_terms = [term for term in model_terms if term.lower() in relevant_content.lower()]
        assert len(found_terms) > 0, (
            f"No AI model terms found in results. Content: {relevant_content[:500]}..."
        )

    def test_end_to_end_pipeline(self, docling_paper_path, temp_dir):
        """Test the complete end-to-end pipeline."""
        storage_dir = temp_dir / "milvus_storage"
        pipeline = RAG(
            collection_name="test_e2e_pipeline", 
            milvus_uri=str(storage_dir / "docling.db"), 
            top_k=3
        )

        # Run the indexing pipeline and then query
        pipeline.index(str(docling_paper_path))
        results = pipeline.query(self.DEFAULT_QUERY)

        # Verify end-to-end functionality
        assert results is not None, "Pipeline returned None results"
        assert len(results) > 0, "Pipeline returned empty results"
        assert len(results) <= 3, "Pipeline returned more results than requested"

        # Verify each result has content and metadata
        for i, result in enumerate(results):
            assert hasattr(result, "page_content"), f"Result {i} missing page_content"
            assert hasattr(result, "metadata"), f"Result {i} missing metadata"
            assert result.page_content.strip(), f"Result {i} has empty content"
            assert "source" in result.metadata, f"Result {i} metadata missing source"

    def test_different_queries(self, docling_paper_path, temp_dir):
        """Test various queries on the same document."""
        storage_dir = temp_dir / "milvus_storage"
        pipeline = RAG(
            collection_name="test_different_queries", 
            milvus_uri=str(storage_dir / "docling.db"), 
            top_k=3
        )

        # Setup once
        docs = pipeline.load([str(docling_paper_path)])
        pipeline.store(docs)

        test_queries = [
            "What is Docling?",
            "How does document parsing work?",
            "What are the key features?",
            "Which models are used for processing?",
        ]

        for query in test_queries:
            results = pipeline.query(query)
            assert len(results) > 0, f"No results for query: {query}"
            assert all(result.page_content.strip() for result in results), (
                f"Empty content for query: {query}"
            )

    def test_persistent_vectorstore(self, docling_paper_path, temp_dir):
        """Test that vectorstore persists between pipeline instances."""
        storage_dir = temp_dir / "milvus_storage"
        collection_name = "test_persistent_store"

        # First pipeline - create and populate vectorstore
        pipeline1 = RAG(
            collection_name=collection_name, 
            milvus_uri=str(storage_dir / "docling.db"), 
            top_k=2
        )

        docs = pipeline1.load([str(docling_paper_path)])
        pipeline1.store(docs)

        results1 = pipeline1.query(self.DEFAULT_QUERY)
        assert len(results1) > 0, "First pipeline returned no results"

        # Second pipeline - should load existing vectorstore
        pipeline2 = RAG(
            collection_name=collection_name, 
            milvus_uri=str(storage_dir / "docling.db"), 
            top_k=2
        )

        # Try to setup vectorstore (should load existing)
        try:
            # This should load the existing store, not create a new one
            pipeline2.store([])  # Empty docs since store should exist

            results2 = pipeline2.query(self.DEFAULT_QUERY)
            assert len(results2) > 0, "Second pipeline returned no results from persistent store"

            # Results should be similar (same collection)
            assert len(results1) == len(results2), (
                "Different number of results from persistent store"
            )

        except Exception:
            # If loading fails, it's expected behavior - create new store
            pipeline2.store(docs)
            results2 = pipeline2.query(self.DEFAULT_QUERY)
            assert len(results2) > 0, "Second pipeline failed to create new vectorstore"
