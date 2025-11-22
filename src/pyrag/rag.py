"""Main RAG pipeline implementation."""

import hashlib
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from .config import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXPORT_TYPE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MILVUS_URI,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_RANK_FUSION_CONSTANT,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_WEIGHT,
)
from .utils import get_supported_files


class RAG:
    """Main RAG pipeline for document processing and retrieval."""

    def __init__(
        self,
        export_type: ExportType = DEFAULT_EXPORT_TYPE,
        embed_model: str = DEFAULT_EMBED_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        milvus_uri: str = DEFAULT_MILVUS_URI,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ):
        self.export_type = export_type
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.milvus_uri = milvus_uri
        self.indexed_documents = None

        # Ensure directory exists for local file URIs only
        if not milvus_uri.startswith(("tcp://", "http://", "https://")):
            milvus_path = Path(milvus_uri)
            if milvus_path.parent != Path("."):
                milvus_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)

        self.vectorstore = self._create_milvus_vectorstore()
        self.retriever = None

    # Public methods
    def load(self, file_paths: list[str]):
        """Load documents using Docling loader."""
        loader = DoclingLoader(
            file_path=file_paths,
            export_type=self.export_type,
            chunker=HybridChunker(
                tokenizer=HuggingFaceTokenizer.from_pretrained(
                    model_name=self.embed_model,
                ),
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
            ),
        )

        documents = loader.load()

        return documents

    def chunk(self, documents):
        """Return pre-chunked documents."""
        return documents

    def embed(self, chunks):
        """Initialize embeddings and pass through chunks."""
        _ = self.embeddings
        return chunks

    def store(self, documents):
        """Set up or load existing Milvus vectorstore with upsert for deduplication."""
        # First, try to load existing documents if we don't have them
        if self.indexed_documents is None:
            self._load_existing_documents_catalog()

        # Add new documents to the catalog (avoid duplicates based on content hash)
        if self.indexed_documents is None:
            self.indexed_documents = documents
        else:
            # Merge new documents with existing catalog, avoiding duplicates
            existing_content_hashes = set()
            for doc in self.indexed_documents:
                content = doc.page_content
                existing_content_hashes.add(self._generate_content_id(content))

            # Add only new documents that aren't already in the catalog
            for doc in documents:
                content = doc.page_content
                doc_hash = self._generate_content_id(content)
                if doc_hash not in existing_content_hashes:
                    self.indexed_documents.append(doc)

        # Reset retriever to force refresh with updated document catalog
        self.retriever = None

        # Generate deterministic IDs based on content
        ids = []
        for doc in documents:
            content = doc.page_content
            doc_id = self._generate_content_id(content)
            ids.append(doc_id)

        # Check if we can use the existing vectorstore (collection exists)
        try:
            # Try to get collection info to verify it exists
            info = self.vectorstore.col.describe()
            if info:
                # Collection exists, use upsert to prevent duplicates
                self.vectorstore.upsert(ids=ids, documents=documents)
                return
        except Exception:
            # Collection doesn't exist, fall through to create new one
            pass

        # Create new vectorstore if it doesn't exist or couldn't connect
        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": self.milvus_uri},
            index_params={"index_type": "FLAT"},
            drop_old=False,
            ids=ids,  # Use our generated IDs
            enable_dynamic_field=True,  # Enable dynamic field for JSON metadata storage
        )

    def reset(self):
        """Reset/clear the vector storage by dropping the collection."""
        try:
            # Drop the collection if it exists
            if self.vectorstore is not None and hasattr(self.vectorstore, "col"):
                # Collection may be None if vectorstore failed to connect
                self.vectorstore.col.drop()

        except Exception:
            # Collection doesn't exist or connection failed, which is fine
            pass

        # Reset internal state
        self.vectorstore = None
        self.retriever = None
        self.indexed_documents = None

    def retrieve(self):
        """Initialize retriever from existing or current vectorstore."""
        # Load existing documents for BM25 if we don't have them yet
        if self.indexed_documents is None:
            self._load_existing_documents_catalog()

        if self.vectorstore is None:
            self.vectorstore = self._create_milvus_vectorstore()

        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # Use ensemble retriever with BM25 when documents are available
        if self.indexed_documents:
            # Use complete document catalog for BM25
            self.bm25_retriever = BM25Retriever.from_documents(self.indexed_documents)
            self.bm25_retriever.k = self.top_k

            # Use LangChain's built-in EnsembleRetriever with weighted rank fusion
            self.retriever = EnsembleRetriever(
                retrievers=[vector_retriever, self.bm25_retriever],
                weights=[DEFAULT_VECTOR_WEIGHT, DEFAULT_BM25_WEIGHT],
                c=DEFAULT_RANK_FUSION_CONSTANT,
            )
        else:
            # Fallback to vector retriever if no documents available
            self.retriever = vector_retriever

    def search(self, query: str):
        """Perform retrieval search."""
        results = self.retriever.invoke(query)
        # Ensure we return only top_k results (EnsembleRetriever might return more)
        return results[: self.top_k]

    def index(self, input_path: str):
        """Index documents using the RAG pipeline."""
        load_dotenv()
        file_paths = get_supported_files(input_path)

        documents = self.load(file_paths)
        chunks = self.chunk(documents)
        embedded_chunks = self.embed(chunks)
        self.store(embedded_chunks)

    def discover(self):
        """Discover all indexed documents in the collection."""
        # Load existing documents if we don't have them
        if self.indexed_documents is None:
            self._load_existing_documents_catalog()

        return self.indexed_documents or []

    def query(self, query_text: str):
        """Perform a search query on the stored documents."""
        # Always refresh retriever if not set (includes after new documents added)
        if self.retriever is None:
            self.retrieve()
        return self.search(query_text)

    # Private methods
    def _generate_content_id(self, content: str, source: str = "") -> str:
        """Generate a deterministic ID based on document content only for deduplication."""
        # Use only content for ID generation to avoid issues with metadata loss
        # during vectorstore reload causing duplicate detection failures
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_existing_documents_catalog(self):
        """Load existing documents catalog from vectorstore if available."""
        # Load documents from existing vectorstore
        existing_docs = self._load_documents_from_vectorstore()
        if existing_docs:
            self.indexed_documents = existing_docs

    def _load_documents_from_vectorstore(self):
        """Load all documents from existing vectorstore using proper Milvus query method."""
        try:
            # Access the underlying collection directly for querying
            if not hasattr(self.vectorstore, "col") or self.vectorstore.col is None:
                return None

            # Use the proper Milvus collection query method to get all documents
            # This is the recommended approach instead of dummy similarity search
            try:
                # With enable_dynamic_field=True, metadata is stored dynamically
                # We need to query for all fields using "*" to get dynamic metadata
                output_fields = ["*"]

                # Query all documents using a simple existence check
                results = self.vectorstore.col.query(
                    expr="pk != ''",  # Get all documents with non-empty primary keys
                    output_fields=output_fields,  # Get all fields including dynamic metadata
                )

                if not results:
                    return None

                # Convert Milvus query results back to LangChain Document objects
                documents = []
                for result in results:
                    # Extract text content
                    text_content = result.get("text", "")

                    # Build metadata from all fields except text, vector, and pk
                    metadata = {}
                    for key, value in result.items():
                        if key not in ["text", "vector", "pk"]:
                            metadata[key] = value

                    # Create Document object
                    doc = Document(page_content=text_content, metadata=metadata)
                    documents.append(doc)

                return documents

            except Exception:
                # If direct query fails, fallback gracefully
                return None

        except Exception:
            # If loading fails, return None - will fallback to vector-only retrieval
            return None

    def _create_milvus_vectorstore(self) -> Milvus:
        """Create a Milvus vectorstore instance with consistent configuration."""
        return Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": self.milvus_uri},
            enable_dynamic_field=True,  # Enable dynamic field for JSON metadata storage
        )
