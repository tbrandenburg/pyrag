"""Main RAG pipeline implementation."""

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

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
from .storage import BaseVectorStorage, MilvusStorage
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

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)

        self.storage: BaseVectorStorage = MilvusStorage(
            uri=self.milvus_uri, embeddings=self.embeddings
        )
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
        """Persist documents using the configured vector storage backend."""
        if self.indexed_documents is None:
            self._load_existing_documents_catalog()

        self.storage.insert_documents(self.collection_name, documents)
        self.indexed_documents = self.storage.get_documents(self.collection_name)
        self.retriever = None

    def reset(self):
        """Reset/clear the vector storage by dropping the collection."""
        self.storage.drop(self.collection_name)
        self.retriever = None
        self.indexed_documents = None

    def retrieve(self):
        """Initialize retriever from existing or current vectorstore."""
        # Load existing documents for BM25 if we don't have them yet
        if self.indexed_documents is None:
            self._load_existing_documents_catalog()

        vector_retriever = self.storage.get_retriever(self.collection_name)
        vector_retriever.search_kwargs = {"k": self.top_k}

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
    def _load_existing_documents_catalog(self):
        """Load existing documents catalog from vectorstore if available."""
        existing_docs = self.storage.get_documents(self.collection_name)
        if existing_docs:
            self.indexed_documents = existing_docs
