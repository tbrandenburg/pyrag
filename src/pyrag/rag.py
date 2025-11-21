"""Main RAG pipeline implementation."""

import hashlib
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from .config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXPORT_TYPE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MILVUS_URI,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TOP_K,
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
        
        # Ensure directory exists for local file URIs only
        if not (milvus_uri.startswith('tcp://') or milvus_uri.startswith('http://') or milvus_uri.startswith('https://')):
            milvus_path = Path(milvus_uri)
            if milvus_path.parent != Path('.'):
                milvus_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)

        self.vectorstore = None
        self.retriever = None

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

        return loader.load()

    def chunk(self, documents):
        """Return pre-chunked documents."""
        return documents

    def embed(self, chunks):
        """Initialize embeddings and pass through chunks."""
        _ = self.embeddings
        return chunks

    def _generate_content_id(self, content: str, source: str = "") -> str:
        """Generate a deterministic ID based on document content and source."""
        # Combine content and source for ID generation
        combined = f"{source}:{content}"
        # Use SHA-256 hash to create a deterministic ID
        return hashlib.sha256(combined.encode()).hexdigest()

    def store(self, documents):
        """Set up or load existing Milvus vectorstore with upsert for deduplication."""
        # Generate deterministic IDs based on content
        ids = []
        for doc in documents:
            content = doc.page_content
            source = doc.metadata.get("source", "")
            doc_id = self._generate_content_id(content, source)
            ids.append(doc_id)

        try:
            # Try opening existing Milvus store
            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.milvus_uri},
            )
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

        except Exception:
            # Connection failed, will create new vectorstore below
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
        )

    def retrieve(self):
        """Initialize retriever from existing or current vectorstore."""
        if self.vectorstore is None:
            # Try to load existing vectorstore
            try:
                self.vectorstore = Milvus(
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args={"uri": self.milvus_uri},
                )
            except Exception as e:
                raise ValueError(
                    f"No existing vectorstore found and none created. "
                    f"Run indexing first. Error: {e}"
                ) from e

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def search(self, query: str):
        """Perform retrieval search."""
        return self.retriever.invoke(query)

    def index(self, input_path: str):
        """Index documents using the RAG pipeline."""
        load_dotenv()
        file_paths = get_supported_files(input_path)

        documents = self.load(file_paths)
        chunks = self.chunk(documents)
        embedded_chunks = self.embed(chunks)
        self.store(embedded_chunks)

    def query(self, query_text: str):
        """Perform a search query on the stored documents."""
        if self.retriever is None:
            self.retrieve()
        return self.search(query_text)
