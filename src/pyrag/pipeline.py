"""Main RAG pipeline implementation."""

import warnings
from pathlib import Path

from docling.chunking import HybridChunker
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from .config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXPORT_TYPE,
    DEFAULT_TOP_K,
)
from .utils import get_supported_files


class RAGPipeline:
    """Main RAG pipeline for document processing and retrieval."""

    def __init__(
        self,
        export_type: ExportType = DEFAULT_EXPORT_TYPE,
        embed_model: str = DEFAULT_EMBED_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        storage_dir: str = "milvus_storage",
    ):
        # Suppress common warnings from external libraries
        warnings.filterwarnings("ignore", category=UserWarning, module="milvus_lite")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling.*")
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*")

        self.export_type = export_type
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.top_k = top_k
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.milvus_uri = str(self.storage_dir / "docling.db")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.vectorstore = None
        self.retriever = None

    def load(self, file_paths: list[str]):
        """Load documents using Docling loader."""
        loader = DoclingLoader(
            file_path=file_paths,
            export_type=self.export_type,
            chunker=HybridChunker(tokenizer=self.embed_model),
        )

        return loader.load()

    def chunk(self, documents):
        """Return pre-chunked documents."""
        return documents

    def embed(self, chunks):
        """Initialize embeddings and pass through chunks."""
        _ = self.embeddings
        return chunks

    def store(self, documents):
        """Set up or load existing Milvus vectorstore."""
        try:
            # Try opening existing Milvus store
            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.milvus_uri},
            )
            self.vectorstore.add_documents(documents)

        except Exception:
            self.vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.milvus_uri},
                index_params={"index_type": "FLAT"},
                drop_old=False,
            )

    def retrieve(self):
        """Initialize retriever."""
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def search(self, query: str):
        """Perform retrieval search."""
        return self.retriever.invoke(query)

    def run(self, input_path: str, query: str = "Which are the main AI models in Docling?"):
        """Run the complete RAG pipeline."""
        load_dotenv()
        file_paths = get_supported_files(input_path)

        documents = self.load(file_paths)
        chunks = self.chunk(documents)
        embedded_chunks = self.embed(chunks)
        self.store(embedded_chunks)
        self.retrieve()
        results = self.search(query)

        return results
