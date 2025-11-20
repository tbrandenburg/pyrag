"""Main RAG pipeline implementation."""

import json
import os
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
from .utils import clip_text, get_supported_files


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

    def load_documents(self, file_paths: list[str]):
        """Load documents using Docling loader."""
        print("\n2) Loading documents using Docling...")

        loader = DoclingLoader(
            file_path=file_paths,
            export_type=self.export_type,
            chunker=HybridChunker(tokenizer=self.embed_model),
        )

        docs = loader.load()
        print(f"   Loaded {len(docs)} document chunks")
        return docs

    def setup_vectorstore(self, documents):
        """Set up or load existing Milvus vectorstore."""
        print(f"\n5) Loading/creating Milvus collection: {self.collection_name}")

        try:
            # Try opening existing Milvus store
            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.milvus_uri},
            )
            print("   ✔ Existing Milvus collection loaded")
            print("   → Appending new documents...")
            self.vectorstore.add_documents(documents)

        except Exception:
            print("   ✘ No existing collection found.")
            print("   → Creating a new Milvus collection")
            self.vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.milvus_uri},
                index_params={"index_type": "FLAT"},
                drop_old=False,
            )

        print("   ✔ Vectorstore ready")

    def setup_retriever(self):
        """Initialize retriever."""
        print(f"\n6) Initializing retriever (TOP_K={self.top_k})")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def search(self, query: str):
        """Perform retrieval search."""
        print("\n7) Searching retrieved chunks...")
        print(f"   Query: {query}")
        print("-" * 80)

        results = self.retriever.invoke(query)

        print(f"\nTop {self.top_k} Retrieved Chunks:\n")
        for i, d in enumerate(results, 1):
            print(f"--- Chunk {i} ---")
            print(clip_text(d.page_content, 350))
            
            # Pretty-print metadata
            metadata = {k: v for k, v in d.metadata.items() if k != "pk"}
            print("\nMetadata:")
            
            # Format key fields more readably
            if 'source' in metadata:
                print(f"  📄 Source: {metadata['source']}")
            
            if 'dl_meta' in metadata and isinstance(metadata['dl_meta'], dict):
                dl_meta = metadata['dl_meta']
                
                # Show document origin info
                if 'origin' in dl_meta:
                    origin = dl_meta['origin']
                    print(f"  📁 File: {origin.get('filename', 'Unknown')}")
                    print(f"  📊 Type: {origin.get('mimetype', 'Unknown')}")
                
                # Show headings/sections
                if 'headings' in dl_meta and dl_meta['headings']:
                    print(f"  📑 Section: {' → '.join(dl_meta['headings'])}")
                
                # Show content items summary
                if 'doc_items' in dl_meta and dl_meta['doc_items']:
                    items = dl_meta['doc_items']
                    content_types = [item.get('label', 'unknown') for item in items if 'label' in item]
                    if content_types:
                        type_counts = {}
                        for ct in content_types:
                            type_counts[ct] = type_counts.get(ct, 0) + 1
                        type_summary = ', '.join([f"{count} {ct}" for ct, count in type_counts.items()])
                        print(f"  📝 Content: {type_summary}")
                
                # Show page information
                if 'doc_items' in dl_meta and dl_meta['doc_items']:
                    pages = set()
                    for item in dl_meta['doc_items']:
                        if 'prov' in item:
                            for prov in item['prov']:
                                if 'page_no' in prov:
                                    pages.add(prov['page_no'])
                    if pages:
                        page_list = sorted(pages)
                        if len(page_list) == 1:
                            print(f"  📄 Page: {page_list[0]}")
                        else:
                            print(f"  📄 Pages: {', '.join(map(str, page_list))}")
            
            print()

        return results

    def run(self, input_path: str, query: str = "Which are the main AI models in Docling?"):
        """Run the complete RAG pipeline."""
        load_dotenv()

        print("=" * 80)
        print("Docling RAG — Retriever Only — Load/Create Milvus + Append")
        print("=" * 80)

        # 1. Collect supported files
        print(f"\n1) Scanning for supported files under: {input_path}")
        file_paths = get_supported_files(input_path)

        print(f"   Found {len(file_paths)} supported files:")
        for f in file_paths:
            print(f"     • {f}")

        # 2-3. Load and process documents
        docs = self.load_documents(file_paths)
        print(f"\n3) Using {len(docs)} Docling-generated chunks")

        # 4. Initialize embeddings (already done in __init__)
        print(f"\n4) Initializing embeddings: {self.embed_model}")

        # 5. Setup vectorstore
        self.setup_vectorstore(docs)

        # 6. Setup retriever
        self.setup_retriever()

        # 7. Perform search
        results = self.search(query)

        print("=" * 80)
        print("RAG retrieval (no LLM) completed successfully.")
        print("=" * 80)

        return results
