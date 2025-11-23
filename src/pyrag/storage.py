from __future__ import annotations

import contextlib
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document
from langchain_milvus import Milvus


class BaseVectorStorage(ABC):
    """
    Abstract base class that defines the interface for a vector storage backend.

    A concrete implementation (e.g., using Milvus via langchain-milvus)
    must implement all abstract methods below.

    Responsibilities covered:
    - Collection CRUD (create via insert, read, delete, list, check existence)
    - Document CRUD (insert/upsert, read by id, delete by id)
    - Reset/wipe entire backend state
    """

    # -----------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------

    @abstractmethod
    def __init__(self, uri: str, collection_name: str, embeddings=None):
        """
        Initialize the storage backend.

        Parameters
        ----------
        uri : str
            Connection URI or address/path to the underlying vector storage system.
        collection_name : str
            Name of the collection to work with.
        embeddings : Any, optional
            Embedding function to use for vectorization.
        """

    # -----------------------------------------------------------
    # Collection-Level CRUD
    # -----------------------------------------------------------

    @abstractmethod
    def drop(self, collection: str):
        """
        Drop (delete) an entire collection.

        Parameters
        ----------
        collection : str
            Name of the collection to remove.
        """

    @abstractmethod
    def get_collections(self) -> list[str]:
        """
        Return a list of all collection names.

        Returns
        -------
        List[str]
            List of collection names known to the backend.
        """

    @abstractmethod
    def has_collection(self, collection: str) -> bool:
        """
        Check whether a given collection exists.

        Parameters
        ----------
        collection : str
            Collection name to check.

        Returns
        -------
        bool
            True if the collection exists, otherwise False.
        """

    # -----------------------------------------------------------
    # Document Retrieval
    # -----------------------------------------------------------

    @abstractmethod
    def get_vectorstore(self, collection: str):
        """
        Return the underlying vectorstore instance for a collection.

        Parameters
        ----------
        collection : str
            Name of the collection.

        Returns
        -------
        Any | None
            The backing vectorstore or ``None`` if unavailable.
        """

    @abstractmethod
    def get_retriever(self, collection: str):
        """
        Return a LangChain-style retriever for querying the collection.

        Parameters
        ----------
        collection : str
            Name of the collection from which to retrieve documents.

        Returns
        -------
        Any
            A LangChain VectorRetriever.
        """

    @abstractmethod
    def get_documents(self, collection: str):
        """
        Return all documents from a collection.

        Parameters
        ----------
        collection : str

        Returns
        -------
        List
            List of LangChain Document objects.
        """

    @abstractmethod
    def get_document_by_id(self, collection: str, doc_id: str):
        """
        Fetch a single document by its deterministic ID.

        Parameters
        ----------
        collection : str
            Name of the collection.

        doc_id : str
            Deterministic document ID (sha256 hash of the content).

        Returns
        -------
        Document or None
        """

    # -----------------------------------------------------------
    # Document Insertion / Deletion
    # -----------------------------------------------------------

    @abstractmethod
    def insert_documents(self, collection: str, docs: list):
        """
        Insert documents into a collection.

        Expected Behavior:
        - Documents are internally hashed using sha256(page_content)
        - Deduplication based on that hash
        - Automatic collection creation if missing
        - Upsert semantics in the vector storage backend

        Parameters
        ----------
        collection : str
            Target collection name.

        docs : List
            List of LangChain Document objects.
        """

    @abstractmethod
    def delete_document_by_id(self, collection: str, doc_id: str):
        """
        Delete a single document by its deterministic ID.

        Parameters
        ----------
        collection : str

        doc_id : str
            Document ID (sha256 hash of its content).
        """

    # -----------------------------------------------------------
    # Reset / Wipe backend
    # -----------------------------------------------------------

    @abstractmethod
    def reset(self):
        """
        Reset/wipe the entire backend.

        Intended Usage:
        - For test environments
        - For clearing all collections
        - For restarting with a clean slate

        NOTE: Real implementations may choose:
              - Drop all collections
              - Recreate database/session
        """


class MilvusStorage(BaseVectorStorage):
    """Milvus-backed implementation of ``BaseVectorStorage``."""

    def __init__(self, uri: str, collection_name: str, embeddings=None):
        self.uri = uri
        self.embeddings = embeddings
        self.collection_name = collection_name

        if not uri.startswith(("tcp://", "http://", "https://")):
            milvus_path = Path(uri)
            if milvus_path.parent != Path("."):
                milvus_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize vectorstore immediately
        self.vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": self.uri},
            enable_dynamic_field=True,
        )

    def drop(self, collection: str):
        if self.vectorstore and self.vectorstore.col and self.vectorstore.col.name == collection:
            self.vectorstore.col.drop()

    def get_collections(self) -> list[str]:
        # Collections listing is not supported without direct pymilvus access
        # Return empty list for now as this is an unsupported feature
        return []

    def has_collection(self, collection: str) -> bool:
        try:
            return bool(self.vectorstore.col.describe())
        except Exception:
            return False

    def get_vectorstore(self, collection: str):
        return self.vectorstore

    def get_retriever(self, collection: str):
        return self.vectorstore.as_retriever()

    def get_documents(self, collection: str):
        return self._load_documents_from_vectorstore(self.vectorstore)

    def get_document_by_id(self, collection: str, doc_id: str):
        try:
            results = self.vectorstore.col.query(expr=f"pk == '{doc_id}'", output_fields=["*"])
        except Exception:
            return None

        if not results:
            return None

        return self._convert_result_to_document(results[0])

    def insert_documents(self, collection: str, docs: list):
        if not docs:
            return

        existing_docs = self.get_documents(collection) or []
        existing_hashes = {self._generate_content_id(doc.page_content) for doc in existing_docs}

        new_docs: list[Document] = []
        ids: list[str] = []
        for doc in docs:
            doc_hash = self._generate_content_id(doc.page_content)
            if doc_hash not in existing_hashes:
                new_docs.append(doc)
                ids.append(doc_hash)
                existing_hashes.add(doc_hash)

        if not new_docs:
            return

        try:
            if self.vectorstore.col.describe():
                self.vectorstore.upsert(ids=ids, documents=new_docs)
                return
        except Exception:
            pass

        # Create new vectorstore if upsert failed
        vectorstore = Milvus.from_documents(
            documents=new_docs,
            embedding=self.embeddings,
            collection_name=collection,
            connection_args={"uri": self.uri},
            index_params={"index_type": "FLAT"},
            drop_old=False,
            ids=ids,
            enable_dynamic_field=True,
        )
        self.vectorstore = vectorstore

    def delete_document_by_id(self, collection: str, doc_id: str):
        with contextlib.suppress(Exception):
            self.vectorstore.col.delete(expr=f"pk in ['{doc_id}']")

    def reset(self):
        if self.vectorstore:
            self.vectorstore.col.drop()

    def _load_documents_from_vectorstore(self, vectorstore: Milvus | None):
        if vectorstore is None or not hasattr(vectorstore, "col") or vectorstore.col is None:
            return None

        try:
            results = vectorstore.col.query(expr="pk != ''", output_fields=["*"])
        except Exception:
            return None

        if not results:
            return None

        return [self._convert_result_to_document(result) for result in results]

    def _convert_result_to_document(self, result: dict):
        text_content = result.get("text", "")
        metadata = {
            key: value for key, value in result.items() if key not in {"text", "vector", "pk"}
        }
        return Document(page_content=text_content, metadata=metadata)

    def _generate_content_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
