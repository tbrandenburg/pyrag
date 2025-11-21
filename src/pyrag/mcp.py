"""Minimal MCP interface for PyRAG tools."""


from fastmcp import FastMCP

from .config import DEFAULT_COLLECTION_NAME, DEFAULT_TOP_K
from .rag import RAG

mcp = FastMCP("PyRAG 🤖")

_rag_cache: dict[str, RAG] = {}


def _get_rag(collection: str = DEFAULT_COLLECTION_NAME, top_k: int = DEFAULT_TOP_K) -> RAG:
    """Get (or create) a cached RAG instance for the collection."""

    rag = _rag_cache.get(collection)
    if rag is None:
        rag = RAG(collection_name=collection, top_k=top_k)
        _rag_cache[collection] = rag
    else:
        rag.top_k = top_k
    return rag


def _preview(text: str, max_chars: int = 200) -> str:
    """Return a compact single-line preview of text."""

    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}…"


def _content_type(document) -> str:
    dl_meta = document.metadata.get("dl_meta", {})
    doc_items = dl_meta.get("doc_items", [])
    if doc_items:
        return doc_items[0].get("label", "unknown")
    return "unknown"


def _document_summary(document) -> dict[str, str]:
    """Convert a Document into a lightweight summary for MCP responses."""

    source = document.metadata.get("source", "unknown")
    filename = document.metadata.get("dl_meta", {}).get("origin", {}).get("filename")
    return {
        "source": filename or source,
        "type": _content_type(document),
        "text": _preview(document.page_content),
    }


@mcp.tool
def add_doc(path: str, collection: str = DEFAULT_COLLECTION_NAME) -> dict[str, str]:
    """Index a file, URL, or directory into the collection."""

    rag = _get_rag(collection)
    rag.index(path)
    return {
        "ok": "indexed",
        "collection": collection,
        "source": path,
    }


@mcp.tool
def reset(collection: str = DEFAULT_COLLECTION_NAME) -> dict[str, str]:
    """Drop the collection and clear cached state."""

    rag = _get_rag(collection)
    rag.reset()
    return {"ok": "reset", "collection": collection}


@mcp.tool
def discover(
    collection: str = DEFAULT_COLLECTION_NAME, limit: int = 20
) -> dict[str, list[dict[str, str]]]:
    """List a compact summary of indexed documents."""

    rag = _get_rag(collection)
    documents = rag.discover()
    summaries = [_document_summary(doc) for doc in documents[:limit]]
    return {
        "collection": collection,
        "count": len(documents),
        "docs": summaries,
    }


@mcp.tool
def search(
    query: str,
    collection: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, list[dict[str, str]]]:
    """Retrieve short answers for a query from the collection."""

    rag = _get_rag(collection, top_k=top_k)
    results = rag.query(query)
    summaries = [_document_summary(doc) for doc in results[:top_k]]
    return {
        "collection": collection,
        "query": query,
        "results": summaries,
    }


def main() -> None:
    """Entry point for launching the MCP server."""

    mcp.run()


if __name__ == "__main__":
    main()
