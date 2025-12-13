"""Minimal MCP interface for PyRAG tools."""

import typer
from fastmcp import FastMCP

from .config import DEFAULT_COLLECTION_NAME, DEFAULT_TOP_K
from .rag import RAG

mcp = FastMCP("PyRAG 🤖")

_rag_cache: dict[str, RAG] = {}


def _get_rag(top_k: int = DEFAULT_TOP_K) -> RAG:
    """Get (or create) the cached RAG instance."""

    rag = _rag_cache.get(DEFAULT_COLLECTION_NAME)
    if rag is None:
        rag = RAG(collection_name=DEFAULT_COLLECTION_NAME, top_k=top_k)
        _rag_cache[DEFAULT_COLLECTION_NAME] = rag
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
def add_doc(path: str) -> dict[str, str]:
    """Index a file, URL, or directory."""

    rag = _get_rag()
    rag.index(path)
    return {"ok": "indexed", "source": path}


@mcp.tool
def reset() -> dict[str, str]:
    """Drop the collection and clear cached state."""

    rag = _get_rag()
    rag.reset()
    return {"ok": "reset"}


@mcp.tool
def discover(limit: int = 20) -> dict[str, list[dict[str, str]]]:
    """List a compact summary of indexed documents."""

    rag = _get_rag()
    documents = rag.discover()
    summaries = [_document_summary(doc) for doc in documents[:limit]]
    return {
        "count": len(documents),
        "docs": summaries,
    }


@mcp.tool
def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, list[dict[str, str]]]:
    """Retrieve short answers for a query from the collection."""

    rag = _get_rag(top_k=top_k)
    results = rag.query(query)
    summaries = [_document_summary(doc) for doc in results[:top_k]]
    return {
        "query": query,
        "results": summaries,
    }


def _serve(
    transport: str = typer.Option(
        "stdio",
        "--type",
        "-t",
        case_sensitive=False,
        help="Transport type for the MCP server (stdio or http)",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        min=1,
        help="Port to bind when using the HTTP transport",
    ),
) -> None:
    """Entry point for launching the MCP server."""

    transport_name = transport.lower()
    if transport_name not in {"stdio", "http"}:
        raise typer.BadParameter("Transport must be either 'stdio' or 'http'")

    if transport_name == "stdio":
        mcp.run(transport=transport_name)
        return

    mcp.run(transport=transport_name, port=port)


def main() -> None:
    """Entry point for launching the MCP server."""

    typer.run(_serve)


if __name__ == "__main__":
    main()
