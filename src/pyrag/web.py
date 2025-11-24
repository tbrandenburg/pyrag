"""Web interface for PyRAG using FastAPI."""

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .rag import RAG

# Default collection name (avoiding config.py import due to langchain_docling dependency)
DEFAULT_COLLECTION_NAME = "rag"

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

app = FastAPI(title="PyRAG API", description="Document indexing and search API")


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict[str, Any]


class SourceGroupResponse(BaseModel):
    source: str
    filenames: list[str]
    heading_count: int
    chunk_count: int
    total_chars: int
    status: str
    preview: str


class IndexRequest(BaseModel):
    path: str
    collection_name: str = DEFAULT_COLLECTION_NAME


class DiscoveryResponse(BaseModel):
    collection_name: str
    total_sources: int
    total_chunks: int
    content_types: dict[str, int]
    sources: list[str]
    filenames: list[str]
    total_content_length: int
    source_groups: list[SourceGroupResponse]


def _analyze_documents(documents: list) -> dict[str, Any]:
    """Analyze documents and group them by source for UI display."""

    def get_content_type(doc):
        dl_meta = doc.metadata.get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])
        if doc_items:
            return doc_items[0].get("label", "unknown")
        return "unknown"

    content_types = {}
    filenames: set[str] = set()
    total_content_length = 0
    grouped_sources: dict[str, dict[str, Any]] = {}
    status_priority = {"failed": 3, "processing": 2, "pending": 2, "indexed": 1}

    for doc in documents:
        metadata = doc.metadata or {}
        dl_meta = metadata.get("dl_meta", {})
        origin = dl_meta.get("origin", {})
        source = metadata.get("source", "Unknown source")
        filename = origin.get("filename")
        headings = dl_meta.get("headings", [])
        status_lower = (metadata.get("status") or "indexed").lower()

        content_type = get_content_type(doc)
        content_types[content_type] = content_types.get(content_type, 0) + 1

        if filename:
            filenames.add(filename)

        total_content_length += len(doc.page_content)

        group = grouped_sources.setdefault(
            source,
            {
                "filenames": set(),
                "headings": set(),
                "chunk_count": 0,
                "total_chars": 0,
                "status": "Indexed",
                "preview": doc.page_content,
            },
        )

        if filename:
            group["filenames"].add(filename)
        for heading in headings:
            group["headings"].add(heading)

        group["chunk_count"] += 1
        group["total_chars"] += len(doc.page_content)

        current_rank = status_priority.get(group["status"].lower(), 0)
        incoming_rank = status_priority.get(status_lower, 0)
        if incoming_rank >= current_rank:
            group["status"] = status_lower.title()

    source_groups = [
        SourceGroupResponse(
            source=source,
            filenames=sorted(data["filenames"]),
            heading_count=len(data["headings"]),
            chunk_count=data["chunk_count"],
            total_chars=data["total_chars"],
            status=data["status"],
            preview=data["preview"],
        )
        for source, data in sorted(grouped_sources.items())
    ]

    return {
        "content_types": content_types,
        "sources": [group.source for group in source_groups],
        "filenames": sorted(filenames),
        "total_content_length": total_content_length,
        "source_groups": source_groups,
        "total_sources": len(source_groups),
        "total_chunks": len(documents),
    }


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    collection_name: str = Query(DEFAULT_COLLECTION_NAME, description="Milvus collection name"),
):
    """Root endpoint showing discovery of all indexed documents."""
    try:
        data = discover_documents(collection_name)
        return templates.TemplateResponse(
            "index_new.html",
            {
                "request": request,
                "collection_name": data.collection_name,
                "total_sources": data.total_sources,
                "total_chunks": data.total_chunks,
                "content_types": data.content_types,
                "sources": data.sources,
                "filenames": data.filenames,
                "total_content_length": data.total_content_length,
                "source_groups": data.source_groups,
            },
        )
    except Exception as e:
        # Fallback to error template or simple HTML
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"


@app.post("/index")
def index_document(request: IndexRequest):
    """Index a document path or URL and return completion status."""
    try:
        rag = RAG(collection_name=request.collection_name)
        rag.index(request.path)
        return {"status": "finished"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def discover_documents(collection_name: str) -> DiscoveryResponse:
    """Discover and analyze all indexed documents in the collection."""
    try:
        rag = RAG(collection_name=collection_name)
        documents = rag.discover()

        if not documents:
            return DiscoveryResponse(
                collection_name=collection_name,
                total_sources=0,
                total_chunks=0,
                content_types={},
                sources=[],
                filenames=[],
                total_content_length=0,
                source_groups=[],
            )

        analysis = _analyze_documents(documents)

        return DiscoveryResponse(
            collection_name=collection_name,
            total_sources=analysis["total_sources"],
            total_chunks=analysis["total_chunks"],
            content_types=analysis["content_types"],
            sources=analysis["sources"],
            filenames=analysis["filenames"],
            total_content_length=analysis["total_content_length"],
            source_groups=analysis["source_groups"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    """Main function for console script entry point."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
