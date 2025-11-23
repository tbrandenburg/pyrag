"""Web interface for PyRAG using FastAPI."""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from .rag import RAG

# Default collection name (avoiding config.py import due to langchain_docling dependency)
DEFAULT_COLLECTION_NAME = "rag"

app = FastAPI(title="PyRAG API", description="Document indexing and search API")


class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class DiscoveryResponse(BaseModel):
    collection_name: str
    total_documents: int
    content_types: Dict[str, int]
    sources: List[str]
    filenames: List[str]
    total_content_length: int
    documents: List[DocumentResponse]


def _analyze_documents(documents: List) -> Dict[str, Any]:
    """Analyze documents for insights similar to CLI discover command."""
    
    def get_content_type(doc):
        dl_meta = doc.metadata.get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])
        if doc_items:
            return doc_items[0].get("label", "unknown")
        return "unknown"

    def get_filename(doc):
        dl_meta = doc.metadata.get("dl_meta", {})
        origin = dl_meta.get("origin", {})
        return origin.get("filename")

    content_types = {}
    sources = set()
    filenames = set()
    total_content_length = 0

    for doc in documents:
        # Content type analysis
        content_type = get_content_type(doc)
        content_types[content_type] = content_types.get(content_type, 0) + 1

        # Source analysis
        source = doc.metadata.get("source", "Unknown")
        sources.add(source)

        # Filename tracking
        filename = get_filename(doc)
        if filename:
            filenames.add(filename)

        # Content length tracking
        total_content_length += len(doc.page_content)

    return {
        "content_types": content_types,
        "sources": list(sources),
        "filenames": list(filenames),
        "total_content_length": total_content_length,
    }


@app.get("/", response_model=DiscoveryResponse)
def index(
    collection_name: str = Query(DEFAULT_COLLECTION_NAME, description="Milvus collection name")
):
    """Root endpoint showing discovery of all indexed documents."""
    return discover_documents(collection_name=collection_name)


def discover_documents(collection_name: str) -> DiscoveryResponse:
    """Discover and analyze all indexed documents in the collection."""
    try:
        rag = RAG(collection_name=collection_name)
        documents = rag.discover()
        
        if not documents:
            return DiscoveryResponse(
                collection_name=collection_name,
                total_documents=0,
                content_types={},
                sources=[],
                filenames=[],
                total_content_length=0,
                documents=[],
            )

        analysis = _analyze_documents(documents)
        
        return DiscoveryResponse(
            collection_name=collection_name,
            total_documents=len(documents),
            content_types=analysis["content_types"],
            sources=analysis["sources"],
            filenames=analysis["filenames"],
            total_content_length=analysis["total_content_length"],
            documents=[
                DocumentResponse(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in documents
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main function for console script entry point."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()