"""Command-line interface for PyRAG."""

import typer
from rich.console import Console

from .rag import RAG

app = typer.Typer(help="PyRAG - Docling-powered modular RAG CLI")
console = Console()


def _show_discovery_insights(console: Console, documents: list) -> None:
    """Display comprehensive metadata-based insights for discovered documents."""

    # Helper function to extract content type from Docling dl_meta
    def get_content_type(doc):
        dl_meta = doc.metadata.get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])
        if doc_items:
            return doc_items[0].get("label", "unknown")
        return "unknown"

    # Helper function to extract filename from Docling dl_meta
    def get_filename(doc):
        dl_meta = doc.metadata.get("dl_meta", {})
        origin = dl_meta.get("origin", {})
        return origin.get("filename")

    # Analyze content types
    content_types = {}
    sources = {}
    filenames = set()
    total_content_length = 0

    for doc in documents:
        # Content type analysis using Docling metadata
        content_type = get_content_type(doc)
        if content_type not in content_types:
            content_types[content_type] = []
        content_types[content_type].append(doc)

        # Source analysis
        source = doc.metadata.get("source", "Unknown")
        if source not in sources:
            sources[source] = {"docs": [], "content_types": set()}
        sources[source]["docs"].append(doc)
        sources[source]["content_types"].add(content_type)

        # Filename tracking using Docling metadata
        filename = get_filename(doc)
        if filename:
            filenames.add(filename)

        # Content length tracking
        total_content_length += len(doc.page_content)

    # Show content type summary
    console.print("\n[bold yellow]📊 Content Type Distribution:[/bold yellow]")
    for content_type, docs in sorted(content_types.items()):
        icon = (
            "📄"
            if content_type == "text"
            else "📊"
            if content_type == "table"
            else "💻"
            if content_type == "code"
            else "❓"
        )
        console.print(f"   {icon} [cyan]{content_type.title()}[/cyan]: {len(docs)} chunks")

    # Show file summary
    console.print("\n[bold yellow]📁 File Summary:[/bold yellow]")
    console.print(f"   • [cyan]{len(filenames)}[/cyan] unique files processed")
    console.print(f"   • [cyan]{len(sources)}[/cyan] unique sources indexed")
    console.print(f"   • [cyan]{total_content_length:,}[/cyan] total characters indexed")

    # Show sources with content type breakdown
    console.print("\n[bold yellow]📂 Sources by Content Type:[/bold yellow]")
    for source, info in sources.items():
        docs = info["docs"]
        content_breakdown = {}
        for doc in docs:
            ct = get_content_type(doc)
            content_breakdown[ct] = content_breakdown.get(ct, 0) + 1

        # Build content type summary string
        ct_summary = ", ".join(
            [f"{ct}({count})" for ct, count in sorted(content_breakdown.items())]
        )

        console.print(f"\n[cyan]📁 {source}[/cyan] ({len(docs)} chunks)")
        console.print(f"   Content: {ct_summary}")

        # Show sample content from each type
        seen_types = set()
        for doc in docs[:3]:  # Show up to 3 examples
            ct = get_content_type(doc)
            if ct not in seen_types:
                seen_types.add(ct)
                preview = doc.page_content[:150].strip().replace("\n", " ")
                if len(doc.page_content) > 150:
                    preview += "..."
                icon = (
                    "📄"
                    if ct == "text"
                    else "📊"
                    if ct == "table"
                    else "💻"
                    if ct == "code"
                    else "❓"
                )
                console.print(f"   {icon} [dim]{preview}[/dim]")

        if len(docs) > 3:
            console.print(f"   [dim]... and {len(docs) - 3} more chunks[/dim]")


@app.command()
def main_command(
    add_path: str = typer.Option(
        None, "--add", "-a", help="File, URL, or directory to process for indexing"
    ),
    query: str = typer.Option(
        None, "--query", "-q", help="Query to search for in indexed documents"
    ),
    discover: bool = typer.Option(
        False, "--discover", "-d", help="List all indexed documents in the collection"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top results to return"),
    collection_name: str = typer.Option("rag", "--collection", help="Milvus collection name"),
    reset: bool = typer.Option(
        False, "--reset", help="Reset/clear the vector storage before adding new documents"
    ),
):
    """Main PyRAG command - index documents or search, or both."""
    # Check if no action is provided
    if not add_path and not query and not discover:
        console.print("[bold blue]Nothing to do.[/bold blue]")
        console.print("Provide either:")
        console.print("  --add/-a <path>     to index documents")
        console.print("  --query/-q <text>   to search indexed documents")
        console.print("  --discover/-d       to list indexed documents")
        console.print("  or combine actions as needed")
        return

    rag = RAG(
        top_k=top_k,
        collection_name=collection_name,
    )

    try:
        # Reset vector storage if requested
        if reset:
            rag.reset()
            console.print(
                f"[bold yellow]Reset vector storage for collection:[/bold yellow] {collection_name}"
            )

        # Index documents if add_path is provided
        if add_path:
            rag.index(add_path)
            console.print(
                f"[bold green]Successfully indexed documents from:[/bold green] {add_path}"
            )
            console.print(f"[bold blue]Collection:[/bold blue] {collection_name}")

        # List indexed documents if discover is requested
        if discover:
            documents = rag.discover()
            if not documents:
                console.print(
                    f"[bold yellow]No indexed documents found in collection:[/bold yellow] "
                    f"{collection_name}"
                )
            else:
                console.print(
                    f"[bold green]Indexed Documents in collection:[/bold green] {collection_name}"
                )
                console.print(f"[bold blue]Found {len(documents)} documents[/bold blue]")

                # Analyze metadata for insights
                _show_discovery_insights(console, documents)

        # Search if query is provided
        if query:
            # Helper functions for extracting Docling metadata (same as in discovery)
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

            results = rag.query(query)
            console.print(f"[bold green]Search Results for:[/bold green] {query}")
            console.print(f"[bold blue]Found {len(results)} results[/bold blue]")

            # Display short insights from the results
            console.print("\n[bold yellow]Key Insights:[/bold yellow]")
            for i, result in enumerate(results[:5], 1):  # Show top 5 results
                # Get first 200 characters as preview
                preview = result.page_content[:200].strip()
                if len(result.page_content) > 200:
                    preview += "..."

                # Get metadata info using Docling metadata
                source = result.metadata.get("source", "Unknown")
                content_type = get_content_type(result)
                filename = get_filename(result)

                # Content type icon
                icon = (
                    "📄"
                    if content_type == "text"
                    else "📊"
                    if content_type == "table"
                    else "💻"
                    if content_type == "code"
                    else "❓"
                )

                console.print(f"\n[cyan]{i}. {icon}[/cyan] {preview}")

                # Show enhanced metadata
                if filename:
                    console.print(f"   [dim]📁 {filename} ({content_type})[/dim]")
                else:
                    console.print(f"   [dim]📁 {source} ({content_type})[/dim]")

            return results
        elif add_path:
            console.print(
                "[bold blue]Documents indexed successfully. Use --query to search them.[/bold blue]"
            )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
