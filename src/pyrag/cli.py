"""Command-line interface for PyRAG."""

import typer
from rich.console import Console

from .rag import RAG

app = typer.Typer(help="PyRAG - Docling-powered modular RAG CLI")
console = Console()


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

                # Group documents by source for better organization
                sources = {}
                for doc in documents:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in sources:
                        sources[source] = []
                    sources[source].append(doc)

                console.print("\n[bold yellow]Documents by Source:[/bold yellow]")
                for source, docs in sources.items():
                    console.print(f"\n[cyan]📁 {source}[/cyan] ({len(docs)} chunks)")
                    # Show first few chunks with preview
                    for i, doc in enumerate(docs[:3], 1):
                        preview = doc.page_content[:150].strip().replace("\n", " ")
                        if len(doc.page_content) > 150:
                            preview += "..."
                        console.print(f"   [dim]{i}. {preview}[/dim]")

                    if len(docs) > 3:
                        console.print(f"   [dim]... and {len(docs) - 3} more chunks[/dim]")

        # Search if query is provided
        if query:
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

                # Get source info if available
                source = result.metadata.get("source", "Unknown")
                console.print(f"\n[cyan]{i}.[/cyan] {preview}")
                console.print(f"   [dim]Source: {source}[/dim]")

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
