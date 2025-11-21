"""Command-line interface for PyRAG."""

import typer
from rich.console import Console

from .pipeline import RAGPipeline

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
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top results to return"),
    collection_name: str = typer.Option(
        "docling_retriever_demo", "--collection", help="Milvus collection name"
    ),
):
    """Main PyRAG command - index documents or search, or both."""
    # Check if neither add_path nor query is provided
    if not add_path and not query:
        console.print("[bold blue]Nothing to do.[/bold blue]")
        console.print("Provide either:")
        console.print("  --add/-a <path>   to index documents")
        console.print("  --query/-q <text> to search indexed documents")
        console.print("  or both to index and then search")
        return

    pipeline = RAGPipeline(
        top_k=top_k,
        collection_name=collection_name,
    )

    try:
        # Index documents if add_path is provided
        if add_path:
            pipeline.index(add_path)
            console.print(
                f"[bold green]Successfully indexed documents from:[/bold green] {add_path}"
            )
            console.print(f"[bold blue]Collection:[/bold blue] {collection_name}")

        # Search if query is provided
        if query:
            results = pipeline.query(query)
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
