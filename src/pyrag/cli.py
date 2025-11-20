"""Command-line interface for PyRAG."""

import sys

import typer
from rich.console import Console

from .pipeline import RAGPipeline

app = typer.Typer(help="PyRAG - Docling-powered modular RAG CLI")
console = Console()


@app.command()
def search(
    input_path: str = typer.Argument(..., help="File, URL, or directory to process"),
    query: str = typer.Option(
        "Which are the main AI models in Docling?",
        "--query",
        "-q",
        help="Query to search for in the documents",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top results to return"),
    collection_name: str = typer.Option(
        "docling_retriever_demo", "--collection", help="Milvus collection name"
    ),
):
    """Search for information in documents using RAG."""
    pipeline = RAGPipeline(
        top_k=top_k,
        collection_name=collection_name,
    )

    try:
        results = pipeline.run(input_path, query)
        return results
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e


def main():
    """Main entry point for the CLI."""
    # Legacy support for the original script interface
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        # Original single-argument interface
        input_path = sys.argv[1]
        pipeline = RAGPipeline()
        try:
            pipeline.run(input_path)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)
    else:
        # Use typer CLI
        app()


if __name__ == "__main__":
    main()
