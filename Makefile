# PyRAG Makefile
# Provides common development tasks for the PyRAG project

.PHONY: help lint format test unit-test system-test qa build run clean install

# Default target
help:
	@echo "PyRAG Development Tasks"
	@echo "======================"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  format      Format code with ruff"
	@echo "  lint        Run ruff linter"
	@echo "  test        Run all tests"
	@echo "  unit-test   Run unit tests only"
	@echo "  system-test Run system tests only"
	@echo "  qa          Run all quality assurance tasks (lint + format + test)"
	@echo "  qa-quick    Run all quick quality assurance tasks (lint + format + unit-test)"
	@echo ""
	@echo "Build & Run:"
	@echo "  build     Build the package"
	@echo "  run       Run the CLI application"
	@echo ""
	@echo "Maintenance:"
	@echo "  install   Install/sync dependencies"
	@echo "  clean     Clean build artifacts and cache"
	@echo ""
	@echo "Example usage:"
	@echo "  make qa                    # Run all quality checks"
	@echo "  make run ARGS='search .'   # Run with arguments"

# Quality Assurance Tasks
format:
	@echo "🎨 Formatting code with ruff..."
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint:
	@echo "🔍 Running ruff linter..."
	uv run ruff check src/ tests/

test:
	@echo "🧪 Running all tests..."
	uv run pytest tests/ -v

unit-test:
	@echo "🔬 Running unit tests..."
	uv run pytest tests/test_unit.py -v

system-test:
	@echo "🏗️ Running system tests..."
	uv run pytest tests/test_system.py -v

qa: format lint test
	@echo "✅ All quality assurance tasks completed successfully!"

qa-quick: format lint unit-test
	@echo "✅ All quick quality assurance tasks completed successfully!"

# Build & Run Tasks
build:
	@echo "📦 Building the package..."
	uv build

run:
	@echo "🚀 Running PyRAG CLI..."
	@if [ -z "$(ARGS)" ]; then \
		echo "Usage: make run ARGS='/path/to/docs --query \"your question\"'"; \
		echo "Or: make run ARGS='--help'"; \
		uv run pyrag --help; \
	else \
		uv run pyrag $(ARGS); \
	fi

# Maintenance Tasks
install:
	@echo "⚙️ Installing/syncing dependencies..."
	uv sync --group dev

clean:
	@echo "🧹 Cleaning build artifacts and cache..."
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned up build artifacts and cache"