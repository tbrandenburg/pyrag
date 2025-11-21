# RAG Pipeline

## Project Structure

The agents are organized within the following directory structure:

```
pyrag/
├── src/pyrag/              # Core agent implementations
│   ├── pipeline.py         # RAG pipeline coordinator
│   ├── cli.py              # CLI interface
│   ├── config.py           # Configuration
│   └── utils.py            # Utilities
├── tests/                  # Testing framework
├── scripts/                # Scripts
├── .github/workflows/      # CI/CD automation agents
│   └── test.yml            # GitHub Actions test agent
└── Makefile                # Development agent orchestration
```

## Development Workflows

### Make Targets for Agent Management

The Makefile provides automated workflows for managing agents:

- **`make qa`**: Run all quality assurance agents (format + lint + test)
- **`make lint`**: Execute ruff linting agent
- **`make format`**: Run code formatting and auto-fix agent
- **`make test`**: Execute full test agent suite
- **`make build`**: Package building agent
- **`make run`**: CLI agent execution with arguments
- **`make install`**: Dependency management agent
- **`make clean`**: Cleanup and maintenance agent

### Development Protocol

**⚠️ IMPORTANT: Always run `make qa` after making any changes to the codebase.**

This ensures:
- 🎨 Code is properly formatted
- 🔍 All linting rules are satisfied
- 🧪 Tests pass and functionality is verified
- 📦 The codebase remains in deployable state

**⚠️ IMPORTANT: Never suppress warnings.**

This ensures:
- We do not ignore severe issues
- We stay future-proof by considering deprecated features

### Example Usage

```bash
# REQUIRED: Run after any code changes
make qa

# Execute RAG pipeline agent with specific query
make run ARGS="https://arxiv.org/pdf/2408.09869 --query 'AI models'"

# Test agent functionality
make test

# Clean agent artifacts
make clean
```

## Technical Notes

### Token Length Warnings
The PyRAG pipeline may show warnings like:
```
Token indices sequence length is longer than the specified maximum sequence length for this model (584 > 512)
```

**This is expected behavior** according to Docling documentation. The `HybridChunker` triggers these warnings during token counting (not actual processing) to assess chunk sizes before splitting. The actual output chunks respect the configured limits.

### LangChain Integration Best Practices
- **Embedding/Tokenizer Compatibility**: Use the same model for both `HuggingFaceEmbeddings` and `HuggingFaceTokenizer.from_pretrained()` to ensure tokenization consistency with the embedding model
- **Supported Models**: All sentence-transformers models support this approach as they share the same underlying tokenizer architecture
- **Performance**: This pattern avoids double model loading and maintains tokenization consistency across the pipeline