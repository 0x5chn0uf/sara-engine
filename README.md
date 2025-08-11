# Sara Memory Bridge

Sara is a lightweight knowledge base that plugs into any project, providing:

- 💡 **Semantic search** over markdown archives, reflections, and documentation
- 🔍 **Selective codebase embedding** with smart chunking and incremental updates
- 📨 **REST API** for runtime content ingestion and retrieval
- 🏷 **Task-aware schema** with support for archives, reflections, docs, rules, and code
- 🛠 **Zero-config CLI** (`sara`) for indexing, search, and HTTP server
- ⚡️ **SQLite + vector embeddings** backend – no external services required
- 🔄 **Content deduplication** and automatic metadata extraction
- 📊 **Health monitoring** and database diagnostics

## Installation

```bash
pip install -e .  # from repo root (editable)
# or once published
pip install sara
```

## Quick Start

```bash
# One-time setup
sara init                     # creates local SQLite database

# Index documentation and content files
sara index                    # scan and index TaskMaster/content files (*.md, *.txt, *.json)
sara index --force            # force reindex all content
sara index --directories ./docs,./archives --workers 8

# Search indexed content
sara search "jwt rotation"    # semantic search
sara search "embeddings schema" --limit 20

# Code embedding and search (separate from content indexing)
sara embed index              # index code files (*.py, *.ts, *.tsx, *.js, *.jsx)
sara embed index --files backend/app/models.py --force
sara embed search "authentication JWT validation" --limit 5
sara embed stats              # embedding statistics

# Run API server
sara serve                    # start server on localhost:8765
sara serve --host 0.0.0.0 --port 9000 --watch

# Alternative entry point
selena search "authentication patterns"
```

## CLI Reference

### Two Types of Indexing

Sara provides two distinct indexing commands for different types of content:

| Command            | Purpose                          | File Types                                   | Use Case                                                |
| ------------------ | -------------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| `sara index`       | Documentation & content indexing | `*.md`, `*.txt`, `*.json`, `*.yaml`, `*.yml` | TaskMaster archives, documentation, configuration files |
| `sara embed index` | Code semantic search             | `*.py`, `*.ts`, `*.tsx`, `*.js`, `*.jsx`     | Semantic code search without exposing full source       |

**When to use which:**

- Use `sara index` for searching documentation, TaskMaster archives, and content files
- Use `sara embed index` for semantic code search and finding relevant code patterns

### Core Commands

#### `sara init`

Initialize Sara database and configuration.

- `-v, --verbose`: Enable verbose logging

#### `sara index [options]`

Scan directories and index memories with semantic embeddings.

- `--directories <paths>`: Comma-separated directories to scan
- `--force`: Force reindex of existing memories
- `--workers <int>`: Number of parallel workers (default: 4)
- `-v, --verbose`: Enable verbose logging

#### `sara search <query> [options]`

Perform semantic search across indexed memories.

- `--limit <int>`: Maximum number of results (default: 10)
- `--advanced`: Advanced search mode (planned feature)
- `-v, --verbose`: Enable verbose logging

#### `sara serve [options]`

Run HTTP API server with automatic documentation.

- `--host <address>`: Server host (default: 127.0.0.1)
- `--port <int>`: Server port (default: 8765)
- `--watch`: Watch for file changes (planned feature)
- `-v, --verbose`: Enable verbose logging

#### `sara delete <task_id> [options]`

Delete indexed entries manually by task ID.

- `--list`: List available entries with their task IDs
- `--limit <int>`: Number of entries to show when listing (default: 20)
- `--show-remaining`: Show remaining entries after deletion
- `-v, --verbose`: Enable verbose logging

#### `sara get <task_id> [options]`

Get specific archive by task ID.

- `-v, --verbose`: Enable verbose logging

#### `sara latest [options]`

Show latest archived entries.

- `--limit <int>`: Number of entries to show (default: 10)
- `-v, --verbose`: Enable verbose logging

#### `sara embed <action> [options]`

Selective codebase embedding for semantic code search.

**Actions:**

- `index [options]`: Index codebase or specific files
  - `--force`: Force reindexing (ignore SHA-256 checksums)
  - `--files <files>`: Specific files to index
  - `--project-root <path>`: Project root directory
- `search <query> [options]`: Search embedded code
  - `--limit <int>`: Maximum results (default: 10)
- `stats`: Show embedding statistics

**Examples:**

```bash
# Index entire codebase
sara embed index

# Index specific files with force reindexing
sara embed index --files backend/app/models.py frontend/src/components/Auth.tsx --force

# Search for authentication-related code
sara embed search "JWT token validation middleware"

# Check embedding statistics
sara embed stats
```

### Content Types

Sara automatically categorizes content by type:

- **ARCHIVE**: Completed task archives and historical records
- **REFLECTION**: Post-completion reflections and lessons learned
- **DOC**: General documentation and guides
- **RULE**: Configuration rules and patterns
- **CODE**: Code snippets and technical references
- **CODEBASE**: Selective code embeddings for semantic search

### Advanced Features

- **Automatic task ID extraction** from file paths
- **Content deduplication** using SHA-256 hashing
- **Chunked embeddings** for large documents
- **Versioned embeddings** for model upgrades
- **Health monitoring** with database metrics
- **Selective code embedding** with configurable file patterns
- **Incremental updates** using SHA-256 change detection
- **Smart code chunking** with structure-aware splitting
- **Language-aware processing** (Python, TypeScript, JavaScript)
- **Comment stripping** for cleaner embeddings

## Configuration

### Embedding Models

Sentence-Transformers is pulled in as an optional dependency. The first call will download the default MiniLM-L6 model (~80 MB). Override via `SARA_MODEL` environment variable if you need a different encoder.

### Code Embedding Configuration

Customize code embedding behavior via environment variables:

```bash
# Core embedding settings
SARA_EMBEDDING_CHUNK_SIZE=4096          # Chunk size in bytes (default: 4096)
SARA_EMBEDDING_OVERLAP_LINES=20         # Lines to overlap between chunks (default: 20)
SARA_EMBEDDING_STRIP_COMMENTS=true      # Remove comments before embedding (default: true)

# File selection patterns (comma-separated globs)
SARA_EMBEDDING_INCLUDE_GLOBS="*.py,*.ts,*.tsx,*.js,*.jsx,backend/app/**/*.py,frontend/src/**/*.ts"
SARA_EMBEDDING_EXCLUDE_GLOBS="**/test*/**,**/tests/**,**/node_modules/**,**/.git/**,**/build/**,**/__pycache__/**"
```

### Default File Patterns

**Included by default:**

- Python files: `*.py`
- TypeScript/JavaScript: `*.ts`, `*.tsx`, `*.js`, `*.jsx`
- Backend code: `backend/app/**/*.py`
- Frontend code: `frontend/src/**/*.ts`, `frontend/src/**/*.tsx`

**Excluded by default:**

- Test files: `**/test*/**`, `**/tests/**`, `**/*test*`
- Build artifacts: `**/build/**`, `**/dist/**`, `**/*.min.js`, `**/*.map`
- Dependencies: `**/node_modules/**`, `**/__pycache__/**`, `**/*.pyc`
- Version control: `**/.git/**`
- Database migrations: `**/migrations/**`
- Coverage reports: `**/coverage/**`, `**/.pytest_cache/**`

## License

MIT © SmartWalletFX Team
