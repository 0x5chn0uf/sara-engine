# Serena CLI Commands Reference

> **Overview**: Complete reference for all Serena CLI commands and their usage patterns  
> **Audience**: Developers, Claude Code, and contributors  
> **Last Updated**: July 2025

## What is Serena CLI?

Serena provides a command-line interface for managing semantic search and memory operations in the SmartWalletFX project. All operations are performed through a local server to ensure consistency and performance.

## Command Categories

### 📋 **Two Types of Indexing**

Serena provides two distinct indexing systems:

| Command | Purpose | File Types | Default Directories |
|---------|---------|------------|--------------------|
| `index` | Documentation & content | `*.md`, `*.txt`, `*.json`, `*.yaml`, `*.yml` | `.taskmaster/`, `docs/`, `.serena/memories` |
| `embed index` | Code semantic search | `*.py`, `*.ts`, `*.tsx`, `*.js`, `*.jsx` | Project-wide code files |

### 🚀 **Core Operations**

- [`init`](#init) - Initialize database and configuration
- [`serve`](#serve) - Start the local server
- [`index`](#index) - Index documentation and content files
- [`search`](#search) - Semantic search across indexed content
- [`context`](#context) - Multi-source context retrieval
- [`embed`](#embed) - Selective codebase embedding and semantic code search

### 📋 **Memory Management**

- [`get`](#get) - Retrieve specific memory by ID
- [`latest`](#latest) - Get recently indexed memories
- [`delete`](#delete) - Remove memories from the system

### 🔧 **System Administration**

- [`maintenance`](#maintenance) - Run database maintenance
- [`pool`](#pool) - Monitor server status and health

---

## Command Details

### `init`

**Purpose**: Initialize the Serena database and configuration files.

```bash
sara init
```

**Usage**: Run once when setting up Serena in a new project. Creates the SQLite database, configuration files, and sets up the initial schema.

**Output**: Creates `.serena/` directory with database and config files.

---

### `serve`

**Purpose**: Start the local Serena server for processing requests.

```bash
sara serve [options]
```

**Arguments**:

- `--host HOST` - Server host (default: localhost)
- `--port PORT` - Server port (default: 8899)
- `--watch` - Enable auto-reload for development
- `-v, --verbose` - Enable verbose logging

**Usage**: Must be running for other commands to work. Starts FastAPI server with semantic search capabilities.

**Example**:

```bash
# Start server with default settings
sara serve

# Start with custom port and auto-reload
sara serve --port 9000 --watch
```

---

### `index`

**Purpose**: Index documentation and content files for semantic search.

```bash
sara index [options]
```

**Arguments**:

- `--directories DIRS` - Comma-separated directories to scan (default: auto-detect TaskMaster dirs)
- `--files FILES` - Comma-separated individual files to index
- `--force` - Force re-indexing of existing content
- `--workers WORKERS` - Number of parallel workers (default: 4)
- `-v, --verbose` - Enable verbose logging

**File Types**: Indexes documentation files (`*.md`, `*.txt`, `*.json`, `*.yaml`, `*.yml`) from TaskMaster archives, documentation directories, and configuration files.

**Usage**: Processes documentation and content files for semantic search. Automatically detects TaskMaster directories if no specific directories are provided.

**Example**:

```bash
# Index current directory
sara index

# Index specific directory with force refresh
sara index /path/to/docs --force

# Index with file limit
sara index --limit 100
```

---

### `search`

**Purpose**: Perform semantic search across indexed memories.

```bash
sara search "query" [options]
```

**Arguments**:

- `query` - Search query string (required)
- `--limit LIMIT` - Number of results to return (default: 10)
- `--format FORMAT` - Output format: `default`, `claude-optimized`, `compact`, `json`
- `--advanced` - Use advanced search mode
- `-v, --verbose` - Enable verbose logging

**Formats**:

- **`default`**: Human-readable verbose format with full details
- **`claude-optimized`**: Token-efficient format for LLM consumption (~60% reduction)
- **`compact`**: Ultra-compact single-line results (~80% reduction)
- **`json`**: Machine-readable structured data

**Usage**: Search through all indexed content using semantic similarity.

**Examples**:

```bash
# Basic search
sara search "user authentication"

# Claude Code optimized search
sara search "database patterns" --format=claude-optimized --limit=5

# Compact format for quick scanning
sara search "FastAPI endpoints" --format=compact

# JSON for programmatic use
sara search "error handling" --format=json
```

**Claude-Optimized Output Example**:

```
SEARCH: user authentication
RESULTS: 3 found | SCORES: 0.465-0.416

1. [T125] JWT Implementation Guide ✅
   🔥 0.465 | 🏷️ task | 📁 auth-guide.md
   💬 Complete JWT authentication flow with middleware...

2. [121] Auth State Management 📋
   📊 0.453 | 🏷️ reflection | 📁 reflection-121.md
   💬 React authentication state with protected routes...
```

---

### `context`

**Purpose**: Retrieve semantically-relevant context from multiple sources for Claude Code.

```bash
serena context "query" [options]
```

**Arguments**:

- `query` - Context query string (required)
- `--limit LIMIT` - Maximum results per source (default: 10)
- `--scope SCOPE` - Limit to: `all`, `tasks`, `memories`, `code`
- `--format FORMAT` - Output format: `claude-optimized`, `json`, `default`
- `-v, --verbose` - Enable verbose logging

**Sources**:

- **TaskMaster tasks**: Project task definitions and progress
- **Memory bank**: Historical context and architectural decisions
- **Semantic memories**: Serena's indexed content with relevance scoring

**Usage**: Optimized for Claude Code to gather comprehensive context efficiently.

**Examples**:

```bash
# Get context for implementation
serena context "implement rate limiting"

# Focus on tasks only
serena context "authentication flow" --scope=tasks

# JSON output for programmatic use
serena context "database migration" --format=json --limit=5
```

**Output Example**:

```
CONTEXT: implement rate limiting
RELEVANCE: 0.84 | SOURCES: 2 tasks, 3 memories, 0 code refs

TASKS:
- [T089] Rate Limiting Middleware - ✅ done
  Key: Implemented Redis-based rate limiting with configurable thresholds

MEMORIES:
- 🧠 middleware-patterns: Rate limiting strategies and implementation patterns
- 📚 redis-integration: Connection pooling and performance optimization

SUMMARY:
Context analysis complete: 2 relevant tasks found, 3 memory entries located.
```

---

### `get`

**Purpose**: Retrieve detailed information about a specific memory by ID.

```bash
sara get TASK_ID [options]
```

**Arguments**:

- `TASK_ID` - Unique identifier for the memory (required)
- `--content` - Include full file content in output
- `-v, --verbose` - Enable verbose logging

**Usage**: Get detailed information about a specific indexed item.

**Example**:

```bash
# Get basic info
sara get T125

# Get with full content
sara get T125 --content
```

---

### `latest`

**Purpose**: Show recently indexed memories.

```bash
sara latest [options]
```

**Arguments**:

- `--limit LIMIT` - Number of recent items to show (default: 10)
- `-v, --verbose` - Enable verbose logging

**Usage**: Quick overview of recently added or updated content.

**Example**:

```bash
# Show 10 most recent
sara latest

# Show 20 most recent
sara latest --limit 20
```

---

### `delete`

**Purpose**: Remove specific memories from the system.

```bash
sara delete [options]
```

**Arguments**:

- `--id ID` - Specific memory ID to delete
- `--interactive` - Interactive deletion mode
- `-v, --verbose` - Enable verbose logging

**Usage**: Clean up obsolete or incorrect memories. Use with caution.

**Example**:

```bash
# Delete specific memory
sara delete --id T125

# Interactive mode
sara delete --interactive
```

---

### `maintenance`

**Purpose**: Run database maintenance operations.

```bash
sara maintenance [options]
```

**Arguments**:

- `--force` - Force maintenance regardless of schedule
- `--vacuum` - Run database vacuum operation (reclaims disk space and optimizes performance)
- `--backup` - Create backup before maintenance
- `-v, --verbose` - Enable verbose logging

**Usage**: Optimize database performance and clean up unused space.

**Example**:

```bash
# Run scheduled maintenance
sara maintenance

# Force vacuum with backup
sara maintenance --force --vacuum --backup
```

---

### `pool`

**Purpose**: Monitor server status and health.

```bash
sara pool <subcommand> [options]
```

**Subcommands**:

- `status` - Show server connection status
- `health` - Display detailed health metrics

**Arguments**:

- `-v, --verbose` - Enable verbose logging

**Usage**: Check if the Serena server is running and healthy.

**Examples**:

```bash
# Check server status
sara pool status

# Get detailed health info
sara pool health
```

---

### `embed`

**Purpose**: Selective codebase embedding for semantic code search without exposing full source code.

```bash
sara embed <action> [options]
```

**Actions**:

#### `embed index`

Index codebase or specific files for embedding.

```bash
sara embed index [options]
```

**Arguments**:

- `--force` - Force reindexing of all files (ignore SHA-256 checksums)
- `--files FILES` - Specific files to index (space-separated)
- `--project-root PATH` - Project root directory (default: current directory)
- `-v, --verbose` - Enable verbose logging

**Usage**: Creates semantic embeddings of code files with smart chunking and incremental updates.

**Examples**:

```bash
# Index entire codebase
sara embed index

# Index specific files
sara embed index --files backend/app/models.py frontend/src/components/Auth.tsx

# Force reindex with verbose output
sara embed index --force --verbose

# Index from specific project root
sara embed index --project-root /path/to/project
```

#### `embed search`

Search embedded code using semantic similarity.

```bash
sara embed search "query" [options]
```

**Arguments**:

- `query` - Search query string (required)
- `--limit LIMIT` - Maximum number of results (default: 10)
- `-v, --verbose` - Enable verbose logging

**Usage**: Find relevant code chunks using natural language queries.

**Examples**:

```bash
# Search for authentication code
sara embed search "JWT token validation middleware"

# Search with custom result limit
sara embed search "database connection pool" --limit 5

# Verbose search for debugging
sara embed search "error handling patterns" --verbose
```

#### `embed stats`

Show code embedding statistics.

```bash
sara embed stats
```

**Usage**: Display metrics about indexed code files and embeddings.

**Example Output**:

```
📊 Code Embedding Statistics:
   📁 Files indexed: 127
   🎯 Embeddings generated: 1,834
   📈 Average chunks per file: 14.4
```

**Key Features**:

- **Smart Chunking**: 4KB chunks with 20-line overlap and structure-aware splitting
- **Incremental Updates**: SHA-256 hash-based change detection
- **Language-Aware Processing**: Comment stripping for Python, TypeScript, JavaScript
- **Configurable File Selection**: Include/exclude patterns via environment variables
- **Token Efficiency**: Semantic search without exposing full source code
- **Binary Vector Storage**: Efficient embedding storage in SQLite

**Configuration**:

Customize behavior via environment variables:

```bash
# Core settings
SARA_EMBEDDING_CHUNK_SIZE=4096
SARA_EMBEDDING_OVERLAP_LINES=20
SARA_EMBEDDING_STRIP_COMMENTS=true

# File patterns
SARA_EMBEDDING_INCLUDE_GLOBS="*.py,*.ts,*.tsx,*.js,*.jsx"
SARA_EMBEDDING_EXCLUDE_GLOBS="**/test*/**,**/node_modules/**"
```

---

## Usage Patterns

### For Claude Code Integration

**Primary Commands**:

```bash
# Get comprehensive context (most common)
serena context "feature to implement" --format=claude-optimized

# Detailed semantic search when needed
sara search "specific pattern" --format=claude-optimized --limit=5

# Search code semantically
sara embed search "authentication middleware patterns" --limit=5

# Get specific task details
sara get T125
```

### For Development Workflow

**Setup**:

```bash
sara init
sara serve --watch  # In background
```

**Content Management**:

```bash
sara index .                    # Index current project
sara embed index                # Index codebase for semantic search
sara search "what I'm looking for"
sara embed search "code patterns I need"
sara latest                     # See recent additions
```

### For System Administration

**Maintenance**:

```bash
sara maintenance --vacuum --backup
sara pool health
```

---

## Output Formats Comparison

| Format             | Use Case         | Token Efficiency | Structure                   |
| ------------------ | ---------------- | ---------------- | --------------------------- |
| `default`          | Human reading    | Baseline         | Verbose, detailed           |
| `claude-optimized` | LLM consumption  | ~60% reduction   | Structured, token-efficient |
| `compact`          | Quick scanning   | ~80% reduction   | Single-line results         |
| `json`             | Programmatic use | Variable         | Machine-readable            |

---

## Error Handling

All commands provide structured error messages and suggested solutions:

```bash
❌ Serena server not available or search failed
   💡 Solution: Start the server with: sara serve
```

Use `--verbose` flag for detailed error information and debugging.

---

## Integration Notes

- **Server Dependency**: Most commands require `sara serve` to be running
- **TaskMaster Integration**: Automatically detects and indexes `.taskmaster/` directories
- **Memory Safety**: All operations include proper cleanup and connection management
- **Performance**: Claude-optimized formats significantly reduce token usage for LLM workflows
