# Serena Development Rules

## Plugin Architecture Principles

**CRITICAL**: Serena is a simple local plugin that lives alongside the codebase. It is NOT designed for online deployment.

### Core Design Philosophy

- **Server-side processing only** - All operations happen on the running server
- **No local queues or background workers** - Keep implementation simple
- **Local development tool** - Designed to run next to your codebase, not deployed
- **Synchronous operations** - No complex async patterns or job queues

## Remote Operation Requirement

**ALL operations must go through the running server via CLI commands** - never bypass the server layer.

## Required Command Pattern

Always use the official Serena CLI commands that communicate with the server:

### Core Operations

- `sara init` - Initialize database and configuration
- `sara index [path]` - Index content via server (synchronous)
- `sara search [query]` - Search indexed content
- `sara embed <action>` - Selective codebase embedding and search
  - `sara embed index` - Index code files for semantic search
  - `sara embed search <query>` - Search embedded code
  - `sara embed stats` - Show embedding statistics
- `sara delete [id]` - Delete specific content by ID
- `sara serve` - Start the simple local server
- `sara maintenance` - Run maintenance operations

### Status and Information

- `sara get [id]` - Retrieve specific content
- `sara latest` - Show recent additions
- `sara pool` - Show connection pool status

## Prohibited Operations

❌ **NEVER DO THESE:**

- Direct SQLite database file manipulation
- Local write queues or background processing
- Complex deployment patterns (Docker, systemd, etc.)
- Local Python imports to manipulate Serena objects
- Direct modification of `.db` files
- Bypassing the server for any operations
- Async job queues or worker processes

## Simplified Architecture Requirements

✅ **KEEP IT SIMPLE:**

- Single server process handling all requests synchronously
- Direct database operations through server only
- No background workers or queues
- Simple REST API for basic operations
- Local development focus only

## Required Workflow

**CRITICAL**: Always run commands from the project root directory and activate the backend virtual environment first:
```bash
# Navigate to project root (trading_bot_smc/)
cd /path/to/trading_bot_smc

# Activate virtual environment
source backend/.venv/bin/activate

# Then run serena commands
sara serve
```

1. Start `sara serve` for a simple local server
2. All operations go through server endpoints via CLI
3. Keep operations synchronous and simple
4. No complex deployment or scaling concerns

## Console-First Interface

**CRITICAL**: Serena is a command shell assistant, NOT an API service.

✅ **CONSOLE OUTPUT ONLY:**

- All user-facing responses must be printed to console
- Use logging and print statements for status information
- Health checks, status reports, and operations should log to terminal

❌ **NO API RESPONSES:**

- Never return JSONResponse for health/status commands, custom response instead
- Internal server endpoints can use JSON, but CLI commands log to console
- Replace with print() statements for console output only

## Development Testing

- Use CLI commands that hit the running server
- Test via simple API endpoints
- Keep test complexity minimal
- Focus on core functionality only

---

**Remember**: Serena is a simple local plugin, not a production system. Keep all operations server-side but maintain simplicity.
