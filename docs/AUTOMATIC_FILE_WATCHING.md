# Automatic File Watching Integration

## Overview

I have successfully integrated automatic file re-indexing functionality with the `sara serve --watch` command. When the `--watch` flag is used, the server will automatically detect file changes and re-index them in real-time.

## Implementation Details

### 1. Server Integration (`serena/infrastructure/server.py`)

- **Environment Variable Detection**: The server checks for `SARA_WATCH_MODE=true` to enable file watching
- **Lifespan Management**: File watcher is started during server startup and stopped during shutdown
- **Memory Adapter**: Created `ServerMemoryAdapter` class to bridge the watcher with the server's database

### 2. CLI Integration (`serena/cli/serve_cmd.py`)

- **Environment Setup**: When `--watch` flag is used, sets `SARA_WATCH_MODE=true` environment variable
- **Uvicorn Compatibility**: Works with both reload and non-reload modes

### 3. File Watcher Usage (`serena/infrastructure/watcher.py`)

- **Existing Infrastructure**: Leverages the existing robust `_WatchdogMemoryWatcher` implementation
- **Watchdog Library**: Uses the reliable `watchdog` library instead of polling
- **TaskMaster Integration**: Automatically discovers and watches TaskMaster directories

## Features

### Automatic Detection
- Monitors markdown files (`.md`), text files (`.txt`), JSON files (`.json`), and other supported formats
- Automatically discovers TaskMaster directories and project files
- Tracks file changes using SHA-256 hashes

### Real-time Processing
- Detects file creation, modification, and deletion events
- Updates the database and search index automatically
- Provides console logging for monitoring activity

### Graceful Handling
- Proper startup and shutdown sequences
- Error handling and logging
- Integration with existing server lifecycle

## Usage

### Starting the Server with File Watching

```bash
# Start server with automatic file watching
sara serve --watch

# Or with custom host/port
sara serve --watch --host 0.0.0.0 --port 8765
```

### Expected Output

```
🚀 Starting Serena server on 127.0.0.1:8765
📖 API docs available at http://127.0.0.1:8765/docs
📊 Health check: http://127.0.0.1:8765/health
...
👀 Starting file watcher for automatic re-indexing...
✅ File watcher started - files will be automatically re-indexed on changes
   - Watching X directories
   - Tracking Y files
👀 Watch mode enabled - files will be automatically re-indexed on changes
🎉 Serena server startup completed in X.XXs
```

### Real-time Monitoring

When files are changed, you'll see output like:
```
🔄 File watcher: modified task_123 from /path/to/file.md
```

## Architecture

```
sara serve --watch
     ↓
CLI sets SARA_WATCH_MODE=true
     ↓
Server lifespan detects watch mode
     ↓
Creates ServerMemoryAdapter
     ↓
Initializes _WatchdogMemoryWatcher
     ↓
Monitors filesystem events
     ↓
Auto-updates database on changes
```

## Benefits

1. **Real-time Updates**: No need to manually re-run indexing commands
2. **Seamless Integration**: Works transparently with existing server functionality
3. **Efficient**: Only processes changed files, not entire directories
4. **Robust**: Uses proven watchdog library for cross-platform file monitoring
5. **Development Friendly**: Perfect for development workflows where files change frequently

## Comparison with Standalone `serena watch`

| Feature | `sara serve --watch` | `serena watch` |
|---------|----------------------|----------------|
| Server Integration | ✅ Built-in | ❌ Requires separate server |
| API Access | ✅ Immediate | ❌ Depends on server |
| Resource Usage | ✅ Single process | ❌ Two processes |
| Development Workflow | ✅ Perfect fit | ❌ More complex setup |

## Technical Notes

- The file watcher uses the existing `_WatchdogMemoryWatcher` class
- The `ServerMemoryAdapter` handles the interface between the watcher and server database
- File changes trigger the same indexing pipeline as manual commands
- The implementation properly handles async/sync context switching
- Graceful shutdown ensures no data loss during server termination

This implementation fulfills the user's request: "I want him to embed the documents when i update them, as well as for the code" by providing automatic re-indexing of both documents and code files when using `sara serve --watch`.