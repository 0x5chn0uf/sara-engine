"""Context command for Claude Code integration."""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from sara.cli.common import RemoteMemory, setup_logging
from sara.core.errors import ErrorCode, get_user_friendly_message


@dataclass
class ContextResult:
    """Structured context result for Claude Code consumption."""
    query: str
    relevance_score: float
    source_count: Dict[str, int]
    tasks: List[Dict[str, Any]]
    memories: List[Dict[str, Any]]
    code_refs: List[Dict[str, Any]]
    summary: str


def _search_taskmaster_context(query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], int]:
    """Search TaskMaster tasks for relevant context."""
    tasks = []
    task_count = 0
    
    try:
        # Look for TaskMaster integration files
        taskmaster_dirs = [
            ".taskmaster/logs",
            ".taskmaster/memory-bank/archive", 
            ".taskmaster/memory-bank/reflection"
        ]
        
        for task_dir in taskmaster_dirs:
            if os.path.exists(task_dir):
                # Simple file-based search for now
                # In a full implementation, this would use TaskMaster's search API
                for file_path in Path(task_dir).rglob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                        if isinstance(data, dict):
                            title = data.get('title', str(file_path.name))
                            description = data.get('description', data.get('details', ''))
                            
                            # Simple relevance check
                            if any(term.lower() in f"{title} {description}".lower() 
                                  for term in query.split()):
                                tasks.append({
                                    'id': data.get('id', file_path.stem),
                                    'title': title,
                                    'status': data.get('status', 'unknown'),
                                    'description': description[:200] + '...' if len(description) > 200 else description,
                                    'file': str(file_path)
                                })
                                task_count += 1
                                
                                if len(tasks) >= limit:
                                    break
                                    
                    except (json.JSONDecodeError, IOError):
                        continue
                        
            if len(tasks) >= limit:
                break
                
    except Exception as e:
        print(f"Warning: Could not search TaskMaster context: {e}")
    
    return tasks[:limit], task_count


def _search_memory_bank(query: str, limit: int = 3) -> Tuple[List[Dict[str, Any]], int]:
    """Search memory bank for relevant context."""
    memories = []
    memory_count = 0
    
    try:
        memory_dirs = [
            ".taskmaster/memory-bank/archive",
            ".taskmaster/memory-bank/reflection", 
            ".taskmaster/memory-bank"
        ]
        
        for memory_dir in memory_dirs:
            if os.path.exists(memory_dir):
                for file_path in Path(memory_dir).rglob("*.md"):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            
                        # Simple relevance check
                        if any(term.lower() in content.lower() for term in query.split()):
                            # Extract first few lines as summary
                            lines = content.split('\n')
                            summary = '\n'.join(lines[:5]).strip()
                            
                            memories.append({
                                'name': file_path.stem,
                                'summary': summary[:300] + '...' if len(summary) > 300 else summary,
                                'file': str(file_path),
                                'type': 'memory-bank'
                            })
                            memory_count += 1
                            
                            if len(memories) >= limit:
                                break
                                
                    except IOError:
                        continue
                        
            if len(memories) >= limit:
                break
                
    except Exception as e:
        print(f"Warning: Could not search memory bank: {e}")
    
    return memories[:limit], memory_count


def _search_semantic_memories(remote_memory: RemoteMemory, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], int]:
    """Search Sara's semantic memories."""
    memories = []
    memory_count = 0
    
    try:
        results_data = remote_memory.search(query, limit=limit)
        
        # Convert API response to structured format (similar to search_cmd.py)
        for item in results_data:
            if isinstance(item, dict):
                memories.append({
                    'task_id': item.get('task_id'),
                    'title': item.get('title', 'Untitled'),
                    'excerpt': (item.get('excerpt', '')[:200] + '...' 
                               if item.get('excerpt') and len(item.get('excerpt', '')) > 200 
                               else item.get('excerpt', '')),
                    'score': item.get('score', 0.0),
                    'filepath': item.get('filepath'),
                    'kind': item.get('kind'),
                    'type': 'semantic'
                })
                memory_count += 1
            
    except Exception as e:
        print(f"Warning: Could not search semantic memories: {e}")
    
    return memories, memory_count


def _format_claude_optimized(context: ContextResult) -> str:
    """Format context result for Claude Code consumption."""
    output = []
    
    # Header with key metrics
    output.append(f"CONTEXT: {context.query}")
    output.append(f"RELEVANCE: {context.relevance_score:.2f} | SOURCES: {context.source_count.get('tasks', 0)} tasks, {context.source_count.get('memories', 0)} memories, {context.source_count.get('code_refs', 0)} code refs")
    output.append("")
    
    # Tasks section
    if context.tasks:
        output.append("TASKS:")
        for task in context.tasks:
            status_emoji = "✅" if task.get('status') == 'done' else "🔄" if task.get('status') == 'in-progress' else "⏳"
            output.append(f"- [{task.get('id', 'T?')}] {task.get('title', 'Untitled')} - {status_emoji} {task.get('status', 'unknown')}")
            if task.get('description'):
                output.append(f"  Key: {task.get('description')}")
        output.append("")
    
    # Memories section  
    if context.memories:
        output.append("MEMORIES:")
        for memory in context.memories:
            mem_type = "🧠" if memory.get('type') == 'semantic' else "📚"
            output.append(f"- {mem_type} {memory.get('name', memory.get('task_id', 'Unknown'))}: {memory.get('title', memory.get('summary', 'No summary'))}")
            if memory.get('score'):
                output.append(f"  Score: {memory.get('score'):.3f}")
        output.append("")
    
    # Code references section
    if context.code_refs:
        output.append("CODE:")
        for ref in context.code_refs:
            output.append(f"- {ref.get('file', 'unknown')}:{ref.get('symbol', 'unknown')} - {ref.get('description', 'No description')}")
        output.append("")
    
    # Summary
    if context.summary:
        output.append("SUMMARY:")
        output.append(context.summary)
    
    return "\n".join(output)


def _format_json(context: ContextResult) -> str:
    """Format context result as JSON."""
    return json.dumps({
        'query': context.query,
        'relevance_score': context.relevance_score,
        'source_count': context.source_count,
        'tasks': context.tasks,
        'memories': context.memories,
        'code_refs': context.code_refs,
        'summary': context.summary
    }, indent=2)


def cmd_context(args) -> None:
    """Provide semantically-relevant context for Claude Code."""
    setup_logging(args.verbose)
    
    try:
        print(f"🔍 Gathering context for: '{args.query}'")
        
        # Validate query
        if not args.query or len(args.query.strip()) == 0:
            print("❌ Error: Context query cannot be empty")
            sys.exit(1)
            
        if len(args.query) > 500:
            print("❌ Error: Context query too long (max 500 characters)")
            sys.exit(1)
        
        # Initialize remote memory connection
        remote_memory = None
        try:
            remote_memory = RemoteMemory()
            server_available = remote_memory.is_server_available()
        except Exception:
            server_available = False
        
        # Collect context from multiple sources
        tasks = []
        memories = []
        code_refs = []
        
        task_count = 0
        memory_count = 0
        code_count = 0
        
        # Search TaskMaster if scope allows
        if not args.scope or args.scope in ['all', 'tasks']:
            tasks, task_count = _search_taskmaster_context(args.query, args.limit // 3)
        
        # Search memory bank if scope allows
        if not args.scope or args.scope in ['all', 'memories']:
            mb_memories, mb_count = _search_memory_bank(args.query, args.limit // 3)
            memories.extend(mb_memories)
            memory_count += mb_count
        
        # Search semantic memories if server available and scope allows
        if server_available and (not args.scope or args.scope in ['all', 'memories']):
            sem_memories, sem_count = _search_semantic_memories(remote_memory, args.query, args.limit // 2)
            memories.extend(sem_memories)
            memory_count += sem_count
        
        # Calculate overall relevance score
        total_results = len(tasks) + len(memories) + len(code_refs)
        relevance_score = min(0.95, total_results / max(args.limit, 1)) if total_results > 0 else 0.0
        
        # Generate summary
        summary_parts = []
        if tasks:
            summary_parts.append(f"{len(tasks)} relevant tasks found")
        if memories:
            summary_parts.append(f"{len(memories)} memory entries located")
        if code_refs:
            summary_parts.append(f"{len(code_refs)} code references identified")
            
        summary = f"Context analysis complete: {', '.join(summary_parts) if summary_parts else 'No relevant context found'}."
        
        # Build context result
        context = ContextResult(
            query=args.query,
            relevance_score=relevance_score,
            source_count={
                'tasks': task_count,
                'memories': memory_count, 
                'code_refs': code_count
            },
            tasks=tasks[:args.limit],
            memories=memories[:args.limit],
            code_refs=code_refs[:args.limit],
            summary=summary
        )
        
        # Output in requested format
        if args.format == 'claude-optimized':
            print("\n" + _format_claude_optimized(context))
        elif args.format == 'json':
            print(_format_json(context))
        else:
            # Default format - simplified version
            print(f"\n✅ Found context from {total_results} sources:")
            print("-" * 50)
            
            if tasks:
                print("📋 TASKS:")
                for task in tasks:
                    print(f"  - [{task.get('id')}] {task.get('title')} ({task.get('status')})")
                print()
                
            if memories:
                print("🧠 MEMORIES:")
                for memory in memories:
                    name = memory.get('name', memory.get('task_id', 'Unknown'))
                    print(f"  - {name}: {memory.get('summary', memory.get('title', 'No summary'))[:100]}...")
                print()
            
            print(f"📊 Summary: {summary}")
        
    except KeyboardInterrupt:
        print("\n🛑 Context search cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Context search failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if remote_memory:
            try:
                remote_memory.close()
            except Exception:
                pass


def register(sub: Any) -> None:
    """Register the context command."""
    p = sub.add_parser("context", help="Get semantic context for Claude Code")
    p.add_argument("query", help="Context query string")
    p.add_argument("--limit", type=int, default=10, help="Maximum number of results per source")
    p.add_argument("--scope", choices=['all', 'tasks', 'memories', 'code'], 
                   help="Limit search scope to specific sources")
    p.add_argument("--format", choices=['default', 'claude-optimized', 'json'], 
                   default='claude-optimized', help="Output format")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    p.set_defaults(func=cmd_context)