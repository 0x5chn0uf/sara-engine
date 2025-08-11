from __future__ import annotations

"""`sara search` and related sub-commands."""

import logging
import sys
from typing import Any

from sara.core.errors import (ErrorCode, SaraException,
                                get_user_friendly_message)
from sara.settings import settings
import json


def _try_server_search(query: str, limit: int):
    """Try to use server API for search with structured error handling."""
    remote_memory = None
    try:
        from sara.cli.common import RemoteMemory

        remote_memory = RemoteMemory()
        if not remote_memory.is_server_available():
            error_info = remote_memory.get_server_error_info()
            if error_info:
                print(
                    "Server unavailable - %s: %s",
                    error_info["code"],
                    error_info["message"],
                )
                if error_info.get("details"):
                    print("Error details: %s", error_info["details"])
            return [], False, error_info

        # Perform search using RemoteMemory (which handles structured errors)
        results_data = remote_memory.search(query, limit)

        # Convert API response to SearchResult objects
        from datetime import datetime

        from sara.core.models import SearchResult, TaskKind, TaskStatus

        results = []
        for item in results_data:
            result = SearchResult(
                task_id=item["task_id"],
                title=item["title"],
                score=item["score"],
                excerpt=item["excerpt"],
                kind=TaskKind(item["kind"]) if item["kind"] else None,
                status=TaskStatus(item["status"]) if item["status"] else None,
                completed_at=datetime.fromisoformat(item["completed_at"])
                if item["completed_at"]
                else None,
                filepath=item["filepath"],
            )
            results.append(result)

        return results, True, None  # Success

    except Exception as exc:
        print("Server search failed: %s", exc, exc_info=True)

        # Extract error info if it's a structured error
        error_info = None
        if isinstance(exc, SaraException):
            error_info = {
                "code": exc.code.value,
                "message": exc.message,
                "details": exc.details,
            }

        return [], False, error_info  # Failed

    finally:
        # Cleanup connections to prevent hanging
        if remote_memory:
            try:
                # Wait for server completion and close connections
                remote_memory.wait_for_server_completion(timeout=5.0)
                remote_memory.close()
            except Exception as cleanup_e:
                print(f"Cleanup warning: {cleanup_e}")


def _format_claude_optimized(results, query):
    """Format search results for Claude Code consumption."""
    output = []
    
    # Header with key metrics
    output.append(f"SEARCH: {query}")
    output.append(f"RESULTS: {len(results)} found | SCORES: {results[0].score:.3f}-{results[-1].score:.3f}" if results else "RESULTS: 0 found")
    output.append("")
    
    # Compact result listing
    for i, result in enumerate(results, 1):
        # Main result line with key info
        status_indicator = "✅" if result.status and "done" in str(result.status) else "📋"
        score_indicator = "🔥" if result.score and result.score > 0.7 else "📊" if result.score and result.score > 0.5 else "📈"
        
        output.append(f"{i}. [{result.task_id}] {result.title} {status_indicator}")
        
        # Compact metadata line
        metadata_parts = []
        if result.score:
            metadata_parts.append(f"{score_indicator} {result.score:.3f}")
        if result.kind:
            metadata_parts.append(f"🏷️ {result.kind.value}")
        if result.filepath:
            # Show just filename, not full path
            filename = result.filepath.split('/')[-1] if '/' in result.filepath else result.filepath
            metadata_parts.append(f"📁 {filename}")
            
        if metadata_parts:
            output.append(f"   {' | '.join(metadata_parts)}")
        
        # Truncated excerpt
        if result.excerpt:
            excerpt = result.excerpt.strip()[:120] + "..." if len(result.excerpt.strip()) > 120 else result.excerpt.strip()
            output.append(f"   💬 {excerpt}")
        
        output.append("")  # Empty line between results
    
    return "\n".join(output)


def _format_compact(results, query):
    """Format search results in compact single-line format."""
    output = []
    output.append(f"SEARCH: {query} | {len(results)} results")
    
    for result in results:
        score_str = f"{result.score:.3f}" if result.score else "0.000"
        title_truncated = result.title[:50] + "..." if len(result.title) > 50 else result.title
        output.append(f"[{result.task_id}] {score_str} | {title_truncated}")
    
    return "\n".join(output)


def _format_json(results, query):
    """Format search results as JSON."""
    results_data = []
    for result in results:
        results_data.append({
            'task_id': result.task_id,
            'title': result.title,
            'score': result.score,
            'excerpt': result.excerpt,
            'kind': result.kind.value if result.kind else None,
            'status': str(result.status) if result.status else None,
            'filepath': result.filepath,
        })
    
    return json.dumps({
        'query': query,
        'result_count': len(results),
        'results': results_data
    }, indent=2)


def cmd_search(args) -> None:
    """Search memories using semantic search with structured error handling."""
    try:
        print(f"🔍 Searching for: '{args.query}'")

        # Validate query
        if not args.query or len(args.query.strip()) == 0:
            print("❌ Error: Search query cannot be empty")
            print("   Please provide a search term")
            sys.exit(1)

        if len(args.query) > 500:
            print("❌ Error: Search query too long (max 500 characters)")
            sys.exit(1)

        results, server_success, error_info = _try_server_search(args.query, args.limit)

        if server_success:
            print("   ⚡ Using server API (preloaded model)")
        else:
            print("❌ Sara server not available or search failed")

            if error_info:
                # Display user-friendly error message using the simplified system
                error_code = ErrorCode(error_info["code"])
                friendly_message = get_user_friendly_message(
                    error_code, error_info["message"]
                )
                print(f"   {friendly_message}")

                if args.verbose and error_info.get("details"):
                    print(f"   Details: {error_info['details']}")
            else:
                print("   💡 Solution: Start the server with: sara serve")

            sys.exit(1)

        if not results:
            if getattr(args, 'format', 'default') in ['claude-optimized', 'compact', 'json']:
                # Structured no-results response
                if args.format == 'json':
                    print(_format_json([], args.query))
                elif args.format == 'compact':
                    print(f"SEARCH: {args.query} | 0 results")
                else:  # claude-optimized
                    print(f"SEARCH: {args.query}")
                    print("RESULTS: 0 found")
                    print("\nNo relevant memories found. Try different keywords or check if content is indexed.")
            else:
                print("❌ No results found")
                print(f"   Try different keywords or check if content is indexed")
            return

        # Format output based on requested format
        if getattr(args, 'format', 'default') == 'claude-optimized':
            print("\n" + _format_claude_optimized(results, args.query))
        elif getattr(args, 'format', 'default') == 'compact':
            print(_format_compact(results, args.query))
        elif getattr(args, 'format', 'default') == 'json':
            print(_format_json(results, args.query))
        else:
            # Default verbose format (existing)
            print(f"\n✅ Found {len(results)} results:")
            print("-" * 50)

            for i, result in enumerate(results, 1):
                print(f"{i}. [{result.task_id}] {result.title}")
                if result.filepath:
                    print(f"   📁 {result.filepath}")
                if result.kind:
                    print(f"   🏷️ {result.kind.value}")
                if result.score:
                    print(f"   📊 Score: {result.score:.3f}")
                if result.excerpt:
                    print(f"   📝 {result.excerpt[:100]}...")
                print()

    except KeyboardInterrupt:
        print("\n🛑 Search cancelled by user")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   💡 Solution: Install required dependencies")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        print("   💡 Solution: Check logs for details or report this issue")
        sys.exit(1)


def register(sub: Any) -> None:
    """Register the search command."""
    p = sub.add_parser("search", help="Semantic search across memories")
    p.add_argument("query", help="Query string")
    p.add_argument("--limit", type=int, default=10, help="Number of results to return")
    p.add_argument("--format", choices=['default', 'claude-optimized', 'compact', 'json'], 
                   default='default', help="Output format (claude-optimized for LLM consumption)")
    p.add_argument("--advanced", action="store_true", help="Use advanced mode")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    p.set_defaults(func=cmd_search)
