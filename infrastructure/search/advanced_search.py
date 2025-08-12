"""Advanced search engine with context-aware features."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.models import SearchResult, TaskKind, TaskStatus
from infrastructure.search.search_core import SearchEngine


class AdvancedSearchEngine(SearchEngine):
    """Enhanced search engine with advanced ranking algorithms and context-aware suggestions."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.context_weights = {
            "task_complexity": 0.15,
            "task_urgency": 0.10,
            "user_history": 0.10,
            "domain_relevance": 0.20,
            "temporal_context": 0.05,
        }
        self.query_cache = {}
        self.suggestion_cache = {}

    def search_with_context(
        self,
        query: str,
        limit: int = 10,
        context: Optional[Dict[str, Any]] = None,
        enhance_query: bool = True,
    ) -> List[SearchResult]:
        """
        Enhanced search with context-aware ranking and query enhancement.

        Args:
            query: Search query
            limit: Maximum results to return
            context: Context information (task_id, complexity, urgency, etc.)
            enhance_query: Whether to enhance query with domain knowledge

        Returns:
            List of search results with enhanced ranking
        """
        # Cache key for enhanced queries
        cache_key = f"{query}_{hash(str(context))}" if context else query

        if cache_key in self.query_cache:
            return self.query_cache[cache_key][:limit]

        # For now, fall back to basic search - advanced features to be implemented
        # This provides a clean migration path
        results = self.search(query, k=limit)

        # Cache results
        self.query_cache[cache_key] = results
        return results

    def get_context_suggestions(
        self, context: Dict[str, Any], limit: int = 5
    ) -> List[str]:
        """
        Generate context-aware search suggestions.

        Args:
            context: Current context (current_task, recent_tasks, domain, etc.)
            limit: Maximum suggestions to return

        Returns:
            List of suggested queries
        """
        cache_key = f"suggestions_{hash(str(context))}"

        if cache_key in self.suggestion_cache:
            return self.suggestion_cache[cache_key][:limit]

        # Basic suggestions based on domain
        domain = context.get("domain", "general")
        suggestions = self._get_domain_suggestions(domain)

        # Cache and return
        self.suggestion_cache[cache_key] = suggestions
        return suggestions[:limit]

    def _get_domain_suggestions(self, domain: str) -> List[str]:
        """Get domain-specific search suggestions."""
        suggestions_map = {
            "backend": [
                "fastapi patterns",
                "database migrations",
                "api security",
                "async patterns",
            ],
            "frontend": [
                "react components",
                "state management",
                "ui patterns",
                "testing strategies",
            ],
            "security": [
                "2fa implementation",
                "jwt best practices",
                "encryption patterns",
                "security audits",
            ],
            "general": [
                "architecture patterns",
                "best practices",
                "common issues",
                "optimization techniques",
            ],
        }

        return suggestions_map.get(domain, suggestions_map["general"])


def search_memories_advanced(
    query: str,
    limit: int = 10,
    context: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> List[SearchResult]:
    """Advanced search with context awareness."""
    from database.session import get_session

    if db_path is None:
        from settings import settings

        db_path = settings.memory_db

    engine = AdvancedSearchEngine(db_path)
    return engine.search_with_context(query, limit, context)


def get_context_suggestions(
    context: Dict[str, Any], limit: int = 5, db_path: Optional[str] = None
) -> List[str]:
    """Get context-aware search suggestions."""
    from database.session import get_session

    if db_path is None:
        from settings import settings

        db_path = settings.memory_db

    engine = AdvancedSearchEngine(db_path)
    return engine.get_context_suggestions(context, limit)
