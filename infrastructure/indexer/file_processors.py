"""File processing utilities and filters."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Set

from sara.core.models import TaskKind, TaskStatus, determine_task_kind


def should_process_file(file_path: str, strategic_code_paths: Set[str]) -> bool:
    """Determine if a file should be processed based on strategic filtering."""
    # Always process docs and design files
    if file_path.startswith(("docs/", "design/", ".taskmaster/", ".serena/")):
        return True

    # Skip certain patterns
    skip_patterns = [
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".git",
        ".test.",
        "test_",
        "_test.",
        ".spec.",
        ".stories.",
        "dist/",
        "build/",
        ".venv/",
        ".ruff_cache/",
    ]

    if any(pattern in file_path for pattern in skip_patterns):
        return False

    # For code files, check if they match strategic paths
    if file_path.startswith(("backend/", "frontend/")):
        return any(file_path.startswith(path) for path in strategic_code_paths)

    return True


def extract_status_from_content(content: str) -> Optional[TaskStatus]:
    """Extract status from file content."""
    content_lower = content.lower()

    # Look for status indicators
    if "status: done" in content_lower or "completed" in content_lower:
        return TaskStatus.DONE
    elif "status: in-progress" in content_lower or "in progress" in content_lower:
        return TaskStatus.IN_PROGRESS
    elif "status: pending" in content_lower:
        return TaskStatus.PENDING
    elif "status: deferred" in content_lower:
        return TaskStatus.DEFERRED
    elif "status: cancelled" in content_lower:
        return TaskStatus.CANCELLED
    elif "status: blocked" in content_lower:
        return TaskStatus.BLOCKED

    # Default assumption based on file type
    if "archive" in content_lower or "reflection" in content_lower:
        return TaskStatus.DONE

    return None


def extract_completion_date(content: str, file_path: str) -> Optional[datetime]:
    """Extract completion date from content or file metadata."""
    # Try to extract from YAML frontmatter
    if content.startswith("---"):
        try:
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith(("completed:", "date:", "completed_at:")):
                        date_str = line.split(":", 1)[1].strip()
                        return datetime.fromisoformat(date_str.strip("\"'"))
        except Exception:
            pass

    # Look for completion markers in content
    lines = content.split("\n")
    for line in lines[:20]:  # Check first 20 lines
        line = line.strip().lower()
        if "completed:" in line or "finished:" in line:
            try:
                # Try to extract date from the line
                date_part = line.split(":", 1)[1].strip()
                return datetime.fromisoformat(date_part)
            except Exception:
                pass

    # Fallback to file modification time
    try:
        mtime = Path(file_path).stat().st_mtime
        return datetime.fromtimestamp(mtime)
    except Exception:
        return None


def determine_content_kind(file_path: str) -> TaskKind:
    """Determine the content kind based on file path."""
    if file_path.startswith("docs/"):
        return TaskKind.DOC
    elif file_path.startswith("design/"):
        return TaskKind.DOC  # Design tokens are documentation
    elif file_path.startswith(("backend/", "frontend/")):
        return TaskKind.CODE
    elif "archive" in file_path:
        return TaskKind.ARCHIVE
    elif "reflection" in file_path:
        return TaskKind.REFLECTION
    else:
        # Fallback to original function
        return determine_task_kind(file_path)
