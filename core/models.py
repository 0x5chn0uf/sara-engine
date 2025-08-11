"""
Domain data models and utility helpers.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import (REAL, Boolean, CheckConstraint, DateTime, ForeignKey,
                        Integer, LargeBinary, String, Text, func)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# SQLAlchemy Base and Models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Archive(Base):
    """Archive model for storing task memories."""

    __tablename__ = "archives"

    task_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    filepath: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="archive")
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    embedding_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )
    embedding_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_embedded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    # Relationships
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding", back_populates="archive", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "kind IN ('archive', 'reflection', 'doc', 'rule', 'code')",
            name="ck_archives_kind",
        ),
    )


class Embedding(Base):
    """Embedding model for storing vector embeddings."""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("archives.task_id"), nullable=False
    )
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Relationships
    archive: Mapped["Archive"] = relationship("Archive", back_populates="embeddings")


class IndexedFiles(Base):
    """Track which files have been indexed to prevent duplicate processing."""

    __tablename__ = "indexed_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filepath: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_modified: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "kind IN ('archive', 'reflection', 'doc', 'rule', 'code')",
            name="ck_indexed_files_kind",
        ),
    )


class Files(Base):
    """Files model for storing code file metadata and chunks."""

    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filepath: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="code")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_modified: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    # Relationships
    embeddings: Mapped[list["CodeEmbedding"]] = relationship(
        "CodeEmbedding", back_populates="file", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "kind IN ('code', 'doc', 'config')",
            name="ck_files_kind",
        ),
    )


class CodeEmbedding(Base):
    """Code embedding model for storing vector embeddings of code chunks."""

    __tablename__ = "code_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("files.id"), nullable=False
    )
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    content_preview: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now()
    )

    # Relationships
    file: Mapped["Files"] = relationship("Files", back_populates="embeddings")


class MaintenanceLog(Base):
    """Maintenance log model for tracking database operations."""

    __tablename__ = "maintenance_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    db_size_before_mb: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    db_size_after_mb: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    space_saved_bytes: Mapped[int] = mapped_column(Integer, nullable=True, default=0)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "operation IN ('checkpoint', 'vacuum', 'health_check')",
            name="ck_maintenance_log_operation",
        ),
    )


# Task Enums and Dataclasses
class TaskKind(Enum):
    """Types of tasks/documents in the memory bridge."""

    ARCHIVE = "archive"
    REFLECTION = "reflection"
    DOC = "doc"
    RULE = "rule"
    CODE = "code"


class TaskStatus(Enum):
    """Task status values."""

    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    DONE = "done"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class ArchiveRecord:
    """Represents a task archive record in the database."""

    task_id: str
    title: str
    filepath: str
    sha256: str
    kind: TaskKind
    status: Optional[TaskStatus] = None
    completed_at: Optional[datetime] = None
    embedding_id: Optional[int] = None
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    embedding_version: int = 1
    last_embedded_at: Optional[datetime] = None

    @classmethod
    def from_content(
        cls,
        task_id: str,
        title: str,
        filepath: str,
        content: str,
        kind: TaskKind,
        status: Optional[TaskStatus] = None,
        completed_at: Optional[datetime] = None,
    ) -> "ArchiveRecord":
        """Create an ArchiveRecord from content with computed SHA-256."""
        sha256 = compute_content_hash(content)
        return cls(
            task_id=task_id,
            title=title,
            filepath=filepath,
            sha256=sha256,
            kind=kind,
            status=status,
            completed_at=completed_at,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )


@dataclass
class EmbeddingRecord:
    """Represents an embedding record in the database."""

    id: Optional[int]
    task_id: str
    chunk_id: int
    position: int
    vector: bytes

    @classmethod
    def from_vector(
        cls, task_id: str, vector: List[float], chunk_id: int = 0, position: int = 0
    ) -> "EmbeddingRecord":
        """Create an EmbeddingRecord from a vector."""
        import numpy as np

        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        return cls(
            id=None,
            task_id=task_id,
            chunk_id=chunk_id,
            position=position,
            vector=vector_bytes,
        )

    def to_vector(self) -> List[float]:
        """Convert bytes back to vector."""
        import numpy as np

        return np.frombuffer(self.vector, dtype=np.float32).tolist()


@dataclass
class SearchResult:
    """Represents a search result with metadata."""

    task_id: str
    title: str
    score: float
    excerpt: str
    kind: TaskKind
    status: Optional[TaskStatus]
    completed_at: Optional[datetime]
    filepath: str

    def __str__(self) -> str:
        """Human-readable string representation."""
        status_str = f" [{self.status.value}]" if self.status else ""
        date_str = f" ({self.completed_at.date()})" if self.completed_at else ""
        return f"[{self.score:.3f}] {self.task_id}: {self.title}{status_str}{date_str}"


@dataclass
class HealthInfo:
    """Database health and statistics information."""

    archive_count: int
    embedding_count: int
    last_migration: Optional[datetime]
    wal_checkpoint_age: Optional[int]  # seconds since last checkpoint
    database_size: int  # bytes
    embedding_versions: dict  # version -> count mapping

    def __str__(self) -> str:
        """Human-readable health summary."""
        size_mb = self.database_size / (1024 * 1024)
        return (
            f"Archives: {self.archive_count}, "
            f"Embeddings: {self.embedding_count}, "
            f"Size: {size_mb:.1f}MB"
        )


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of cleaned content for deduplication.

    Args:
        content: Raw markdown content

    Returns:
        str: Hexadecimal SHA-256 hash
    """
    # Clean content by normalizing whitespace and removing metadata
    cleaned = content.strip()

    # Remove YAML frontmatter if present
    if cleaned.startswith("---"):
        parts = cleaned.split("---", 2)
        if len(parts) >= 3:
            cleaned = parts[2].strip()

    # Normalize line endings and compress whitespace
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())

    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def extract_task_id_from_path(filepath: str) -> Optional[str]:
    """
    Extract task ID from file path.

    Args:
        filepath: Path to the file

    Returns:
        Optional[str]: Extracted task ID or None
    """
    from pathlib import Path

    filename = Path(filepath).stem

    # Handle various naming patterns with type prefixes to avoid collisions
    # archive-123.md -> archive-123
    # reflection-123.md -> reflection-123  
    # task-123.md -> task-123
    # 123.md -> 123

    if filename.startswith(("archive-", "reflection-", "task-")):
        # Keep the full prefix to avoid ID collisions between archive-50 and reflection-50
        return filename
    elif filename.isdigit():
        return filename
    elif "." in filename and filename.split(".")[0].isdigit():
        return filename.split(".")[0]

    # For common duplicate files, return None to trigger path-based ID generation
    # This allows the indexer to create unique IDs based on directory structure
    common_duplicates = ["readme", "index", "main", "config", "settings"]
    if filename.lower() in common_duplicates:
        return None

    return None


def determine_task_kind(filepath: str) -> TaskKind:
    """
    Determine task kind from file path.

    Args:
        filepath: Path to the file

    Returns:
        TaskKind: Determined kind
    """
    from pathlib import Path
    
    filepath_lower = filepath.lower()
    file_ext = Path(filepath).suffix.lower()
    
    # Check for code files first by extension
    code_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx', '.go', '.rs', '.java', '.cpp', '.c', '.h', '.hpp'}
    if file_ext in code_extensions:
        return TaskKind.CODE
    
    # Check for code directories (backend, frontend, src)
    if any(code_dir in filepath_lower for code_dir in ['backend/', 'frontend/', 'src/', '/app/', '/components/', '/services/', '/utils/']):
        return TaskKind.CODE

    # Check for specific content types in filepath
    if "archive" in filepath_lower:
        return TaskKind.ARCHIVE
    elif "reflection" in filepath_lower:
        return TaskKind.REFLECTION
    elif any(
        doc_indicator in filepath_lower for doc_indicator in ["doc", "readme", "guide"]
    ):
        return TaskKind.DOC
    elif "rule" in filepath_lower:
        return TaskKind.RULE
    else:
        # Default to archive for most task files
        return TaskKind.ARCHIVE


def generate_summary(content: str, max_length: int = 400) -> str:
    """
    Generate a brief summary of the content.

    Args:
        content: Full content text
        max_length: Maximum summary length

    Returns:
        str: Generated summary
    """
    # Simple extractive summary - take first meaningful paragraphs
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    # Skip YAML frontmatter
    start_idx = 0
    if lines and lines[0].startswith("---"):
        for i, line in enumerate(lines[1:], 1):
            if line.startswith("---"):
                start_idx = i + 1
                break

    # Find first substantial paragraph
    summary_parts: list[str] = []
    current_length = 0

    for line in lines[start_idx:]:
        # Skip markdown headers and empty lines
        if line.startswith("#") or not line:
            continue

        # Add line if it fits
        if current_length + len(line) <= max_length:
            summary_parts.append(line)
            current_length += len(line) + 1  # +1 for space
        else:
            # Truncate last line if needed
            remaining = max_length - current_length
            if remaining > 20:  # Only truncate if reasonable space left
                summary_parts.append(line[: remaining - 3] + "...")
            break

    return " ".join(summary_parts) if summary_parts else content[:max_length]


# Re-export public names for * import convenience inside the package
__all__ = [
    # SQLAlchemy Models
    "Base",
    "Archive",
    "Embedding",
    "IndexedFiles",
    "Files",
    "CodeEmbedding",
    "MaintenanceLog",
    # Enums
    "TaskKind",
    "TaskStatus",
    # Dataclasses
    "ArchiveRecord",
    "EmbeddingRecord",
    "SearchResult",
    "HealthInfo",
    # Helpers
    "compute_content_hash",
    "extract_task_id_from_path",
    "determine_task_kind",
    "generate_summary",
]
