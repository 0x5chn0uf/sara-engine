"""
Code embedding system for selective codebase indexing and semantic search.

This module provides functionality to selectively embed parts of the codebase
into the memory database for semantic searching, without exposing the entire
source code in chat contexts.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.models import CodeEmbedding, Files
from settings import settings
from database.session import get_db_session
from infrastructure.embeddings import get_default_generator


class CodeChunker:
    """Handles chunking of code files into manageable pieces for embedding."""

    def __init__(
        self,
        max_chunk_size: int = 4096,  # 4KB chunks
        overlap_lines: int = 20,  # 20-line overlap
        strip_comments: bool = True,
    ):
        """Initialize the code chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in bytes
            overlap_lines: Number of lines to overlap between chunks
            strip_comments: Whether to strip comments and docstrings
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_lines = overlap_lines
        self.strip_comments = strip_comments

    def chunk_code_file(self, filepath: str, content: str) -> List[Dict]:
        """Chunk a code file into overlapping segments.

        Args:
            filepath: Path to the code file
            content: Raw file content

        Returns:
            List of chunk dictionaries with metadata
        """
        # Preprocess content if needed
        if self.strip_comments:
            content = self._strip_comments(content, filepath)

        lines = content.split("\n")
        chunks = []

        if not lines:
            return chunks

        chunk_id = 0
        start_line = 0

        while start_line < len(lines):
            # Calculate end line for this chunk
            end_line = self._find_chunk_end(lines, start_line)

            # Extract chunk content
            chunk_lines = lines[start_line : end_line + 1]
            chunk_content = "\n".join(chunk_lines)

            # Skip empty chunks
            if not chunk_content.strip():
                start_line = end_line + 1
                continue

            # Create chunk metadata
            chunk = {
                "chunk_id": chunk_id,
                "start_line": start_line + 1,  # 1-indexed for humans
                "end_line": end_line + 1,  # 1-indexed for humans
                "content": chunk_content,
                "content_preview": self._create_preview(chunk_content),
                "size_bytes": len(chunk_content.encode("utf-8")),
            }

            chunks.append(chunk)
            chunk_id += 1

            # Calculate next start line with overlap
            next_start = max(start_line + 1, end_line + 1 - self.overlap_lines)
            if next_start <= start_line:
                break  # Prevent infinite loops
            start_line = next_start

        return chunks

    def _find_chunk_end(self, lines: List[str], start_line: int) -> int:
        """Find the optimal end line for a chunk based on size and structure.

        Args:
            lines: All lines in the file
            start_line: Starting line index (0-indexed)

        Returns:
            End line index (0-indexed, inclusive)
        """
        current_size = 0
        current_line = start_line

        # Try to find a natural break point (function/class boundary)
        natural_break = None
        in_function = False
        brace_level = 0

        while current_line < len(lines):
            line = lines[current_line]
            line_size = len(line.encode("utf-8")) + 1  # +1 for newline

            # Check if adding this line would exceed size limit
            if (
                current_size + line_size > self.max_chunk_size
                and current_line > start_line
            ):
                break

            current_size += line_size

            # Track code structure for natural breaks
            stripped = line.strip()

            # Python/TypeScript function/class detection
            if re.match(r"^(def |class |function |export |const \w+\s*=)", stripped):
                if current_line > start_line:  # Don't break immediately
                    natural_break = current_line - 1
                in_function = True
            elif re.match(r"^(}|^$)", stripped) and in_function:
                natural_break = current_line
                in_function = False

            # Track brace level for languages like JavaScript/Java
            brace_level += stripped.count("{") - stripped.count("}")

            current_line += 1

        # Use natural break if we found one and it's reasonable
        if natural_break is not None and natural_break > start_line:
            return min(natural_break, current_line - 1)

        return min(current_line - 1, len(lines) - 1)

    def _strip_comments(self, content: str, filepath: str) -> str:
        """Strip comments and docstrings from code content.

        Args:
            content: Raw file content
            filepath: File path to determine language

        Returns:
            Content with comments stripped
        """
        ext = Path(filepath).suffix.lower()

        if ext == ".py":
            return self._strip_python_comments(content)
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            return self._strip_js_comments(content)
        else:
            # For other languages, use basic line comment stripping
            return self._strip_basic_comments(content)

    def _strip_python_comments(self, content: str) -> str:
        """Strip Python comments and docstrings."""
        # Remove docstrings (triple quotes)
        content = re.sub(r'"""[\s\S]*?"""', "", content)
        content = re.sub(r"'''[\s\S]*?'''", "", content)

        # Remove line comments
        lines = []
        for line in content.split("\n"):
            # Find # not inside strings
            in_string = False
            quote_char = None
            i = 0
            while i < len(line):
                char = line[i]
                if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == "#" and not in_string:
                    line = line[:i].rstrip()
                    break
                i += 1
            lines.append(line)

        return "\n".join(lines)

    def _strip_js_comments(self, content: str) -> str:
        """Strip JavaScript/TypeScript comments."""
        # Remove multi-line comments /* */
        content = re.sub(r"/\*[\s\S]*?\*/", "", content)

        # Remove line comments //
        lines = []
        for line in content.split("\n"):
            # Find // not inside strings
            in_string = False
            quote_char = None
            i = 0
            while i < len(line) - 1:
                char = line[i]
                if char in ('"', "'", "`") and (i == 0 or line[i - 1] != "\\"):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == "/" and line[i + 1] == "/" and not in_string:
                    line = line[:i].rstrip()
                    break
                i += 1
            lines.append(line)

        return "\n".join(lines)

    def _strip_basic_comments(self, content: str) -> str:
        """Strip basic line comments for other languages."""
        # Remove common comment patterns
        lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            # Skip lines that are entirely comments
            if not (
                stripped.startswith("//")
                or stripped.startswith("#")
                or stripped.startswith("*")
            ):
                lines.append(line)
        return "\n".join(lines)

    def _create_preview(self, content: str, max_length: int = 200) -> str:
        """Create a preview of chunk content for search display.

        Args:
            content: Full chunk content
            max_length: Maximum preview length

        Returns:
            Preview string
        """
        # Take first few meaningful lines
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        preview_lines = []
        current_length = 0

        for line in lines[:5]:  # Max 5 lines
            if current_length + len(line) > max_length:
                break
            preview_lines.append(line)
            current_length += len(line) + 1

        preview = " | ".join(preview_lines)
        if len(preview) > max_length:
            preview = preview[: max_length - 3] + "..."

        return preview


class RepositoryCrawler:
    """Crawls repository for code files to embed based on inclusion/exclusion patterns."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
    ):
        """Initialize the repository crawler.

        Args:
            project_root: Root directory to crawl
            include_globs: Glob patterns for files to include
            exclude_globs: Glob patterns for files to exclude
        """
        self.project_root = Path(project_root or ".")

        # Use centralized patterns from settings, with fallback to provided patterns
        if include_globs is not None:
            self.include_globs = include_globs
        else:
            from settings import settings
            self.include_globs = settings.embedding_include_globs_list

        if exclude_globs is not None:
            self.exclude_globs = exclude_globs
        else:
            from settings import settings
            self.exclude_globs = settings.embedding_exclude_globs_list

    def find_code_files(self) -> List[str]:
        """Find all code files matching inclusion/exclusion patterns.

        Returns:
            List of file paths relative to project root
        """
        matching_files = set()

        # Find files matching include patterns
        for pattern in self.include_globs:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.project_root)
                    matching_files.add(str(relative_path))

        # Remove files matching exclude patterns
        excluded_files = set()
        for pattern in self.exclude_globs:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.project_root)
                    excluded_files.add(str(relative_path))

        return list(matching_files - excluded_files)

    def get_file_info(self, filepath: str) -> Dict:
        """Get file metadata for change detection.

        Args:
            filepath: Relative path to file

        Returns:
            Dictionary with file metadata
        """
        full_path = self.project_root / filepath

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        stat = full_path.stat()

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "filepath": filepath,
            "size": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content": content,
        }


class CodeEmbeddingSystem:
    """Main system for code embedding with incremental updates."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        """Initialize the code embedding system.

        Args:
            project_root: Root directory of the project
            db_path: Path to SQLite database
        """
        self.project_root = project_root or "."
        self.db_path = db_path
        self.chunker = CodeChunker(
            max_chunk_size=settings.embedding_chunk_size,
            overlap_lines=settings.embedding_overlap_lines,
            strip_comments=settings.embedding_strip_comments,
        )
        self.crawler = RepositoryCrawler(project_root)
        self.embedding_generator = get_default_generator()

    def embed_codebase(self, force_reindex: bool = False) -> Dict[str, int]:
        """Embed the entire codebase with selective file processing.

        Args:
            force_reindex: Whether to reindex all files regardless of changes

        Returns:
            Statistics about the embedding process
        """
        stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": 0,
        }

        print("🔍 Discovering code files...")
        code_files = self.crawler.find_code_files()
        stats["files_found"] = len(code_files)

        if not code_files:
            print("No code files found matching criteria")
            return stats

        print(f"Found {len(code_files)} code files to process")

        with get_db_session(self.db_path) as session:
            for filepath in code_files:
                try:
                    processed = self._process_file(session, filepath, force_reindex)
                    if processed:
                        stats["files_processed"] += 1
                        # Stats will be updated by _process_file
                    else:
                        stats["files_skipped"] += 1

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    stats["errors"] += 1

        return stats

    def embed_file(self, filepath: str, force_reindex: bool = False) -> bool:
        """Embed a single file.

        Args:
            filepath: Path to the file to embed
            force_reindex: Whether to force reindexing

        Returns:
            True if file was processed, False if skipped
        """
        with get_db_session(self.db_path) as session:
            return self._process_file(session, filepath, force_reindex)

    def _process_file(self, session, filepath: str, force_reindex: bool) -> bool:
        """Process a single file for embedding.

        Args:
            session: Database session
            filepath: Path to file
            force_reindex: Whether to force reindexing

        Returns:
            True if processed, False if skipped
        """
        try:
            # Get file info
            file_info = self.crawler.get_file_info(filepath)

            # Check if file already exists and is up to date
            from core.models import IndexedFiles
            
            existing_file = session.query(Files).filter_by(filepath=filepath).first()
            existing_indexed = session.query(IndexedFiles).filter_by(filepath=filepath).first()

            if not force_reindex and existing_indexed:
                if existing_indexed.sha256 == file_info["sha256"]:
                    print(f"⏭️  Skipping {filepath} (unchanged)")
                    return False

            print(f"📝 Processing {filepath}")

            # Chunk the file
            chunks = self.chunker.chunk_code_file(filepath, file_info["content"])

            if not chunks:
                print(f"⚠️  No chunks generated for {filepath}")
                return False

            # Create or update file record
            if existing_file:
                existing_file.sha256 = file_info["sha256"]
                existing_file.file_size = file_info["size"]
                existing_file.last_modified = file_info["last_modified"]
                existing_file.updated_at = datetime.now()
                # Remove old embeddings
                session.query(CodeEmbedding).filter_by(
                    file_id=existing_file.id
                ).delete()
                file_record = existing_file
            else:
                file_record = Files(
                    filepath=filepath,
                    sha256=file_info["sha256"],
                    kind="code",
                    file_size=file_info["size"],
                    last_modified=file_info["last_modified"],
                    start_line=1,
                    end_line=len(file_info["content"].split("\n")),
                )
                session.add(file_record)
                session.flush()  # Get the ID

            # Create or update indexed_files tracking record
            from core.models import IndexedFiles
            
            existing_indexed = session.query(IndexedFiles).filter_by(filepath=filepath).first()
            if existing_indexed:
                # Update existing record
                existing_indexed.sha256 = file_info["sha256"]
                existing_indexed.file_size = file_info["size"]
                existing_indexed.last_modified = file_info["last_modified"]
                existing_indexed.updated_at = datetime.now()
            else:
                # Create new tracking record
                indexed_record = IndexedFiles(
                    filepath=filepath,
                    sha256=file_info["sha256"],
                    kind="code",
                    file_size=file_info["size"],
                    last_modified=file_info["last_modified"],
                    indexed_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(indexed_record)

            # Generate embeddings for chunks
            embeddings_created = 0
            for chunk in chunks:
                try:
                    # Generate embedding
                    vector = self.embedding_generator.generate_embedding(
                        chunk["content"]
                    )

                    # Create embedding record
                    embedding = CodeEmbedding(
                        file_id=file_record.id,
                        chunk_id=chunk["chunk_id"],
                        start_line=chunk["start_line"],
                        end_line=chunk["end_line"],
                        vector=self._vector_to_bytes(vector),
                        content_preview=chunk["content_preview"],
                    )
                    session.add(embedding)
                    embeddings_created += 1

                except Exception as e:
                    print(
                        f"⚠️  Failed to embed chunk {chunk['chunk_id']} of {filepath}: {e}"
                    )

            session.commit()
            print(
                f"✅ Embedded {filepath}: {len(chunks)} chunks, {embeddings_created} embeddings"
            )
            return True

        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            session.rollback()
            return False

    def _vector_to_bytes(self, vector: List[float]) -> bytes:
        """Convert vector to bytes for storage."""
        import numpy as np

        return np.array(vector, dtype=np.float32).tobytes()

    def search_code(self, query: str, limit: int = 10) -> List[Dict]:
        """Search embedded code using semantic similarity.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results with metadata
        """
        # Generate query embedding
        query_vector = self.embedding_generator.generate_embedding(query)
        query_bytes = self._vector_to_bytes(query_vector)

        results = []

        # This is a simplified implementation
        # In practice, you'd want to use a vector database or efficient similarity search
        with get_db_session(self.db_path) as session:
            embeddings = session.query(CodeEmbedding).join(Files).limit(1000).all()

            for embedding in embeddings:
                # Calculate similarity (simplified cosine similarity)
                similarity = self._calculate_similarity(query_bytes, embedding.vector)

                if similarity > 0.1:  # Threshold
                    results.append(
                        {
                            "filepath": embedding.file.filepath,
                            "chunk_id": embedding.chunk_id,
                            "start_line": embedding.start_line,
                            "end_line": embedding.end_line,
                            "similarity": similarity,
                            "preview": embedding.content_preview,
                        }
                    )

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def _calculate_similarity(self, vec1_bytes: bytes, vec2_bytes: bytes) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        vec1 = np.frombuffer(vec1_bytes, dtype=np.float32)
        vec2 = np.frombuffer(vec2_bytes, dtype=np.float32)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norms == 0:
            return 0.0

        return float(dot_product / norms)

    def get_stats(self) -> Dict:
        """Get embedding statistics."""
        with get_db_session(self.db_path) as session:
            from core.models import IndexedFiles
            
            file_count = session.query(Files).count()
            embedding_count = session.query(CodeEmbedding).count()
            indexed_files_count = session.query(IndexedFiles).filter_by(kind="code").count()

            return {
                "files_indexed": file_count,
                "embeddings_generated": embedding_count,
                "indexed_files_tracked": indexed_files_count,
                "average_chunks_per_file": (
                    embedding_count / file_count if file_count > 0 else 0
                ),
            }
