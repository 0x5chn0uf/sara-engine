"""Core memory indexer functionality."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set

from sara.core.models import IndexedFiles, extract_task_id_from_path

from .content_extractors import extract_code_content, extract_title_from_content
from .file_processors import (
    determine_content_kind,
    extract_completion_date,
    extract_status_from_content,
    should_process_file,
)
from .id_generators import (
    generate_code_id,
    generate_design_id,
    generate_doc_id,
    generate_path_based_id,
    generate_readme_id,
)


class MemoryIndexer:
    """Scans and indexes files from TaskMaster and Serena directories."""

    def __init__(self, memory=None, max_workers: int = 4, watcher=None):
        """Initialize the indexer."""
        self.max_workers = max_workers
        self.watcher = watcher

        # Initialize memory (remote only)
        if memory:
            self.memory = memory
        else:
            # Only remote memory is supported - no fallbacks
            from sara.cli.common import RemoteMemory

            remote_memory = RemoteMemory()
            if not remote_memory.is_server_available():
                raise RuntimeError(
                    "❌ Server not available - indexing requires Serena server to be running. Start it with: sara serve"
                )
            self.memory = remote_memory
            print("✅ Using server-based memory for indexing (async writes enabled)")

        # Use centralized directory and file patterns from settings
        from sara.settings import settings

        # Default scan directories from settings
        self.scan_dirs = settings.index_directories_list + [
            ".taskmaster/memory-bank/reflections",
            ".taskmaster/memory-bank/archives",
        ]

        # File extensions derived from include patterns
        # Extract extensions from glob patterns like *.md, *.txt, etc.
        self.extensions = set()
        for pattern in settings.index_include_globs_list:
            if pattern.startswith("*."):
                self.extensions.add(
                    pattern[1:]
                )  # Remove '*' to get '.md', '.txt', etc.

        # Fallback to common extensions if no patterns found
        if not self.extensions:
            self.extensions = {".md", ".txt", ".json", ".yaml", ".yml"}

        self.strategic_code_paths = {
            "backend/app/models/",
            "backend/app/repositories/",
            "backend/app/services/",
            "backend/app/utils/",
            "backend/app/domain/schemas/",
            "backend/app/api/dependencies.py",
            # Frontend high-value patterns
            "frontend/src/hooks/",
            "frontend/src/components/auth/",
            "frontend/src/services/api.ts",
            "frontend/src/store/",
            "frontend/src/utils/",
        }

        # Track processed files to avoid reprocessing
        self._processed_files: Set[str] = set()

        # Load already processed files from database
        self._load_processed_files()

    def _load_processed_files(self) -> None:
        """Load already processed files from IndexedFiles table."""
        # Only works with RemoteMemory - for local database access
        if hasattr(self.memory, "db_path"):
            db_path = self.memory.db_path
            if db_path and Path(db_path).exists():
                try:
                    from sara.database.session import get_db_session as get_session

                    with get_session(db_path) as session:
                        indexed_files = session.query(IndexedFiles).all()
                        for indexed_file in indexed_files:
                            self._processed_files.add(indexed_file.filepath)
                    print(
                        f"Loaded {len(self._processed_files)} already processed files"
                    )
                except Exception as exc:
                    print(f"Failed to load processed files from database: {exc}")

    def _is_file_already_indexed(self, file_path: str, content_hash: str) -> bool:
        """Check if file is already indexed with same content hash."""
        if hasattr(self.memory, "db_path"):
            db_path = self.memory.db_path
            if db_path and Path(db_path).exists():
                try:
                    from sara.database.session import get_db_session as get_session

                    with get_session(db_path) as session:
                        indexed_file = (
                            session.query(IndexedFiles)
                            .filter_by(filepath=file_path, sha256=content_hash)
                            .first()
                        )
                        return indexed_file is not None
                except Exception as exc:
                    print(f"Error checking if file is indexed: {exc}")
        return False

    def scan_directories(
        self,
        directories: Optional[List[str]] = None,
        force_reindex: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """Scan directories for markdown files and index them."""
        start_time = time.time()

        if directories is None:
            directories = self.scan_dirs

        stats = {
            "files_found": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "directories_scanned": len(directories),
            "scan_time_seconds": 0.0,
            "indexing_time_seconds": 0.0,
            "total_time_seconds": 0.0,
        }

        files_to_process = []
        scan_start = time.time()

        for entry in directories:
            print(f"Scanning directory: {entry}")

            p = Path(entry)

            if p.is_file():
                if p.suffix.lower() in self.extensions:
                    files_to_process.append(str(p))
                else:
                    print(f"Skipping non-supported file: {p}")
                continue

            if not p.exists():
                print(f"Directory not found: {entry}")
                continue

            for file_path in p.rglob("*"):
                if file_path.suffix.lower() in self.extensions:
                    files_to_process.append(str(file_path))

        scan_time = time.time() - scan_start
        stats["scan_time_seconds"] = scan_time
        stats["files_found"] = len(files_to_process)

        if not files_to_process:
            print("No files found to index")
            stats["total_time_seconds"] = time.time() - start_time
            return stats

        print(f"Found {len(files_to_process)} files to process in {scan_time:.2f}s")

        # Process files with progress tracking
        indexing_start = time.time()

        if show_progress:
            self._process_files_with_progress(files_to_process, force_reindex, stats)
        else:
            self._process_files_batch(files_to_process, force_reindex, stats)

        indexing_time = time.time() - indexing_start
        total_time = time.time() - start_time

        stats["indexing_time_seconds"] = indexing_time
        stats["total_time_seconds"] = total_time

        # Calculate throughput
        files_per_second = (
            stats["files_indexed"] / indexing_time if indexing_time > 0 else 0
        )

        print(
            f"Indexing complete: {stats['files_indexed']} files indexed in {total_time:.2f}s "
            f"({files_per_second:.1f} files/sec)"
        )
        return stats

    def scan_files(
        self,
        files: List[str],
        force_reindex: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """Index individual files directly."""
        start_time = time.time()

        stats = {
            "files_found": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "directories_scanned": 0,
            "scan_time_seconds": 0.0,
            "indexing_time_seconds": 0.0,
            "total_time_seconds": 0.0,
        }

        # Validate and filter files
        scan_start = time.time()
        files_to_process = []

        for file_path in files:
            p = Path(file_path)

            if not p.exists():
                print(f"File not found: {file_path}")
                stats["files_failed"] += 1
                continue

            if not p.is_file():
                print(f"Path is not a file: {file_path}")
                stats["files_failed"] += 1
                continue

            if p.suffix.lower() not in self.extensions:
                print(f"Unsupported file extension: {file_path}")
                stats["files_skipped"] += 1
                continue

            files_to_process.append(str(p.resolve()))  # Use absolute path

        scan_time = time.time() - scan_start
        stats["scan_time_seconds"] = scan_time
        stats["files_found"] = len(files_to_process)

        if not files_to_process:
            print("No valid files found to index")
            stats["total_time_seconds"] = time.time() - start_time
            return stats

        print(
            f"Found {len(files_to_process)} valid files to process in {scan_time:.2f}s"
        )

        # Process files with progress tracking
        indexing_start = time.time()

        # Disable progress bar for single file to avoid clutter
        show_progress_bar = show_progress and len(files_to_process) > 1

        if show_progress_bar:
            self._process_files_with_progress(files_to_process, force_reindex, stats)
        else:
            self._process_files_batch(files_to_process, force_reindex, stats)

        indexing_time = time.time() - indexing_start
        total_time = time.time() - start_time

        stats["indexing_time_seconds"] = indexing_time
        stats["total_time_seconds"] = total_time

        # Calculate throughput
        files_per_second = (
            stats["files_indexed"] / indexing_time if indexing_time > 0 else 0
        )

        print(
            f"File indexing complete: {stats['files_indexed']} files indexed in {total_time:.2f}s "
            f"({files_per_second:.1f} files/sec)"
        )
        return stats

    def _process_files_with_progress(
        self, files: List[str], force_reindex: bool, stats: Dict[str, int]
    ) -> None:
        """Process files with progress display."""
        self._process_files(files, force_reindex, stats, show_progress=True)

    def _process_files_batch(
        self, files: List[str], force_reindex: bool, stats: Dict[str, int]
    ) -> None:
        """Process files in parallel batches."""
        self._process_files(files, force_reindex, stats, show_progress=False)

    def _process_files(
        self,
        files: List[str],
        force_reindex: bool,
        stats: Dict[str, int],
        show_progress: bool = True,
    ) -> None:
        """Process files with unified logic for both progress and batch processing."""
        total_files = len(files)
        processing_start = time.time()

        if show_progress:
            # Sequential processing with progress display
            for i, file_path in enumerate(files, 1):
                try:
                    if self._process_single_file(file_path, force_reindex):
                        stats["files_indexed"] += 1
                    else:
                        stats["files_skipped"] += 1

                    # Show progress every 10 files or at end
                    if i % 10 == 0 or i == total_files:
                        elapsed = time.time() - processing_start
                        progress = (i / total_files) * 100
                        rate = i / elapsed if elapsed > 0 else 0
                        eta = (total_files - i) / rate if rate > 0 else 0

                        print(
                            f"Progress: {i}/{total_files} ({progress:.1f}%) - "
                            f"Indexed: {stats['files_indexed']}, "
                            f"Skipped: {stats['files_skipped']} - "
                            f"Rate: {rate:.1f} files/s - "
                            f"ETA: {eta:.0f}s"
                        )

                except Exception as e:  # noqa: BLE001
                    print(f"Failed to process {file_path}: {e}")
                    stats["files_failed"] += 1
        else:
            # Enhanced parallel processing - always use individual processing for remote API
            # (RemoteMemory doesn't support batch operations)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []

                for file_path in files:
                    future = executor.submit(
                        self._process_single_file, file_path, force_reindex
                    )
                    futures.append((future, file_path))

                # Collect results
                for future, file_path in futures:
                    try:
                        if future.result():
                            stats["files_indexed"] += 1
                        else:
                            stats["files_skipped"] += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"Failed to process {file_path}: {e}")
                        stats["files_failed"] += 1

    def _process_single_file(self, file_path: str, force_reindex: bool) -> bool:
        """Process a single file for indexing."""
        try:
            # Read file content first to compute hash
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
            except Exception:  # noqa: BLE001
                print(f"Failed to read file {file_path}")
                return False

            if not raw_content.strip():
                print(f"Empty file: {file_path}")
                return False

            # Apply smart content extraction for code files
            if file_path.startswith(("backend/", "frontend/")):
                content = extract_code_content(file_path, raw_content)
            else:
                content = raw_content

            # Compute content hash for duplicate detection
            from sara.core.models import compute_content_hash

            content_hash = compute_content_hash(content)

            # Check if file is already indexed with same content
            if not force_reindex and self._is_file_already_indexed(
                file_path, content_hash
            ):
                self._processed_files.add(file_path)
                print(f"File already indexed with same content: {file_path}")
                return False

            # Extract task ID from file path or generate one for docs/design/code
            task_id = extract_task_id_from_path(file_path)
            if not task_id:
                task_id = self._generate_id_for_file(file_path)
                if not task_id:
                    print(f"Could not extract task ID from {file_path}")
                    return False

            # Determine metadata
            kind = determine_content_kind(file_path)
            title = extract_title_from_content(content, file_path)
            status = extract_status_from_content(content)
            completed_at = extract_completion_date(content, file_path)

            # Index the file via server API (RemoteMemory only)
            success = self.memory.upsert(
                task_id=task_id,
                markdown_text=content,
                filepath=file_path,
                title=title,
                kind=kind.value if kind else "archive",
                status=status.value if status else None,
                completed_at=completed_at.isoformat() if completed_at else None,
            )

            if success:
                self._processed_files.add(file_path)

                # Update IndexedFiles table if using local database
                self._update_indexed_files_record(
                    file_path, content_hash, kind, task_id
                )

                print(f"Indexed {file_path} as task {task_id}")

                # Notify watcher to auto-add directory if configured
                if self.watcher:
                    self.watcher.add_directory_for_file(file_path)

                return True
            else:
                print(f"Failed to index {file_path}")
                return False

        except Exception:  # noqa: BLE001
            print(f"Error processing file {file_path}")
            return False

    def _update_indexed_files_record(
        self, file_path: str, content_hash: str, kind, task_id: str
    ) -> None:
        """Update IndexedFiles table with indexed file information."""
        if hasattr(self.memory, "db_path"):
            db_path = self.memory.db_path
            if db_path and Path(db_path).exists():
                try:
                    from sara.database.session import get_db_session as get_session
                    from datetime import datetime

                    file_stat = Path(file_path).stat()
                    file_size = file_stat.st_size
                    last_modified = datetime.fromtimestamp(file_stat.st_mtime)

                    with get_session(db_path) as session:
                        # Try to find existing record
                        existing = (
                            session.query(IndexedFiles)
                            .filter_by(filepath=file_path)
                            .first()
                        )

                        if existing:
                            # Update existing record
                            existing.sha256 = content_hash
                            existing.kind = (
                                kind.value if hasattr(kind, "value") else str(kind)
                            )
                            existing.task_id = task_id
                            existing.file_size = file_size
                            existing.last_modified = last_modified
                            existing.updated_at = datetime.now()
                        else:
                            # Create new record
                            indexed_file = IndexedFiles(
                                filepath=file_path,
                                sha256=content_hash,
                                kind=(
                                    kind.value if hasattr(kind, "value") else str(kind)
                                ),
                                task_id=task_id,
                                file_size=file_size,
                                last_modified=last_modified,
                                indexed_at=datetime.now(),
                                updated_at=datetime.now(),
                            )
                            session.add(indexed_file)

                        session.commit()
                        print(f"Updated IndexedFiles record for {file_path}")

                except Exception as exc:
                    print(
                        f"Failed to update IndexedFiles record for {file_path}: {exc}"
                    )

    def _generate_id_for_file(self, file_path: str) -> Optional[str]:
        """Generate appropriate ID based on file type and location."""
        # For documentation files, generate ID from path
        if file_path.startswith("docs/"):
            return generate_doc_id(file_path)
        elif file_path.startswith("design/"):
            return generate_design_id(file_path)
        elif file_path.startswith(("backend/", "frontend/")):
            return generate_code_id(file_path)
        elif file_path.startswith((".taskmaster/", ".serena/")):
            # Generate path-based ID for TaskMaster and Serena files
            return generate_path_based_id(file_path, Path(file_path).stem)
        elif Path(file_path).name.lower().startswith("readme"):
            return generate_readme_id(file_path)
        else:
            # Handle other common duplicate filenames
            filename = Path(file_path).stem.lower()
            if filename in ["index", "main", "config", "settings"]:
                return generate_path_based_id(file_path, filename)

            # For arbitrary files (e.g., individual files being indexed),
            # generate ID from filename and parent directory to avoid collisions
            path_obj = Path(file_path)
            parent_name = (
                path_obj.parent.name if path_obj.parent.name != "." else "root"
            )

            # Clean up dots from parent name to avoid consecutive dots in ID
            # .taskmaster.memory-bank -> taskmaster-memory-bank
            clean_parent = parent_name.lstrip(".").replace(".", "-")
            if not clean_parent:
                clean_parent = "root"

            return f"{clean_parent}-{path_obj.stem}"

        return None

    def watch_directories(self, directories: Optional[List[str]] = None) -> None:
        """Watch directories for changes and auto-index new/modified files."""
        if directories is None:
            directories = self.scan_dirs

        if not self.watcher:
            print("Creating new file watcher for continuous monitoring")
            from sara.infrastructure.watcher import create_memory_watcher

            def indexer_callback(action: str, task_id: str, path: str) -> None:
                """Callback for watcher events."""
                print(f"File {action}: {task_id} at {path}")

            self.watcher = create_memory_watcher(
                memory=self.memory, auto_add_taskmaster=True, callback=indexer_callback
            )

        # Add all files in specified directories to tracking
        files_to_track = []
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if (
                        file_path.suffix.lower() in self.extensions
                        and file_path.is_file()
                    ):
                        files_to_track.append(str(file_path))

        if files_to_track:
            added_count = self.watcher.add_files_to_tracking(files_to_track)
            print(f"Added {added_count} files to watcher tracking")

        # Start the watcher
        try:
            self.watcher.start(catch_up=True)
            print(f"File watcher started, monitoring {len(directories)} directories")

            # Log tracking statistics
            stats = self.watcher.get_tracking_stats()
            print(f"Watcher stats: {stats}")

        except Exception as exc:
            print(f"Failed to start file watcher: {exc}")
            raise

    def stop_watching(self) -> None:
        """Stop the file watcher if running."""
        if self.watcher and self.watcher.is_running():
            self.watcher.stop()
            print("File watcher stopped")
        else:
            print("No active file watcher to stop")
