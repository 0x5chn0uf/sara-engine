# -*- coding: utf-8 -*-
"""File-watcher for Serena"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from watchdog.events import FileSystemEventHandler  # noqa: WPS433
from watchdog.observers import Observer

from core.models import (
    Archive,
    Embedding,
    IndexedFiles,
    compute_content_hash,
    determine_task_kind,
    extract_task_id_from_path,
)
from database.session import get_db_session as get_session

logger = logging.getLogger(__name__)


@dataclass
class TrackedFile:
    """Represents a file being tracked by the watcher."""
    
    task_id: str
    filepath: str
    sha256: str = ""
    last_modified: Optional[datetime] = None
    file_size: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Initialize file metadata if not provided."""
        if not self.last_modified or not self.file_size:
            self._update_metadata()
    
    def _update_metadata(self) -> None:
        """Update file metadata from filesystem."""
        try:
            path = Path(self.filepath)
            if path.exists():
                stat = path.stat()
                self.last_modified = datetime.fromtimestamp(stat.st_mtime)
                self.file_size = stat.st_size
        except Exception as exc:
            logger.debug(f"Failed to update metadata for {self.filepath}: {exc}")
    
    def has_changed(self) -> bool:
        """Check if file has changed since last tracking."""
        try:
            path = Path(self.filepath)
            if not path.exists():
                return True  # File deleted
            
            stat = path.stat()
            current_modified = datetime.fromtimestamp(stat.st_mtime)
            current_size = stat.st_size
            
            # Check if modification time or size changed
            if (self.last_modified != current_modified or 
                self.file_size != current_size):
                return True
            
            return False
        except Exception as exc:
            logger.debug(f"Error checking file changes for {self.filepath}: {exc}")
            return True  # Assume changed on error
    
    def update_tracking(self, new_sha256: str) -> None:
        """Update tracking information after successful indexing."""
        self.sha256 = new_sha256
        self._update_metadata()
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "task_id": self.task_id,
            "sha256": self.sha256
        }


class _SerenaEventHandler(FileSystemEventHandler):
    """Handle FS events and dispatch to the parent watcher."""

    def __init__(self, watcher: "_WatchdogMemoryWatcher") -> None:
        super().__init__()
        self._watcher = watcher

    def on_modified(self, event):  # noqa: D401, ANN001
        if event.is_directory:
            return
        self._watcher.process_change(event.src_path)

    def on_created(self, event):  # noqa: D401, ANN001
        if event.is_directory:
            return
        self._watcher.process_change(event.src_path)

    def on_deleted(self, event):  # noqa: D401, ANN001
        if event.is_directory:
            return
        self._watcher.process_deletion(event.src_path)


class _WatchdogMemoryWatcher:
    """Observes markdown archives and keeps the memory index up-to-date."""

    def __init__(
        self,
        *,
        memory,
        auto_add_taskmaster: bool = True,
        callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.memory = memory
        self.auto_add_taskmaster = auto_add_taskmaster
        self.callback = callback or (lambda *_: None)

        self._running = False
        self._observer: Optional[Observer] = None

        # path -> TrackedFile instance
        self._tracked: Dict[str, TrackedFile] = {}
        
        # Debouncing mechanism to prevent duplicate events
        self._pending_changes: Dict[str, float] = {}  # path -> timestamp
        self._debounce_delay = 0.5  # 500ms debounce delay

        self._setup_tracking()

    @property
    def watched_paths(self) -> List[str]:  # noqa: D401
        """List of directories being watched (for CLI display)."""

        dirs: Set[str] = {str(Path(p).parent) for p in self._tracked.keys()}
        return sorted(dirs)

    def start(self, catch_up: bool = True):  # noqa: D401, ANN001
        """Start the observer thread."""

        if self._running:
            return

        if not self._tracked:
            print("No files/directories to watch – falling back to no-op")
            return

        # Perform offline catch-up scan BEFORE starting observer so that we
        # don’t miss rapid edits during startup.
        if catch_up:
            self._initial_crawl()

        self._observer = Observer()
        handler = _SerenaEventHandler(self)

        for directory in self.watched_paths:
            self._observer.schedule(handler, path=directory, recursive=True)

        self._observer.start()
        self._running = True

    def stop(self):  # noqa: D401
        if not self._running or self._observer is None:
            return

        self._observer.stop()
        self._observer.join(timeout=5)
        self._running = False

    def is_running(self) -> bool:  # noqa: D401
        return self._running

    def _setup_tracking(self) -> None:
        """Populate *self._tracked* with existing indexed files and archives."""

        db_path = getattr(self.memory, "db_path", None)
        if db_path and Path(db_path).exists():
            try:
                with get_session(db_path) as session:
                    # First load from indexed_files table for accurate tracking
                    indexed_files = session.query(IndexedFiles).all()
                    for indexed_file in indexed_files:
                        self._tracked[indexed_file.filepath] = TrackedFile(
                            task_id=indexed_file.task_id or indexed_file.filepath,
                            filepath=indexed_file.filepath,
                            sha256=indexed_file.sha256,
                            last_modified=indexed_file.last_modified,
                            file_size=indexed_file.file_size,
                        )
                    
                    # Fallback to archives table for files not in indexed_files
                    archives = session.query(Archive).all()
                    for archive in archives:
                        if archive.filepath and archive.filepath not in self._tracked:
                            self._tracked[archive.filepath] = TrackedFile(
                                task_id=archive.task_id,
                                filepath=archive.filepath,
                                sha256=archive.sha256 or "",
                            )
                            
                logger.info(f"Loaded {len(self._tracked)} existing files for tracking")
            except Exception as exc:
                logger.error(f"Failed to load existing files for tracking: {exc}")

        if self.auto_add_taskmaster:
            from cli.common import detect_taskmaster_directories  # lazy import

            taskmaster_files = 0
            for directory in detect_taskmaster_directories():
                for md_file in Path(directory).rglob("*.md"):
                    file_path = str(md_file)
                    if file_path not in self._tracked:  # Don't override existing
                        task_id = extract_task_id_from_path(file_path) or str(md_file.stem)
                        self._tracked[file_path] = TrackedFile(
                            task_id=task_id,
                            filepath=file_path,
                            sha256="",  # Unknown until computed
                        )
                        taskmaster_files += 1
                        
            if taskmaster_files > 0:
                logger.info(f"Auto-added {taskmaster_files} TaskMaster files for tracking")

    def _initial_crawl(self) -> None:
        """Detect changes that occurred while the watcher was offline."""

        logger.info(f"Running initial crawl across {len(self._tracked)} files")
        processed = 0
        changed = 0
        deleted = 0
        
        for path in list(self._tracked.keys()):
            if Path(path).exists():
                if self._maybe_upsert(path, initial_scan=True):
                    changed += 1
                processed += 1
            else:
                self.process_deletion(path)
                deleted += 1
        
        logger.info(f"Initial crawl complete: {processed} processed, {changed} changed, {deleted} deleted")
        
        # Show completion message
        print(f"👀 File watcher is running. Press Ctrl+C to stop.")
        print(f"📊 Initial scan statistics:")
        print(f"   - Files processed: {processed}")
        print(f"   - Files changed: {changed}")  
        print(f"   - Files deleted: {deleted}")
        
        # Show tracking statistics
        stats = self.get_tracking_stats()
        print(f"📈 Current tracking status:")
        for key, value in stats.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")

    def process_change(self, path: str) -> None:  # noqa: D401
        """Process file change event with extension and path filtering."""
        import time
        
        # Skip files in excluded directories
        path_obj = Path(path)
        path_parts = path_obj.parts
        
        # Exclude git, build, cache, and other irrelevant directories
        excluded_dirs = {
            '.git', '.gitignore', 'node_modules', '__pycache__', '.pytest_cache',
            'build', 'dist', '.venv', 'venv', '.env', '.DS_Store',
            'coverage', '.coverage', '.nyc_output', 'temp', 'tmp',
            '.cache', '.tox', '.mypy_cache', '.ruff_cache'
        }
        
        # Check if any part of the path contains excluded directories
        if any(part in excluded_dirs or part.startswith('.') and part not in {'.taskmaster', '.serena'} 
               for part in path_parts):
            logger.debug(f"Skipping excluded path: {path}")
            return
        
        # Skip temporary files and editor backup files
        filename = path_obj.name
        if (filename.startswith('.') or 
            filename.endswith(('.tmp', '.temp', '.swp', '.swo', '.bak', '.orig', '~')) or
            filename.startswith('#') and filename.endswith('#')):  # Emacs temp files
            logger.debug(f"Skipping temporary file: {path}")
            return
        
        # Only process supported file extensions
        file_ext = path_obj.suffix.lower()
        supported_extensions = {".md", ".txt", ".json", ".py", ".ts", ".tsx", ".js", ".jsx", ".yaml", ".yml"}
        
        if file_ext not in supported_extensions:
            logger.debug(f"Skipping unsupported file extension: {path}")
            return
        
        # Debouncing logic - check if we recently processed this file
        current_time = time.time()
        last_processed = self._pending_changes.get(path, 0)
        
        if current_time - last_processed < self._debounce_delay:
            logger.debug(f"Debouncing file change: {path} (too recent)")
            return
        
        # Update the timestamp for this file
        self._pending_changes[path] = current_time
        
        # Clean up old entries from pending changes (older than 5 seconds)
        cutoff_time = current_time - 5.0
        self._pending_changes = {
            p: t for p, t in self._pending_changes.items() 
            if t > cutoff_time
        }
        
        # Log the file being processed with INFO level to make it visible
        relative_path = str(path_obj.relative_to(Path.cwd())) if path_obj.is_absolute() else path
        logger.info(f"📝 Processing file change: {relative_path}")
            
        self._maybe_upsert(path, initial_scan=False)

    def process_deletion(self, path: str) -> None:  # noqa: D401
        """Handle file deletion events."""
        tracked_file = self._tracked.pop(path, None)
        if not tracked_file:
            return  # Not tracked – ignore

        task_id = tracked_file.task_id
        deleted = False

        if hasattr(self.memory, "delete"):
            try:
                deleted = bool(self.memory.delete(task_id))  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Backend delete failed for {task_id}: {exc}")
        else:
            # Local DB path – use ORM for deletion
            db_path = getattr(self.memory, "db_path", None)
            if db_path and Path(db_path).exists():
                try:
                    with get_session(db_path) as session:
                        archive = (
                            session.query(Archive).filter_by(task_id=task_id).first()
                        )
                        if archive:
                            session.delete(archive)
                            session.commit()
                            deleted = True
                except Exception as exc:
                    logger.error(f"Local delete failed for {task_id}: {exc}")

        if deleted:
            # Also remove from indexed_files table
            db_path = getattr(self.memory, "db_path", None)
            if db_path and Path(db_path).exists():
                try:
                    with get_session(db_path) as session:
                        indexed_file = session.query(IndexedFiles).filter_by(filepath=path).first()
                        if indexed_file:
                            session.delete(indexed_file)
                            session.commit()
                            logger.debug(f"Removed {path} from indexed_files table")
                except Exception as exc:
                    logger.error(f"Failed to remove {path} from indexed_files table: {exc}")
            
            logger.info(f"Deleted task {task_id} due to file removal")
            self.callback("deleted", task_id, path)
        else:
            logger.warning(f"Failed to delete task {task_id} for removed file {path}")

    def add_directory_for_file(self, file_path: str) -> None:
        """Add a file to tracking and ensure its directory is watched.
        
        This method is called by the MemoryIndexer to add newly discovered files.
        """
        file_path = str(Path(file_path).resolve())  # Normalize path
        
        if file_path in self._tracked:
            logger.debug(f"File {file_path} already being tracked")
            return
        
        # Extract task ID
        task_id = extract_task_id_from_path(file_path) or Path(file_path).stem
        
        # Create tracked file
        tracked_file = TrackedFile(
            task_id=task_id,
            filepath=file_path,
            sha256=""  # Will be computed on first change
        )
        
        self._tracked[file_path] = tracked_file
        logger.info(f"Added {file_path} to watcher tracking as task {task_id}")
        
        # If watcher is running, add the directory to observer
        if self._running and self._observer:
            directory = str(Path(file_path).parent)
            
            # Check if directory is already being watched
            if directory not in self.watched_paths:
                try:
                    handler = _SerenaEventHandler(self)
                    self._observer.schedule(handler, path=directory, recursive=True)
                    logger.info(f"Added directory {directory} to file watcher")
                except Exception as exc:
                    logger.error(f"Failed to add directory {directory} to watcher: {exc}")

    def add_files_to_tracking(self, file_paths: List[str]) -> int:
        """Add multiple files to tracking in batch.
        
        Args:
            file_paths: List of file paths to add
            
        Returns:
            int: Number of files successfully added
        """
        added_count = 0
        
        for file_path in file_paths:
            try:
                self.add_directory_for_file(file_path)
                added_count += 1
            except Exception as exc:
                logger.warning(f"Failed to add {file_path} to tracking: {exc}")
        
        logger.info(f"Added {added_count}/{len(file_paths)} files to tracking")
        return added_count

    def get_tracking_stats(self) -> Dict[str, int]:
        """Get statistics about tracked files."""
        total = len(self._tracked)
        with_hash = sum(1 for f in self._tracked.values() if f.sha256)
        
        return {
            "total_tracked": total,
            "with_hash": with_hash,
            "without_hash": total - with_hash,
            "directories_watched": len(self.watched_paths)
        }

    def _maybe_upsert(self, path: str, *, initial_scan: bool) -> bool:
        """Upsert file *path* if its hash changed since last index.
        
        Returns:
            bool: True if file was actually upserted, False if skipped
        """

        if not Path(path).exists():
            return False

        try:
            content = Path(path).read_text(encoding="utf-8")
            file_stat = Path(path).stat()
            file_size = file_stat.st_size
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Unable to read {path}: {exc}")
            return False

        sha256 = compute_content_hash(content)

        tracked_file = self._tracked.get(path)
        if tracked_file and tracked_file.sha256 == sha256:
            return False  # No change

        task_id = (
            tracked_file.task_id
            if tracked_file
            else extract_task_id_from_path(path) or Path(path).stem
        )

        kind = determine_task_kind(path)

        try:
            # Convert TaskKind enum to string value for JSON serialization
            kind_value = kind.value if hasattr(kind, 'value') else str(kind)
            self.memory.upsert(task_id, content, filepath=str(path), kind=kind_value)
            
            # Update indexed_files table for local database
            db_path = getattr(self.memory, "db_path", None)
            if db_path and Path(db_path).exists():
                self._update_indexed_files_table(
                    path, sha256, kind_value, task_id, file_size, last_modified
                )
            
            # Update or create tracked file
            if tracked_file:
                tracked_file.update_tracking(sha256)
            else:
                self._tracked[path] = TrackedFile(
                    task_id=task_id,
                    filepath=path,
                    sha256=sha256,
                    last_modified=last_modified,
                    file_size=file_size
                )

            action = "indexed" if initial_scan else "modified"
            self.callback(action, task_id, path)
            
            # Enhanced logging for real-time monitoring
            if initial_scan:
                logger.debug(f"Indexed {task_id} from {path}")
            else:
                logger.info(f"🔄 Real-time update: {task_id} modified at {path}")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Upsert failed for {path}: {exc}")
            return False

    def _update_indexed_files_table(
        self, filepath: str, sha256: str, kind: str, task_id: str, 
        file_size: int, last_modified: datetime
    ) -> None:
        """Update or insert record in indexed_files table."""
        db_path = getattr(self.memory, "db_path", None)
        if not db_path or not Path(db_path).exists():
            return
            
        try:
            with get_session(db_path) as session:
                # Try to find existing record
                existing = session.query(IndexedFiles).filter_by(filepath=filepath).first()
                
                if existing:
                    # Update existing record
                    existing.sha256 = sha256
                    existing.kind = kind
                    existing.task_id = task_id
                    existing.file_size = file_size
                    existing.last_modified = last_modified
                    existing.updated_at = datetime.now()
                else:
                    # Create new record
                    indexed_file = IndexedFiles(
                        filepath=filepath,
                        sha256=sha256,
                        kind=kind,
                        task_id=task_id,
                        file_size=file_size,
                        last_modified=last_modified,
                        indexed_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    session.add(indexed_file)
                
                session.commit()
                logger.debug(f"Updated indexed_files table for {filepath}")
                
        except Exception as exc:
            logger.error(f"Failed to update indexed_files table for {filepath}: {exc}")


def create_memory_watcher(
    *, memory, auto_add_taskmaster: bool = True, callback=None
):  # noqa: ANN001
    """Create a :class:`_WatchdogMemoryWatcher` instance."""

    return _WatchdogMemoryWatcher(
        memory=memory, auto_add_taskmaster=auto_add_taskmaster, callback=callback
    )


__all__ = ["create_memory_watcher", "TrackedFile"]
