from __future__ import annotations

"""Centralised configuration using Pydantic Settings.

This replaces the previous bespoke ``serena.config`` layering with a single
``SerenaSettings`` object that automatically merges the following sources
(in order of precedence):

1. *Initialization kwargs* – values passed explicitly when instantiating
   ``SerenaSettings`` (used by CLI commands for --flag overrides).
2. *Environment variables* – variables prefixed with ``SERENA_``.
3. *serena/config.json* – user-editable file co-located with the package.
4. *Hard-coded defaults* defined on the dataclass fields.

An already-instantiated ``settings`` singleton is exposed for application-wide
use so existing code can simply do::

    from sara.settings import settings
    db_path = settings.memory_db

The legacy helper functions in ``serena.config`` have been re-implemented to
proxy to this object so external imports stay backward-compatible.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Pydantic Settings implementation
# ---------------------------------------------------------------------------


class _Intervals(BaseModel):
    health_check: str = "1d"
    checkpoint: str = "7d"
    vacuum: str = "30d"


class _Enabled(BaseModel):
    health_check: bool = True
    checkpoint: bool = True
    vacuum: bool = True


class _Thresholds(BaseModel):
    large_db_size_mb: int = 1000
    large_entry_count: int = 100000
    warning_db_size_mb: int = 500
    critical_db_size_mb: int = 2000


class _Notifications(BaseModel):
    enable_console_output: bool = True
    enable_file_logging: bool = True
    log_file: str = str(Path(__file__).parent / "database" / "maintenance.log")


class _Performance(BaseModel):
    max_checkpoint_duration_seconds: int = 300
    max_vacuum_duration_seconds: int = 1800
    auto_optimize_intervals: bool = True


class _Backup(BaseModel):
    enable_pre_vacuum_backup: bool = False
    backup_directory: str = "database/backups"
    max_backup_files: int = 5


class MaintenanceConfig(BaseModel):
    """All tunables for automated database maintenance tasks."""

    intervals: _Intervals = _Intervals()
    enabled: _Enabled = _Enabled()
    thresholds: _Thresholds = _Thresholds()
    notifications: _Notifications = _Notifications()
    performance: _Performance = _Performance()
    backup: _Backup = _Backup()

    class Config:
        validate_assignment = True


class DatabaseConfig(BaseModel):
    """Centralized database configuration with path resolution and validation."""

    def __init__(self, db_path: Optional[str] = None, **data):
        super().__init__(**data)
        self._db_path = db_path
        self._serena_root: Optional[Path] = None
        self._validated = False

    @property
    def serena_root(self) -> Path:
        """Get the Serena package root directory."""
        if self._serena_root is None:
            self._serena_root = Path(__file__).parent
        return self._serena_root

    @property
    def db_path(self) -> str:
        """Get the resolved database path."""
        if self._db_path is not None:
            return self._db_path

        # Import settings here to avoid circular import
        from sara.settings import settings

        return settings.memory_db

    @property
    def db_url(self) -> str:
        """Get the SQLAlchemy database URL."""
        return f"sqlite:///{self.db_path}"

    @property
    def alembic_config_path(self) -> Path:
        """Get the path to alembic.ini."""
        return self.serena_root / "alembic.ini"

    @property
    def migrations_path(self) -> Path:
        """Get the path to migrations directory."""
        return self.serena_root / "migrations"

    def validate_configuration(self) -> None:
        """Validate all configuration paths and settings.

        Raises:
            SaraException: If configuration is invalid
        """
        if self._validated:
            return

        errors = []

        # Validate database path
        try:
            db_path = Path(self.db_path)

            # Check if parent directory exists or can be created
            db_dir = db_path.parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    errors.append(
                        f"Cannot create database directory: {db_dir} (permission denied)"
                    )
                except OSError as e:
                    errors.append(f"Cannot create database directory: {db_dir} ({e})")

            # Check write permissions
            if db_path.exists() and not os.access(db_path, os.W_OK):
                errors.append(f"Database file is not writable: {db_path}")
            elif not os.access(db_dir, os.W_OK):
                errors.append(f"Database directory is not writable: {db_dir}")

        except Exception as e:
            errors.append(f"Invalid database path '{self.db_path}': {e}")

        # Validate alembic configuration
        if not self.alembic_config_path.exists():
            errors.append(
                f"Alembic configuration not found: {self.alembic_config_path}"
            )

        # Validate migrations directory
        if not self.migrations_path.exists():
            errors.append(f"Migrations directory not found: {self.migrations_path}")

        if errors:
            from sara.core.errors import ValidationError

            raise ValidationError(
                message="Database configuration validation failed",
                details={"validation_errors": errors},
            )

        self._validated = True

    def ensure_database_directory(self) -> None:
        """Ensure the database directory exists."""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_user_friendly_status(self) -> dict:
        """Get user-friendly configuration status."""
        try:
            self.validate_configuration()
            return {
                "status": "valid",
                "db_path": self.db_path,
                "db_exists": Path(self.db_path).exists(),
                "alembic_config": str(self.alembic_config_path),
                "migrations_dir": str(self.migrations_path),
            }
        except Exception as e:
            return {
                "status": "invalid",
                "error": str(e),
                "db_path": self.db_path if hasattr(self, "_db_path") else "unknown",
            }


class SerenaSettings(BaseSettings):
    """All runtime configuration for Serena in one place."""

    # Environment configuration
    environment: str = Field(
        default="development",
        env="SERENA_ENVIRONMENT",
        description="Environment: development, testing, production",
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        env="SERENA_LOG_LEVEL",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="SERENA_LOG_FORMAT",
        description="Log message format string",
    )
    log_file: str | None = Field(
        default=None,
        env="SERENA_LOG_FILE",
        description="Optional file path for log output",
    )

    # Paths
    memory_db: str = Field(
        default=str(Path(__file__).parent / "database" / "memory_index.db"),
        env="SERENA_MEMORY_DB",
        description="Absolute path to the SQLite database used by Serena.",
    )

    # HTTP server
    server_host: str = Field(
        default="127.0.0.1",
        env="SERENA_SERVER_HOST",
        description="Host address to bind the server to",
    )
    server_port: int = Field(
        default=8765,
        env="SERENA_SERVER_PORT",
        description="Port to bind the server to",
    )
    server_url: str = Field(
        default="http://localhost:8765",
        env="SERENA_SERVER_URL",
        description="Base URL where a Serena HTTP server is running.",
    )

    # Server behavior
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        env="SERENA_MAX_REQUEST_SIZE",
        description="Maximum request size in bytes",
    )
    request_timeout: int = Field(
        default=30,
        env="SERENA_REQUEST_TIMEOUT",
        description="Request timeout in seconds",
    )
    worker_timeout: int = Field(
        default=120,
        env="SERENA_WORKER_TIMEOUT",
        description="Worker timeout in seconds",
    )

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        env="SERENA_CORS_ORIGINS",
        description="Comma-separated list of Allowed CORS origins.",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        env="SERENA_CORS_ALLOW_CREDENTIALS",
        description="Whether to allow credentials in CORS responses.",
    )
    cors_allow_methods: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS",
        env="SERENA_CORS_ALLOW_METHODS",
        description="Comma-separated list of HTTP methods for CORS.",
    )
    cors_allow_headers: str = Field(
        default="*",
        env="SERENA_CORS_ALLOW_HEADERS",
        description="Allowed request headers for CORS.",
    )

    # Embeddings / hardware
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="SARA_EMBEDDING_MODEL",
        description="Name or path of the sentence-transformers model to use.",
    )
    device: str | None = Field(
        default=None,
        env="SERENA_DEVICE",
        description="Computation device override (cpu | cuda | mps).",
    )

    # Performance configuration
    async_write: bool = Field(
        default=False,
        env="SERENA_ASYNC_WRITE",
        description="Write to DB via background queue instead of sync call.",
    )
    write_queue_size: int = Field(
        default=1000,
        env="SERENA_WRITE_QUEUE_SIZE",
        description="Maximum size of the write queue",
    )
    write_batch_size: int = Field(
        default=10,
        env="SERENA_WRITE_BATCH_SIZE",
        description="Batch size for write operations",
    )
    write_batch_timeout_ms: int = Field(
        default=500,
        env="SERENA_WRITE_BATCH_TIMEOUT_MS",
        description="Timeout for batching write operations in milliseconds",
    )

    # Resource limits
    max_embedding_batch_size: int = Field(
        default=100,
        env="SERENA_MAX_EMBEDDING_BATCH_SIZE",
        description="Maximum batch size for embedding generation",
    )
    max_content_size_mb: int = Field(
        default=10,
        env="SERENA_MAX_CONTENT_SIZE_MB",
        description="Maximum content size in MB for indexing",
    )

    # Code embedding configuration
    embedding_chunk_size: int = Field(
        default=4096,  # 4KB chunks
        env="SARA_EMBEDDING_CHUNK_SIZE",
        description="Maximum size of code chunks in bytes for embedding",
    )
    embedding_overlap_lines: int = Field(
        default=20,
        env="SARA_EMBEDDING_OVERLAP_LINES",
        description="Number of lines to overlap between code chunks",
    )
    embedding_strip_comments: bool = Field(
        default=True,
        env="SARA_EMBEDDING_STRIP_COMMENTS",
        description="Whether to strip comments and docstrings from code before embedding",
    )
    embedding_include_globs: str = Field(
        default="*.py,*.ts,*.tsx,*.js,*.jsx,backend/app/**/*.py,frontend/src/**/*.ts,frontend/src/**/*.tsx",
        env="SARA_EMBEDDING_INCLUDE_GLOBS",
        description="Comma-separated glob patterns for files to include in code embedding",
    )
    embedding_exclude_globs: str = Field(
        default="**/test*/**,**/tests/**,**/*test*,**/migrations/**,**/node_modules/**,**/.git/**,**/build/**,**/dist/**,**/__pycache__/**,**/*.pyc,**/*.min.js,**/*.map,**/coverage/**,**/.pytest_cache/**",
        env="SARA_EMBEDDING_EXCLUDE_GLOBS",
        description="Comma-separated glob patterns for files to exclude from code embedding",
    )

    # Content indexing configuration (for sara index command)
    index_directories: str = Field(
        default=".taskmaster/memory-bank,.taskmaster/logs,.serena/memories,docs",
        env="SERENA_INDEX_DIRECTORIES",
        description="Comma-separated directory patterns for content indexing (sara index)",
    )
    index_include_globs: str = Field(
        default="**/*.md,**/*.txt,**/*.json,**/*.yaml,**/*.yml",
        env="SERENA_INDEX_INCLUDE_GLOBS",
        description="Comma-separated glob patterns for files to include in content indexing",
    )
    index_exclude_globs: str = Field(
        default="**/node_modules/**,**/.git/**,**/build/**,**/dist/**,**/__pycache__/**,**/*.pyc,**/coverage/**,**/.pytest_cache/**,CLAUDE.local.md",
        env="SERENA_INDEX_EXCLUDE_GLOBS",
        description="Comma-separated glob patterns for files to exclude from content indexing",
    )

    # Maintenance section (parsed from JSON defaults, override via env prefixes
    maintenance: MaintenanceConfig = Field(
        default_factory=MaintenanceConfig,
        description="Nested configuration for automated DB maintenance",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def cors_methods_list(self) -> list[str]:
        return [m.strip() for m in self.cors_allow_methods.split(",") if m.strip()]

    @property
    def embedding_include_globs_list(self) -> list[str]:
        """Get list of include glob patterns for code embedding."""
        return [g.strip() for g in self.embedding_include_globs.split(",") if g.strip()]

    @property
    def embedding_exclude_globs_list(self) -> list[str]:
        """Get list of exclude glob patterns for code embedding."""
        return [g.strip() for g in self.embedding_exclude_globs.split(",") if g.strip()]

    @property
    def index_directories_list(self) -> list[str]:
        """Get list of directories for content indexing."""
        return [d.strip() for d in self.index_directories.split(",") if d.strip()]

    @property
    def index_include_globs_list(self) -> list[str]:
        """Get list of include glob patterns for content indexing."""
        return [g.strip() for g in self.index_include_globs.split(",") if g.strip()]

    @property
    def index_exclude_globs_list(self) -> list[str]:
        """Get list of exclude glob patterns for content indexing."""
        return [g.strip() for g in self.index_exclude_globs.split(",") if g.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    def get_effective_log_level(self) -> str:
        """Get effective log level based on environment."""
        if self.is_production:
            # In production, be more conservative with logging
            level_map = {
                "DEBUG": "INFO",
                "INFO": "INFO",
                "WARNING": "WARNING",
                "ERROR": "ERROR",
                "CRITICAL": "CRITICAL",
            }
            return level_map.get(self.log_level.upper(), "INFO")
        return self.log_level.upper()

    def validate_early(self) -> None:
        """Perform early validation of critical settings.

        This should be called immediately after settings initialization
        to catch configuration issues before they cause runtime failures.

        Raises:
            ValidationError: If critical configuration is invalid
        """
        errors = []

        # Validate database path
        try:
            db_path = Path(self.memory_db)
            db_dir = db_path.parent

            # Check if parent directory can be created
            if not db_dir.exists():
                try:
                    # Test directory creation (but don't actually create it yet)
                    test_path = db_dir
                    while not test_path.exists() and test_path != test_path.parent:
                        test_path = test_path.parent
                    if not os.access(test_path, os.W_OK):
                        errors.append(
                            f"Cannot create database directory: {db_dir} (permission denied)"
                        )
                except Exception as e:
                    errors.append(f"Invalid database path '{self.memory_db}': {e}")

        except Exception as e:
            errors.append(f"Invalid database path configuration: {e}")

        # Validate server URL format
        if not self.server_url.startswith(("http://", "https://")):
            errors.append(f"Invalid server URL format: {self.server_url}")

        if errors:
            from sara.core.errors import ValidationError

            raise ValidationError(
                message="Settings validation failed during startup",
                details={"validation_errors": errors},
            )

    def get_database_config(self, db_path: Optional[str] = None) -> "DatabaseConfig":
        """Get a database configuration instance.

        Args:
            db_path: Optional database path override

        Returns:
            DatabaseConfig: Configured database configuration
        """
        return DatabaseConfig(db_path=db_path)

    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        import logging
        import sys

        # Set log level
        level = getattr(logging, self.get_effective_log_level(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create custom formatter without date and replace logger names
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                # Replace specific logger names
                if record.name == "watchfiles.main":
                    record.name = "WATCHER"
                elif record.name.startswith("sara.infrastructure.watcher"):
                    record.name = "WATCHER"

                # Format the message properly
                if record.args:
                    message = record.msg % record.args
                else:
                    message = record.msg

                # Special format for WATCHER - no level indicator
                if record.name == "WATCHER":
                    return f"WATCHER - {message}"
                else:
                    # Use format without timestamp for other loggers
                    return f"{record.name} - {record.levelname} - {message}"

        formatter = CustomFormatter()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Configure ALL watchfiles loggers to prevent timestamps
        for logger_name in ["watchfiles", "watchfiles.main", "watchfiles.watcher"]:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.addHandler(console_handler)
            logger.setLevel(level)
            logger.propagate = False  # Prevent propagation to root logger

        # File handler (if specified)
        if self.log_file:
            try:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                # Also add to watchfiles loggers
                for logger_name in [
                    "watchfiles",
                    "watchfiles.main",
                    "watchfiles.watcher",
                ]:
                    logging.getLogger(logger_name).addHandler(file_handler)
            except Exception as e:
                root_logger.warning(
                    f"Warning: Could not set up file logging to {self.log_file}: {e}"
                )

        # Suppress noisy third-party loggers in production
        if self.is_production:
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            logging.getLogger("transformers").setLevel(logging.WARNING)

    def get_deployment_info(self) -> dict:
        """Get deployment information for health checks and debugging."""
        import platform
        import sys

        return {
            "environment": self.environment,
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "memory_db_path": self.memory_db,
            "server_url": self.server_url,
            "log_level": self.get_effective_log_level(),
            "async_write_enabled": self.async_write,
        }


# Singleton instance used throughout the package
settings = SerenaSettings()

# Global database configuration instance
database_config = settings.get_database_config()


# Perform early validation (with graceful error handling for CLI commands)
def validate_settings_on_import() -> bool:
    """Validate settings on module import with graceful error handling.

    Returns:
        bool: True if validation passed, False if validation failed
    """
    try:
        settings.validate_early()
        # Configure logging on successful validation
        settings.configure_logging()
        return True
    except Exception:
        # Don't fail on import - let individual commands handle validation
        # This allows CLI commands to show user-friendly error messages
        return False


# Validate settings (but don't fail on import)
_settings_valid = validate_settings_on_import()


# Environment-specific configuration helpers
def load_environment_config(env_name: str) -> SerenaSettings:
    """Load configuration for specific environment."""
    return SerenaSettings(environment=env_name)


def get_production_config() -> SerenaSettings:
    """Get production-optimized configuration."""
    return SerenaSettings(
        environment="production",
        log_level="INFO",
        async_write=False,
        write_batch_size=20,  # Larger batches in production
        write_queue_size=2000,  # Larger queue in production
        max_embedding_batch_size=50,  # More efficient batching
    )


def get_development_config() -> SerenaSettings:
    """Get development-optimized configuration."""
    return SerenaSettings(
        environment="development",
        log_level="DEBUG",
        async_write=False,
        write_batch_size=5,  # Smaller batches for faster feedback
        write_queue_size=100,  # Smaller queue for development
    )
