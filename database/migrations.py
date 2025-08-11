"""Database migration utilities using Alembic."""

from typing import Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from sara.settings import database_config, settings


class MigrationManager:
    """Manages database migrations using Alembic."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize migration manager.

        Args:
            db_path: Path to SQLite database file
        """
        # Use centralized database configuration
        if db_path is not None:
            self.db_config = settings.get_database_config(db_path)
        else:
            self.db_config = database_config

        # Validate configuration early
        try:
            self.db_config.validate_configuration()
        except Exception as e:
            print("Migration configuration validation failed: %s", e)
            raise

        self.db_path = self.db_config.db_path
        self.db_url = self.db_config.db_url
        self.alembic_cfg_path = self.db_config.alembic_config_path

        if not self.alembic_cfg_path.exists():
            raise FileNotFoundError(
                f"Alembic config not found: {self.alembic_cfg_path}"
            )

        self.alembic_cfg = Config(str(self.alembic_cfg_path))
        # Override the database URL
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.db_url)
        # Set the correct script location using centralized config
        self.alembic_cfg.set_main_option(
            "script_location", str(self.db_config.migrations_path)
        )

    def get_current_revision(self) -> Optional[str]:
        """Get the current database revision.

        Returns:
            Optional[str]: Current revision ID or None if not stamped
        """
        try:
            engine = create_engine(self.db_url)
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as exc:
            print("Failed to get current revision: %s", exc)
            return None

    def get_head_revision(self) -> Optional[str]:
        """Get the head revision from migration scripts.

        Returns:
            Optional[str]: Head revision ID
        """
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            return script.get_current_head()
        except Exception as exc:
            print("Failed to get head revision: %s", exc)
            return None

    def is_up_to_date(self) -> bool:
        """Check if database is up to date with migrations.

        Returns:
            bool: True if database is current
        """
        current = self.get_current_revision()
        head = self.get_head_revision()
        return current == head

    def needs_migration(self) -> bool:
        """Check if database needs migration.

        Returns:
            bool: True if migration is needed
        """
        return not self.is_up_to_date()

    def create_migration(self, message: str, autogenerate: bool = True) -> str:
        """Create a new migration.

        Args:
            message: Migration description
            autogenerate: Whether to auto-generate migration from model changes

        Returns:
            str: Generated revision ID
        """
        try:
            if autogenerate:
                result = command.revision(
                    self.alembic_cfg, message=message, autogenerate=True
                )
            else:
                result = command.revision(self.alembic_cfg, message=message)

            print("Created migration: %s", message)
            return result.revision
        except Exception as exc:
            print("❌ Failed to create migration: %s", exc)
            raise

    def upgrade(self, revision: str = "head") -> None:
        """Upgrade database to specified revision.

        Args:
            revision: Target revision (default: "head")
        """
        try:
            # Ensure database directory exists using centralized method
            self.db_config.ensure_database_directory()

            command.upgrade(self.alembic_cfg, revision)
            print("Upgraded database to %s", revision)
        except Exception as exc:
            print("❌ Failed to upgrade database: %s", exc)
            raise

    def downgrade(self, revision: str) -> None:
        """Downgrade database to specified revision.

        Args:
            revision: Target revision
        """
        try:
            command.downgrade(self.alembic_cfg, revision)
            print("Downgraded database to %s", revision)
        except Exception as exc:
            print("❌ Failed to downgrade database: %s", exc)
            raise

    def stamp(self, revision: str = "head") -> None:
        """Stamp database with specified revision without running migrations.

        Args:
            revision: Revision to stamp (default: "head")
        """
        try:
            # Ensure database directory exists using centralized method
            self.db_config.ensure_database_directory()

            command.stamp(self.alembic_cfg, revision)
            print(f"Stamped database with {revision}")
        except Exception as exc:
            print("❌ Failed to stamp database: %s", exc)
            raise

    def show_history(self) -> None:
        """Show migration history."""
        try:
            command.history(self.alembic_cfg)
        except Exception as exc:
            print("❌ Failed to show history: %s", exc)
            raise

    def show_current(self) -> None:
        """Show current revision."""
        try:
            command.current(self.alembic_cfg)
        except Exception as exc:
            print("❌ Failed to show current revision: %s", exc)
            raise

    def init_database(self) -> None:
        """Initialize database with current schema and stamp with head revision."""
        try:
            # Create tables using SQLAlchemy
            from sara.core.models import Base

            engine = create_engine(self.db_url)
            Base.metadata.create_all(engine)

            # Stamp with head revision
            self.stamp("head")

            print("Initialized database at %s", self.db_path)
        except Exception as exc:
            print("❌ Failed to initialize database: %s", exc)
            raise


def get_migration_manager(db_path: Optional[str] = None) -> MigrationManager:
    """Get migration manager instance.

    Args:
        db_path: Path to database file

    Returns:
        MigrationManager: Migration manager
    """
    return MigrationManager(db_path)


def auto_migrate(db_path: Optional[str] = None) -> bool:
    """Automatically migrate database if needed.

    Args:
        db_path: Path to database file

    Returns:
        bool: True if migration was performed
    """
    migration_manager = get_migration_manager(db_path)

    current = migration_manager.get_current_revision()
    if current is None:
        # Database not initialized or not stamped
        print("Database not stamped, initializing...")
        migration_manager.init_database()
        return True
    elif migration_manager.needs_migration():
        # Migration needed
        print("Database needs migration, upgrading...")
        migration_manager.upgrade()
        return True
    else:
        # Up to date
        print("Database is up to date")
        return False
