from __future__ import annotations

"""Common helpers for Sara CLI."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sara.core.errors import ErrorCode, SaraException, ServerUnavailableError

# consolidated settings
from sara.settings import settings


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def validate_configuration(show_details: bool = False) -> bool:
    """Validate Sara configuration with user-friendly error handling.

    Args:
        show_details: Whether to show detailed configuration status

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    from sara.core.errors import get_user_friendly_message
    from sara.settings import database_config, settings

    try:
        # Validate settings
        settings.validate_early()

        # Validate database configuration
        database_config.validate_configuration()

        if show_details:
            status = database_config.get_user_friendly_status()
            print(f"✓ Configuration is valid")
            print(f"  Database: {status['db_path']}")
            print(f"  Database exists: {status['db_exists']}")
            print(f"  Alembic config: {status['alembic_config']}")
            print(f"  Migrations dir: {status['migrations_dir']}")

        return True

    except Exception as e:
        # Get user-friendly error message
        friendly_msg = get_user_friendly_message(e)
        print(f"✗ Configuration validation failed: {friendly_msg}")

        if show_details:
            print(f"\nConfiguration status:")
            status = database_config.get_user_friendly_status()
            if status["status"] == "invalid":
                print(f"  Error: {status['error']}")
                print(f"  Database path: {status['db_path']}")

        return False


def remote_upsert(task_id: str, markdown_text: str, **meta) -> bool:
    """Remote upsert placeholder - not implemented yet."""
    logging.warning("Remote upsert not implemented in new CLI")
    return False


def detect_taskmaster_directories() -> List[str]:
    """Detect TaskMaster directories in the current project using centralized configuration."""
    directories = []

    # Use centralized directory patterns from settings
    directory_patterns = settings.index_directories_list

    # Add some additional patterns that are commonly used but may not be in base config
    additional_patterns = [
        ".taskmaster/memory-bank/reflections",
        ".taskmaster/memory-bank/archives",
    ]

    # Combine patterns
    all_patterns = directory_patterns + additional_patterns

    for pattern in all_patterns:
        path = Path(pattern)
        if path.exists() and path.is_dir():
            directories.append(str(path))

    return directories


class RemoteMemory:
    """Remote memory client that uses server API for operations."""

    def __init__(self, server_url: str = None):
        self.server_url = server_url or settings.server_url
        if not self.server_url.startswith("http"):
            self.server_url = f"http://{self.server_url}"
        self._session = None

        # Add db_path property for IndexedFiles table access
        # This allows local tracking even when using remote API
        from sara.settings import database_config

        self.db_path = database_config.db_path

    @property
    def session(self):
        """Get or create HTTP session for connection pooling."""
        if self._session is None:
            import requests

            self._session = requests.Session()
            # Set reasonable connection limits
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10, pool_maxsize=20, max_retries=0
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to server with structured error handling."""
        import logging

        import requests

        url = f"{self.server_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, timeout=30, **kwargs)

            # Handle HTTP errors
            if not response.ok:
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        # Structured error response
                        error_info = error_data["error"]
                        raise SaraException(
                            code=ErrorCode(error_info.get("code", "INTERNAL_ERROR")),
                            message=error_info.get("message", "Unknown error"),
                            details=error_info.get("details", {}),
                        )
                    else:
                        # Legacy error response
                        raise RuntimeError(
                            f"Server error {response.status_code}: {error_data}"
                        )
                except ValueError:
                    # Non-JSON error response
                    raise RuntimeError(
                        f"Server error {response.status_code}: {response.text}"
                    )

            # Parse successful response
            response_data = response.json()

            # Handle structured success responses
            if "success" in response_data:
                if response_data["success"]:
                    return response_data.get("data", response_data)
                else:
                    # Structured error in success response format
                    error_info = response_data.get("error", {})
                    raise SaraException(
                        code=ErrorCode(error_info.get("code", "INTERNAL_ERROR")),
                        message=error_info.get("message", "Unknown error"),
                        details=error_info.get("details", {}),
                    )

            # Legacy response format - return as-is
            return response_data

        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection to {url} failed: {e}")
            raise ServerUnavailableError(self.server_url, cause=e)
        except requests.exceptions.Timeout as e:
            logging.error(f"Request to {url} timed out: {e}")
            raise SaraException(
                code=ErrorCode.SERVER_TIMEOUT,
                message=f"Request to {url} timed out",
                details={"url": url, "timeout": 30},
            )
        except SaraException:
            # Re-raise SaraExceptions as-is
            raise
        except Exception as e:
            logging.error(f"Request to {url} failed: {e}")
            raise RuntimeError(f"Server request failed: {e}")

    def close(self):
        """Close HTTP session and cleanup connections."""
        if self._session:
            self._session.close()
            self._session = None

    def wait_for_server_completion(self, timeout: float = 30.0) -> bool:
        """Wait for server to complete any pending operations."""
        import logging
        import time

        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                # Check server queue status
                response = self._make_request("GET", "/queue/status")
                queue_size = response.get("current_queue_size", 0)

                if queue_size == 0:
                    logging.debug("Server queue is empty")
                    return True

                logging.debug(f"Server queue has {queue_size} pending operations")
                time.sleep(0.5)  # Wait a bit before checking again
            except SaraException as e:
                logging.error(f"Queue status check failed: {e.message}")
                return False
            except Exception as e:
                logging.error(f"Queue status check failed: {e}")
                return False
            except Exception as e:
                logging.error(f"Queue status check failed: {e}")
                return False

        logging.warning(f"Timeout waiting for server completion after {timeout}s")
        return False

    def search(self, query: str, limit: int = 10):
        """Search archives using server API."""
        try:
            response = self._make_request(
                "GET", "/search", params={"q": query, "limit": limit}
            )
            return response.get("results", [])
        except SaraException as e:
            logging.error(f"Remote search failed: {e.message}")
            # For CLI usage, return empty list but log structured error
            if e.details:
                logging.debug(f"Search error details: {e.details}")
            return []
        except Exception as e:
            logging.error(f"Remote search failed: {e}")
            return []
        except Exception as e:
            logging.error(f"Remote search failed: {e}")
            return []

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get archive by task ID using server API."""
        try:
            response = self._make_request("GET", f"/archives/{task_id}")
            return response
        except SaraException as e:
            if e.code == ErrorCode.RESOURCE_NOT_FOUND:
                # Expected when checking if archive exists
                logging.debug(f"Archive not found: {task_id}")
                return None
            else:
                logging.error(f"Remote get failed for {task_id}: {e.message}")
                if e.details:
                    logging.debug(f"Get error details: {e.details}")
                return None
        except Exception as e:
            logging.error(f"Remote get failed for {task_id}: {e}")
            return None
        except Exception as e:
            logging.error(f"Remote get failed for {task_id}: {e}")
            return None

    def upsert(
        self,
        task_id: str,
        markdown_text: str,
        filepath: str = None,
        title: str = None,
        kind: str = "archive",
        status: str = None,
        completed_at: str = None,
    ) -> bool:
        """Upsert archive using server API."""
        try:
            data = {
                "task_id": task_id,
                "markdown_text": markdown_text,
            }

            if filepath:
                data["filepath"] = filepath
            if title:
                data["title"] = title
            if kind:
                data["kind"] = kind
            if status:
                data["status"] = status
            if completed_at:
                data["completed_at"] = completed_at

            response = self._make_request("POST", "/archives", json=data)
            # For structured responses, success is indicated by no exception
            return True
        except SaraException as e:
            logging.error(f"Remote upsert failed for {task_id}: {e.message}")
            if e.details:
                logging.debug(f"Upsert error details: {e.details}")
            return False
        except Exception as e:
            logging.error(f"Remote upsert failed for {task_id}: {e}")
            return False
        except Exception as e:
            logging.error(f"Remote upsert failed for {task_id}: {e}")
            return False

    def is_server_available(self) -> bool:
        """Check if server is available."""
        try:
            response = self._make_request("GET", "/health")
            # Check if health response indicates healthy status
            if isinstance(response, dict):
                status = response.get("status", "unknown")
                return status == "healthy"
            return True
        except Exception:
            return False

    def get_server_error_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed error information from the last failed request."""
        try:
            response = self._make_request("GET", "/health")
            return None  # No error if health check succeeds
        except SaraException as e:
            return {"code": e.code.value, "message": e.message, "details": e.details}
        except Exception as e:
            return {"code": "CONNECTION_ERROR", "message": str(e), "details": {}}
        except Exception as e:
            return {
                "code": "CONNECTION_ERROR",
                "message": str(e),
                "operation": "health_check",
                "details": {},
                "correlation_id": None,
            }

    def health(self) -> Dict[str, Any]:
        """Get comprehensive health information from the server."""
        try:
            response = self._make_request("GET", "/health")
            return response.get("data", {})
        except Exception as e:
            # Return a basic health structure if server is unavailable
            return {
                "status": "unavailable",
                "error": str(e),
                "database": {"archive_count": 0, "database_size": 0},
                "server": {"available": False},
            }


__all__ = [
    "setup_logging",
    "remote_upsert",
    "detect_taskmaster_directories",
    "RemoteMemory",
]
