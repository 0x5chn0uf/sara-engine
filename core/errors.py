"""Simplified error handling for local project use."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(Enum):
    """Standardized error codes for Sara operations."""

    # Server/Connection Errors
    SERVER_UNAVAILABLE = "SERVER_UNAVAILABLE"
    SERVER_CONNECTION_FAILED = "SERVER_CONNECTION_FAILED"
    SERVER_TIMEOUT = "SERVER_TIMEOUT"

    # Resource Errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"

    # Validation Errors
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_TASK_ID = "INVALID_TASK_ID"
    INVALID_CONTENT_FORMAT = "INVALID_CONTENT_FORMAT"

    # Database Errors
    DATABASE_CONNECTION_FAILED = "DATABASE_CONNECTION_FAILED"
    DATABASE_QUERY_FAILED = "DATABASE_QUERY_FAILED"
    DATABASE_TRANSACTION_FAILED = "DATABASE_TRANSACTION_FAILED"

    # Embedding/Search Errors
    EMBEDDING_SERVICE_UNAVAILABLE = "EMBEDDING_SERVICE_UNAVAILABLE"
    EMBEDDING_SERVICE_ERROR = "EMBEDDING_SERVICE_ERROR"
    EMBEDDING_GENERATION_FAILED = "EMBEDDING_GENERATION_FAILED"
    SEARCH_QUERY_FAILED = "SEARCH_QUERY_FAILED"

    # Content Processing Errors
    CONTENT_PROCESSING_FAILED = "CONTENT_PROCESSING_FAILED"
    FILE_READ_FAILED = "FILE_READ_FAILED"
    CONTENT_TOO_LARGE = "CONTENT_TOO_LARGE"

    # Indexing Errors
    INDEXING_FAILED = "INDEXING_FAILED"

    # Internal Errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"


@dataclass
class SaraError:
    """Simple error response for Sara operations."""

    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for API responses."""
        return {
            "success": False,
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details or {},
            },
        }


class SaraException(Exception):
    """Base exception class for Sara operations."""

    def __init__(
        self, code: ErrorCode, message: str, details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        error = SaraError(self.code, self.message, self.details)
        return error.to_dict()


class ServerUnavailableError(SaraException):
    """Raised when server is not available for remote operations."""

    def __init__(self, server_url: str, cause: Optional[Exception] = None):
        details = {"server_url": server_url}
        if cause:
            details["cause"] = str(cause)

        super().__init__(
            code=ErrorCode.SERVER_UNAVAILABLE,
            message=f"Server not available at {server_url}",
            details=details,
        )


class ResourceNotFoundError(SaraException):
    """Raised when requested resource is not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} not found: {resource_id}",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ValidationError(SaraException):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            code=ErrorCode.INVALID_INPUT,
            message=f"Invalid {field}: {reason}",
            details={"field": field, "value": str(value), "reason": reason},
        )


class EmbeddingServiceError(SaraException):
    """Raised when embedding service operations fail."""

    def __init__(self, operation: str, cause: Optional[Exception] = None):
        details = {"operation": operation}
        if cause:
            details["cause"] = str(cause)

        super().__init__(
            code=ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
            message=f"Embedding service failed during {operation}",
            details=details,
        )


class DatabaseError(SaraException):
    """Raised when database operations fail."""

    def __init__(self, operation: str, cause: Optional[Exception] = None):
        details = {"operation": operation}
        if cause:
            details["cause"] = str(cause)

        super().__init__(
            code=ErrorCode.DATABASE_QUERY_FAILED,
            message=f"Database operation failed: {operation}",
            details=details,
        )


class ContentProcessingError(SaraException):
    """Raised when content processing fails."""

    def __init__(self, file_path: str, reason: str, cause: Optional[Exception] = None):
        details = {"file_path": file_path, "reason": reason}
        if cause:
            details["cause"] = str(cause)

        super().__init__(
            code=ErrorCode.CONTENT_PROCESSING_FAILED,
            message=f"Content processing failed for {file_path}: {reason}",
            details=details,
        )


def create_error_response(
    code: ErrorCode, message: str, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error response."""
    error = SaraError(code=code, message=message, details=details)
    return error.to_dict()


def create_success_response(
    data: Any = None, message: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized success response."""
    response = {"success": True, "data": data}
    if message:
        response["message"] = message
    return response


# User-friendly error messages for CLI
ERROR_SOLUTIONS = {
    ErrorCode.SERVER_UNAVAILABLE: "💡 Start the server with: sara serve",
    ErrorCode.SERVER_CONNECTION_FAILED: "💡 Check if the server is running and accessible",
    ErrorCode.SERVER_TIMEOUT: "💡 Server may be overloaded, try again in a moment",
    ErrorCode.RESOURCE_NOT_FOUND: "💡 Check the ID and try again, or use --list to see available items",
    ErrorCode.INVALID_TASK_ID: "💡 Task IDs should be alphanumeric and under 255 characters",
    ErrorCode.CONTENT_TOO_LARGE: "💡 Break large files into smaller chunks before indexing",
    ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE: "💡 Check your Python environment and model dependencies",
    ErrorCode.DATABASE_CONNECTION_FAILED: "💡 Try: sara init to reset the database",
    ErrorCode.SEARCH_QUERY_FAILED: "💡 Try a different search term or check server logs",
    ErrorCode.INDEXING_FAILED: "💡 Check file permissions and content format",
    ErrorCode.CONFIGURATION_ERROR: "💡 Review your .sara/config.json settings",
}


def get_user_friendly_message(error_code: ErrorCode, original_message: str) -> str:
    """Get user-friendly error message with solution."""
    solution = ERROR_SOLUTIONS.get(error_code, "💡 Check logs for more details")
    return f"❌ {original_message}\n   {solution}"
