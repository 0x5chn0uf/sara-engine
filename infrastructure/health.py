"""Enhanced health check endpoints for production deployment."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
from sqlalchemy import text

logger = logging.getLogger(__name__)


class HealthCheckManager:
    """Manages comprehensive health checks for production deployment."""

    def __init__(self):
        self.startup_time = time.time()
        self.last_db_check = 0
        self.last_embedding_check = 0
        self.cached_db_health = None
        self.cached_embedding_health = None

    def get_liveness_check(self) -> Dict[str, Any]:
        """Simple liveness check - is the service responding?"""
        return {
            "alive": True,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - self.startup_time, 2),
        }

    def get_readiness_check(self) -> Dict[str, Any]:
        """Readiness check - is the service ready to handle requests?"""
        checks = {"ready": True, "timestamp": datetime.now().isoformat(), "checks": {}}

        # Database connectivity check
        try:
            from database.session import get_db_session

            with get_db_session() as session:
                session.execute(text("SELECT 1")).fetchone()
            checks["checks"]["database"] = "ok"
        except Exception as e:
            checks["ready"] = False
            checks["checks"]["database"] = f"failed: {str(e)}"

        # Write queue health check
        try:
            from infrastructure.write_queue import write_queue

            if write_queue is None:
                checks["ready"] = False
                checks["checks"]["write_queue"] = "not initialized"
                return checks

            queue_health = write_queue.health_check()
            if queue_health["status"] in ["healthy", "degraded"]:
                checks["checks"]["write_queue"] = "ok"
            else:
                checks["ready"] = False
                checks["checks"]["write_queue"] = f"unhealthy: {queue_health['status']}"
        except Exception as e:
            checks["ready"] = False
            checks["checks"]["write_queue"] = f"failed: {str(e)}"

        return checks

    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Comprehensive health check with detailed system information."""
        start_time = time.time()
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0",
            "uptime_seconds": round(time.time() - self.startup_time, 2),
            "checks": {},
            "metrics": {},
            "warnings": [],
        }

        # Database health check with caching
        health_data["checks"]["database"] = self._check_database_health()

        # System metrics
        health_data["metrics"]["system"] = self._get_system_metrics()

        # Process metrics
        health_data["metrics"]["process"] = self._get_process_metrics()

        # Application metrics
        health_data["checks"]["write_queue"] = self._check_write_queue_health()
        health_data["checks"]["embeddings"] = self._check_embedding_health()

        # Database file metrics
        health_data["metrics"]["database"] = self._get_database_metrics()

        # Determine overall health status and warnings
        self._evaluate_health_status(health_data)

        # Add response time
        health_data["response_time_ms"] = round((time.time() - start_time) * 1000, 2)

        return health_data

    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health with caching to avoid overload."""
        current_time = time.time()

        # Cache database health check for 30 seconds
        if self.cached_db_health and (current_time - self.last_db_check) < 30:
            return self.cached_db_health

        try:
            from database.session import get_db_session

            db_start = time.time()
            with get_db_session() as session:
                # Test basic connectivity
                session.execute(text("SELECT 1")).fetchone()

                # Get table counts
                archive_count = session.execute(
                    text("SELECT COUNT(*) FROM archives")
                ).scalar()
                embedding_count = session.execute(
                    text("SELECT COUNT(*) FROM embeddings")
                ).scalar()

                # Check database settings
                wal_mode = session.execute(text("PRAGMA journal_mode")).scalar()

                db_response_time = (time.time() - db_start) * 1000

                self.cached_db_health = {
                    "status": "healthy",
                    "response_time_ms": round(db_response_time, 2),
                    "archive_count": archive_count,
                    "embedding_count": embedding_count,
                    "wal_mode": wal_mode,
                    "last_checked": datetime.now().isoformat(),
                }

        except Exception as e:
            self.cached_db_health = {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat(),
            }

        self.last_db_check = current_time
        return self.cached_db_health

    def _check_embedding_health(self) -> Dict[str, Any]:
        """Check embedding service health with caching."""
        current_time = time.time()

        # Cache embedding health check for 60 seconds
        if (
            self.cached_embedding_health
            and (current_time - self.last_embedding_check) < 60
        ):
            return self.cached_embedding_health

        try:
            from infrastructure.embeddings import get_default_generator

            embedding_start = time.time()
            generator = get_default_generator()

            if generator.model is not None:
                # Test embedding generation with small input
                test_embedding = generator.generate_embeddings_batch(
                    ["health check test"]
                )
                embedding_response_time = (time.time() - embedding_start) * 1000

                self.cached_embedding_health = {
                    "status": "healthy",
                    "model": generator.model_name,
                    "dimension": generator.embedding_dim,
                    "response_time_ms": round(embedding_response_time, 2),
                    "test_vector_length": len(test_embedding[0])
                    if test_embedding
                    else 0,
                    "last_checked": datetime.now().isoformat(),
                }
            else:
                self.cached_embedding_health = {
                    "status": "degraded",
                    "message": "Model not loaded, using fallback search",
                    "last_checked": datetime.now().isoformat(),
                }

        except Exception as e:
            self.cached_embedding_health = {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat(),
            }

        self.last_embedding_check = current_time
        return self.cached_embedding_health

    def _check_write_queue_health(self) -> Dict[str, Any]:
        """Check write queue health."""
        try:
            from infrastructure.write_queue import write_queue

            if write_queue is None:
                return {"status": "unhealthy", "error": "Write queue not initialized"}

            return write_queue.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            memory_info = psutil.virtual_memory()

            from settings import settings

            db_path = Path(settings.memory_db)
            disk_info = psutil.disk_usage(str(db_path.parent))

            return {
                "memory": {
                    "total_gb": round(memory_info.total / (1024**3), 2),
                    "available_gb": round(memory_info.available / (1024**3), 2),
                    "percent_used": memory_info.percent,
                },
                "disk": {
                    "total_gb": round(disk_info.total / (1024**3), 2),
                    "free_gb": round(disk_info.free / (1024**3), 2),
                    "percent_used": round((disk_info.used / disk_info.total) * 100, 1),
                },
            }
        except Exception as e:
            return {"error": f"Failed to collect system metrics: {e}"}

    def _get_process_metrics(self) -> Dict[str, Any]:
        """Get process-level metrics."""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "memory_mb": round(process.memory_info().rss / (1024**2), 2),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "create_time": process.create_time(),
            }
        except Exception as e:
            return {"error": f"Failed to collect process metrics: {e}"}

    def _get_database_metrics(self) -> Dict[str, Any]:
        """Get database file metrics."""
        try:
            from settings import settings

            db_path = Path(settings.memory_db)

            if db_path.exists():
                db_size_mb = db_path.stat().st_size / (1024 * 1024)
                return {
                    "size_mb": round(db_size_mb, 2),
                    "path": str(db_path),
                    "exists": True,
                    "writable": os.access(db_path, os.W_OK),
                }
            else:
                return {"exists": False, "path": str(db_path)}
        except Exception as e:
            return {"error": f"Failed to collect database metrics: {e}"}

    def _evaluate_health_status(self, health_data: Dict[str, Any]):
        """Evaluate overall health status and add warnings."""
        warnings = []
        unhealthy_checks = []

        # Check individual component health
        for check_name, check_data in health_data["checks"].items():
            if isinstance(check_data, dict):
                status = check_data.get("status", "unknown")
                if status == "unhealthy":
                    unhealthy_checks.append(check_name)
                elif status == "degraded":
                    warnings.append(f"{check_name} is degraded")

        # Check system metrics for warnings
        system_metrics = health_data["metrics"].get("system", {})
        if "memory" in system_metrics:
            memory_percent = system_metrics["memory"].get("percent_used", 0)
            if memory_percent > 90:
                warnings.append(f"High memory usage: {memory_percent}%")

        if "disk" in system_metrics:
            disk_percent = system_metrics["disk"].get("percent_used", 0)
            if disk_percent > 90:
                warnings.append(f"High disk usage: {disk_percent}%")

        # Check database size
        db_metrics = health_data["metrics"].get("database", {})
        if "size_mb" in db_metrics:
            db_size_mb = db_metrics["size_mb"]
            if db_size_mb > 1000:
                warnings.append(f"Large database size: {db_size_mb:.1f}MB")

        # Set overall status
        if unhealthy_checks:
            health_data["status"] = "unhealthy"
            warnings.append(f"Unhealthy components: {', '.join(unhealthy_checks)}")
        elif warnings:
            health_data["status"] = "degraded"

        health_data["warnings"] = warnings

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment-specific information."""
        import platform
        import sys

        from settings import settings

        return {
            "environment": settings.environment,
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "server_url": settings.server_url,
            "database_path": settings.memory_db,
            "log_level": settings.get_effective_log_level(),
            "async_write_enabled": settings.async_write,
            "startup_time": self.startup_time,
        }


# Global health check manager
_health_manager: Optional[HealthCheckManager] = None


def get_health_manager() -> HealthCheckManager:
    """Get global health check manager instance."""
    global _health_manager
    if _health_manager is None:
        _health_manager = HealthCheckManager()
    return _health_manager
