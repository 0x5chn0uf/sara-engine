#!/usr/bin/env python3
"""
Production-ready startup script for Serena server.

This script handles:
- Environment validation
- Database initialization
- Graceful startup with health checks
- Process management signals
- Logging configuration
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sara.database.session import get_db_manager
from sara.infrastructure.server import create_app
from sara.settings import (get_development_config, get_production_config,
                             settings)


class ServerManager:
    """Manages Serena server lifecycle with proper error handling."""

    def __init__(self, host: str = None, port: int = None, environment: str = None):
        self.host = host or settings.server_host
        self.port = port or settings.server_port
        self.environment = environment or settings.environment

        # Use environment-specific configuration
        if self.environment == "production":
            self.settings = get_production_config()
        elif self.environment == "development":
            self.settings = get_development_config()
        else:
            self.settings = settings

        self.app = None
        self.server = None
        self.shutdown_event = asyncio.Event()

        # Configure logging
        self.settings.configure_logging()
        self.logger = logging.getLogger(__name__)

    def validate_environment(self) -> dict:
        """Validate environment and dependencies before startup."""
        validation = {"status": "success", "checks": [], "warnings": [], "errors": []}

        # Check Python version
        if sys.version_info < (3, 8):
            validation["errors"].append(f"Python 3.8+ required, got {sys.version}")
            validation["status"] = "failed"
        else:
            validation["checks"].append(f"Python version: {sys.version}")

        # Check required dependencies
        try:
            import fastapi
            import sentence_transformers
            import sqlalchemy

            validation["checks"].append("All required dependencies available")
        except ImportError as e:
            validation["errors"].append(f"Missing dependency: {e}")
            validation["status"] = "failed"

        # Check database configuration
        try:
            self.settings.validate_early()
            validation["checks"].append("Configuration validation passed")
        except Exception as e:
            validation["errors"].append(f"Configuration invalid: {e}")
            validation["status"] = "failed"

        # Check database initialization
        try:
            db_manager = get_db_manager()
            init_result = db_manager.initialize_for_deployment()

            if init_result["status"] == "success":
                validation["checks"].append("Database initialization successful")
            elif init_result["status"] == "degraded":
                validation["warnings"].extend(init_result["warnings"])
                validation["checks"].append(
                    "Database initialization completed with warnings"
                )
            else:
                validation["errors"].extend(init_result["errors"])
                validation["status"] = "failed"

        except Exception as e:
            validation["errors"].append(f"Database initialization failed: {e}")
            validation["status"] = "failed"

        # Check port availability
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((self.host, self.port))
            validation["checks"].append(f"Port {self.port} available")
        except OSError:
            validation["errors"].append(f"Port {self.port} already in use")
            validation["status"] = "failed"

        # Environment-specific checks
        if self.environment == "production":
            # Check for production readiness
            if self.settings.log_level.upper() == "DEBUG":
                validation["warnings"].append("Debug logging enabled in production")

            # Check memory requirements
            try:
                import psutil

                memory = psutil.virtual_memory()
                if memory.available < 512 * 1024 * 1024:  # 512MB
                    validation["warnings"].append("Less than 512MB memory available")
            except ImportError:
                validation["warnings"].append("Could not check system resources")

        return validation

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            asyncio.create_task(self.shutdown())

        # Handle common termination signals
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, signal_handler)

        # Handle SIGHUP for configuration reload (if supported)
        if hasattr(signal, "SIGHUP"):

            def reload_handler(signum, frame):
                self.logger.info(
                    "Received SIGHUP, configuration reload not implemented"
                )

            signal.signal(signal.SIGHUP, reload_handler)

    async def startup_health_check(
        self, max_retries: int = 30, retry_delay: float = 1.0
    ) -> bool:
        """Wait for server to be healthy after startup."""
        import aiohttp

        url = f"http://{self.host}:{self.port}/ready"

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            self.logger.info("Server health check passed")
                            return True
                        else:
                            self.logger.warning(
                                f"Health check returned status {response.status}"
                            )
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.debug(f"Health check attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(
                        f"Health check failed after {max_retries} attempts: {e}"
                    )

        return False

    async def start_server(self) -> bool:
        """Start the FastAPI server with proper error handling."""
        try:
            import uvicorn

            # Create the FastAPI app
            self.app = create_app()

            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level=self.settings.get_effective_log_level().lower(),
                access_log=not self.settings.is_production,  # Disable access logs in production
                server_header=False,  # Don't expose server info
                date_header=False,  # Don't expose server date
                reload=False,  # Never use reload in production scripts
                workers=1,  # Single worker for SQLite
                timeout_keep_alive=30,
                timeout_graceful_shutdown=30,
            )

            # Start server
            server = uvicorn.Server(config)
            self.server = server

            self.logger.info(
                f"Starting Serena server on {self.host}:{self.port} ({self.environment} environment)"
            )

            # Start server in background
            server_task = asyncio.create_task(server.serve())

            # Wait a moment for server to start
            await asyncio.sleep(2)

            # Perform startup health check
            if await self.startup_health_check():
                self.logger.info("✅ Serena server started successfully")

                # Wait for shutdown signal
                await self.shutdown_event.wait()

                # Graceful shutdown
                self.logger.info("Initiating graceful shutdown...")
                server.should_exit = True
                await server_task

                return True
            else:
                self.logger.error("❌ Server failed health check during startup")
                server.should_exit = True
                await server_task
                return False

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False

    async def shutdown(self):
        """Signal shutdown and cleanup."""
        self.logger.info("Shutdown requested")
        self.shutdown_event.set()

    def run(self) -> int:
        """Run the server with full lifecycle management."""
        start_time = time.time()

        print(f"🚀 Starting Serena Memory Bridge Server")
        print(f"   Environment: {self.environment}")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Database: {self.settings.memory_db}")
        print()

        # Validate environment
        print("📋 Validating environment...")
        validation = self.validate_environment()

        if validation["errors"]:
            print("❌ Environment validation failed:")
            for error in validation["errors"]:
                print(f"   • {error}")
            return 1

        if validation["warnings"]:
            print("⚠️  Environment warnings:")
            for warning in validation["warnings"]:
                print(f"   • {warning}")

        print("✅ Environment validation passed")
        for check in validation["checks"]:
            print(f"   • {check}")
        print()

        # Setup signal handlers
        self.setup_signal_handlers()

        # Start server
        try:
            success = asyncio.run(self.start_server())

            startup_time = time.time() - start_time
            if success:
                print(f"✅ Server shutdown completed gracefully in {startup_time:.2f}s")
                return 0
            else:
                print(f"❌ Server failed to start properly after {startup_time:.2f}s")
                return 1

        except KeyboardInterrupt:
            print("\n⚠️  Received keyboard interrupt")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start Serena Memory Bridge Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_server.py                          # Start with default settings
  python start_server.py --host 0.0.0.0          # Bind to all interfaces
  python start_server.py --port 8080             # Use different port
  python start_server.py --env production        # Production mode
  python start_server.py --validate-only         # Just validate environment
        """,
    )

    parser.add_argument(
        "--host", default=None, help="Host to bind to (default: from settings)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from settings)",
    )

    parser.add_argument(
        "--env",
        "--environment",
        choices=["development", "production", "testing"],
        default=None,
        help="Environment mode (default: from settings)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment, don't start server",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Create server manager
    server_manager = ServerManager(host=args.host, port=args.port, environment=args.env)

    if args.validate_only:
        print("🔍 Validating environment only...")
        validation = server_manager.validate_environment()

        if validation["errors"]:
            print("❌ Validation failed:")
            for error in validation["errors"]:
                print(f"   • {error}")
            return 1

        print("✅ Environment validation passed")
        if not args.quiet:
            for check in validation["checks"]:
                print(f"   • {check}")

        if validation["warnings"]:
            print("⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"   • {warning}")

        return 0

    # Run server
    return server_manager.run()


if __name__ == "__main__":
    sys.exit(main())
