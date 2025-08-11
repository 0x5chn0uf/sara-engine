#!/usr/bin/env python3
"""
Serena daemon management script for production deployments.

This script provides process management functionality:
- Start/stop/restart server
- Status monitoring
- Log management
- Health checks
- Automatic restarts on failure
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


class SerenaDaemon:
    """Manages Serena server as a daemon process."""

    def __init__(self, working_dir: str = None):
        self.working_dir = (
            Path(working_dir) if working_dir else Path(__file__).parent.parent
        )
        self.pid_file = self.working_dir / "sara.pid"
        self.log_file = self.working_dir / "logs" / "sara.log"
        self.status_file = self.working_dir / "serena_status.json"

        # Ensure directories exist
        self.log_file.parent.mkdir(exist_ok=True)

    def is_running(self) -> Optional[int]:
        """Check if Serena server is running and return PID."""
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())

            # Check if process is actually running
            try:
                os.kill(pid, 0)  # Send null signal to check if process exists
                return pid
            except ProcessLookupError:
                # Process doesn't exist, clean up stale PID file
                self.pid_file.unlink()
                return None

        except (ValueError, FileNotFoundError):
            return None

    def start(
        self,
        host: str = None,
        port: int = None,
        environment: str = None,
        background: bool = True,
    ) -> bool:
        """Start the Serena server."""
        current_pid = self.is_running()
        if current_pid:
            print(f"❌ Serena server is already running (PID: {current_pid})")
            return False

        print("🚀 Starting Serena server...")

        # Build command
        start_script = self.working_dir / "scripts" / "start_server.py"
        cmd = [sys.executable, str(start_script)]

        if host:
            cmd.extend(["--host", host])
        if port:
            cmd.extend(["--port", str(port)])
        if environment:
            cmd.extend(["--env", environment])

        try:
            if background:
                # Start as background process
                with open(self.log_file, "a") as log:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=self.working_dir,
                        start_new_session=True,  # Detach from current session
                    )

                # Save PID
                with open(self.pid_file, "w") as f:
                    f.write(str(process.pid))

                # Wait a moment and check if it's still running
                time.sleep(2)
                if self.is_running():
                    print(f"✅ Serena server started successfully (PID: {process.pid})")
                    print(f"📋 Logs: {self.log_file}")

                    # Update status
                    self._update_status(
                        {
                            "status": "running",
                            "pid": process.pid,
                            "started_at": time.time(),
                            "command": " ".join(cmd),
                        }
                    )

                    return True
                else:
                    print("❌ Server failed to start (check logs for details)")
                    return False
            else:
                # Run in foreground
                return subprocess.run(cmd, cwd=self.working_dir).returncode == 0

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def stop(self, timeout: int = 30) -> bool:
        """Stop the Serena server gracefully."""
        pid = self.is_running()
        if not pid:
            print("❌ Serena server is not running")
            return False

        print(f"🛑 Stopping Serena server (PID: {pid})...")

        try:
            # Try graceful shutdown first
            os.kill(pid, signal.SIGTERM)

            # Wait for graceful shutdown
            for i in range(timeout):
                if not self.is_running():
                    print("✅ Server stopped gracefully")
                    self._update_status(
                        {"status": "stopped", "stopped_at": time.time()}
                    )
                    return True
                time.sleep(1)

            # If still running, force kill
            print("⚠️  Graceful shutdown timeout, forcing termination...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)

            if not self.is_running():
                print("✅ Server terminated")
                self._update_status({"status": "killed", "stopped_at": time.time()})
                return True
            else:
                print("❌ Failed to terminate server")
                return False

        except ProcessLookupError:
            print("✅ Server was already stopped")
            self.pid_file.unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f"❌ Error stopping server: {e}")
            return False

    def restart(self, **kwargs) -> bool:
        """Restart the Serena server."""
        print("🔄 Restarting Serena server...")

        if self.is_running():
            if not self.stop():
                return False

        # Wait a moment before restarting
        time.sleep(1)

        return self.start(**kwargs)

    def status(self, detailed: bool = False) -> Dict:
        """Get server status information."""
        pid = self.is_running()

        if pid:
            status_info = {
                "status": "running",
                "pid": pid,
                "log_file": str(self.log_file),
            }

            # Get additional process info if detailed
            if detailed:
                try:
                    import psutil

                    process = psutil.Process(pid)
                    status_info.update(
                        {
                            "cpu_percent": process.cpu_percent(),
                            "memory_mb": round(
                                process.memory_info().rss / (1024**2), 2
                            ),
                            "create_time": process.create_time(),
                            "num_threads": process.num_threads(),
                            "status": process.status(),
                        }
                    )
                except ImportError:
                    status_info["detailed_info"] = "psutil not available"
                except Exception as e:
                    status_info["detailed_error"] = str(e)

            # Try to get server health
            if detailed:
                try:
                    health = self._check_server_health()
                    status_info["health"] = health
                except Exception as e:
                    status_info["health_error"] = str(e)

        else:
            status_info = {"status": "stopped"}

            # Check for crash information
            if self.status_file.exists():
                try:
                    with open(self.status_file, "r") as f:
                        saved_status = json.load(f)
                    status_info.update(saved_status)
                except Exception:
                    pass

        return status_info

    def logs(self, lines: int = 50, follow: bool = False) -> None:
        """Show server logs."""
        if not self.log_file.exists():
            print("❌ No log file found")
            return

        if follow:
            # Follow logs in real-time
            print(f"📋 Following logs from {self.log_file} (Ctrl+C to stop)...")
            try:
                subprocess.run(["tail", "-f", str(self.log_file)])
            except KeyboardInterrupt:
                print("\n✅ Stopped following logs")
        else:
            # Show last N lines
            try:
                result = subprocess.run(
                    ["tail", "-n", str(lines), str(self.log_file)],
                    capture_output=True,
                    text=True,
                )
                print(f"📋 Last {lines} lines from {self.log_file}:")
                print("-" * 50)
                print(result.stdout)
            except Exception as e:
                print(f"❌ Error reading logs: {e}")

    def health_check(self) -> bool:
        """Perform health check on running server."""
        if not self.is_running():
            print("❌ Server is not running")
            return False

        try:
            health = self._check_server_health()

            if health.get("status") == "healthy":
                print("✅ Server is healthy")
                print(f"   Response time: {health.get('response_time_ms', 'N/A')}ms")
                return True
            else:
                print(f"⚠️  Server health: {health.get('status', 'unknown')}")
                if health.get("warnings"):
                    for warning in health["warnings"]:
                        print(f"   Warning: {warning}")
                return False

        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def _check_server_health(self) -> Dict:
        """Check server health via HTTP endpoint."""
        import requests

        from sara.settings import settings

        url = f"{settings.server_url}/health"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get("data", {})
            else:
                return {"status": "unhealthy", "http_status": response.status_code}
        except requests.RequestException as e:
            return {"status": "unreachable", "error": str(e)}

    def _update_status(self, status_data: Dict) -> None:
        """Update status file with current information."""
        try:
            existing_status = {}
            if self.status_file.exists():
                with open(self.status_file, "r") as f:
                    existing_status = json.load(f)

            existing_status.update(status_data)

            with open(self.status_file, "w") as f:
                json.dump(existing_status, f, indent=2)
        except Exception:
            pass  # Don't fail operations due to status file issues


def main():
    """Main entry point for daemon management."""
    parser = argparse.ArgumentParser(
        description="Serena Server Daemon Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serena_daemon.py start                    # Start server in background
  python serena_daemon.py start --foreground       # Start server in foreground
  python serena_daemon.py stop                     # Stop server
  python serena_daemon.py restart                  # Restart server
  python serena_daemon.py status                   # Show status
  python serena_daemon.py status --detailed        # Show detailed status
  python serena_daemon.py logs                     # Show recent logs
  python serena_daemon.py logs --follow            # Follow logs in real-time
  python serena_daemon.py health                   # Check server health
        """,
    )

    parser.add_argument(
        "command",
        choices=["start", "stop", "restart", "status", "logs", "health"],
        help="Command to execute",
    )

    parser.add_argument("--host", help="Host to bind to (for start/restart)")

    parser.add_argument("--port", type=int, help="Port to bind to (for start/restart)")

    parser.add_argument(
        "--env",
        "--environment",
        choices=["development", "production", "testing"],
        help="Environment mode (for start/restart)",
    )

    parser.add_argument("--working-dir", help="Working directory for Serena")

    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground (for start command)",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information (for status command)",
    )

    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow logs in real-time (for logs command)",
    )

    parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of log lines to show (default: 50)",
    )

    args = parser.parse_args()

    # Create daemon manager
    daemon = SerenaDaemon(working_dir=args.working_dir)

    # Execute command
    if args.command == "start":
        success = daemon.start(
            host=args.host,
            port=args.port,
            environment=args.env,
            background=not args.foreground,
        )
        return 0 if success else 1

    elif args.command == "stop":
        success = daemon.stop()
        return 0 if success else 1

    elif args.command == "restart":
        success = daemon.restart(host=args.host, port=args.port, environment=args.env)
        return 0 if success else 1

    elif args.command == "status":
        status = daemon.status(detailed=args.detailed)

        print("Serena Server Status:")
        print("=" * 30)

        if status["status"] == "running":
            print(f"✅ Status: Running (PID: {status['pid']})")

            if args.detailed and "cpu_percent" in status:
                print(f"📊 CPU: {status['cpu_percent']}%")
                print(f"💾 Memory: {status['memory_mb']} MB")
                print(f"🧵 Threads: {status['num_threads']}")

                if "health" in status:
                    health = status["health"]
                    print(f"🏥 Health: {health.get('status', 'unknown')}")
                    if health.get("warnings"):
                        for warning in health["warnings"]:
                            print(f"   ⚠️  {warning}")
        else:
            print(f"❌ Status: {status['status'].title()}")

        if "log_file" in status:
            print(f"📋 Logs: {status['log_file']}")

        return 0

    elif args.command == "logs":
        daemon.logs(lines=args.lines, follow=args.follow)
        return 0

    elif args.command == "health":
        success = daemon.health_check()
        return 0 if success else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
