"""Database connection pool monitoring and management commands."""

from typing import Any

from sara.database.session import get_db_manager, get_pool_metrics


def cmd_pool_status(args) -> None:
    """Show database connection pool status."""
    try:
        metrics = get_pool_metrics()

        print("📊 Database Connection Pool Status")
        print("=" * 40)

        if not metrics:
            print("❌ Unable to retrieve pool metrics")
            return

        # Display pool metrics
        pool_size = metrics.get("pool_size", 0)
        checked_in = metrics.get("pool_checked_in", 0)
        checked_out = metrics.get("pool_checked_out", 0)
        overflow = metrics.get("pool_overflow", 0)
        invalid = metrics.get("pool_invalid", 0)

        print(f"Pool Size:       {pool_size}")
        print(f"Available:       {checked_in}")
        print(f"In Use:          {checked_out}")
        print(f"Overflow:        {overflow}")
        print(f"Invalid:         {invalid}")

        # Calculate utilization
        if pool_size > 0:
            utilization = (checked_out / pool_size) * 100
            print(f"Utilization:     {utilization:.1f}%")

            # Status indicators
            if utilization > 90:
                print("🔴 Status:        Critical - Very high usage")
            elif utilization > 70:
                print("🟡 Status:        Warning - High usage")
            else:
                print("🟢 Status:        Normal")
        else:
            print("⚪ Status:        Pool not initialized")

    except Exception as exc:
        print("❌ Failed to get pool status: %s", exc)


def cmd_pool_health(args) -> None:
    """Perform database pool health check."""
    try:
        db_manager = get_db_manager()

        print("🏥 Database Pool Health Check")
        print("=" * 35)

        # Basic connectivity test
        is_healthy = db_manager.health_check()

        if is_healthy:
            print("✅ Database connectivity: OK")
        else:
            print("❌ Database connectivity: FAILED")
            return

        # Pool metrics
        metrics = get_pool_metrics()
        if metrics:
            checked_out = metrics.get("pool_checked_out", 0)
            pool_size = metrics.get("pool_size", 0)
            invalid = metrics.get("pool_invalid", 0)

            if invalid > 0:
                print(f"⚠️  Invalid connections: {invalid}")
            else:
                print("✅ Connection validity: OK")

            if pool_size > 0 and checked_out < pool_size:
                print("✅ Pool availability: OK")
            else:
                print("⚠️  Pool availability: Limited")

        print(
            "\n📋 Overall Health: HEALTHY"
            if is_healthy
            else "\n❌ Overall Health: UNHEALTHY"
        )

    except Exception as exc:
        print(f"❌ Health check failed: {exc}")


def register(sub: Any) -> None:
    """Register pool management commands."""
    pool_parser = sub.add_parser("pool", help="Database connection pool management")
    pool_sub = pool_parser.add_subparsers(dest="pool_command", help="Pool commands")

    # Pool status command
    status_parser = pool_sub.add_parser("status", help="Show connection pool status")
    status_parser.set_defaults(func=cmd_pool_status)

    # Pool health command
    health_parser = pool_sub.add_parser("health", help="Check connection pool health")
    health_parser.set_defaults(func=cmd_pool_health)

    # Default to status if no subcommand
    pool_parser.set_defaults(func=cmd_pool_status)
