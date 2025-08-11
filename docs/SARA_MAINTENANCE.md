# Serena Maintenance Configuration Guide

> **Overview**: Configuration reference for Serena's automated database maintenance system  
> **Audience**: Developers, DevOps engineers, system administrators  
> **Last Updated**: July 2025

## What is Serena?

Serena is SmartWalletFX's semantic memory bridge system that provides intelligent search over project artifacts using SQLite with vector embeddings. It maintains a searchable knowledge base of code, documentation, and project context that enables AI-powered development workflows.

## Maintenance System Overview

Serena's maintenance system ensures optimal database performance through automated operations:

- **Health Monitoring**: Continuous database health checks and metrics collection
- **WAL Checkpointing**: Periodic Write-Ahead Log optimization for performance
- **Database Vacuuming**: Space reclamation and index optimization
- **Backup Management**: Optional pre-maintenance backups
- **Performance Tuning**: Adaptive interval adjustment based on usage patterns

## Configuration Structure

Maintenance behavior is controlled through the `MaintenanceConfig` class in `serena/settings.py`, which can be customized via environment variables or configuration files.

### Core Configuration Sections

#### 1. Operation Intervals (`intervals`)

Controls the frequency of maintenance operations:

```python
class _Intervals(BaseModel):
    health_check: str = "1d"    # Database health monitoring
    checkpoint: str = "7d"      # WAL checkpoint operations  
    vacuum: str = "30d"         # Database vacuum/optimization
```

**Supported Formats**:
- Duration strings: `"1d"`, `"24h"`, `"60m"`, `"3600s"`
- Object format: `{"days": 7, "hours": 2, "minutes": 30}`

**Recommended Values**:
- **Development**: `health_check: "1h"`, `checkpoint: "1d"`, `vacuum: "7d"`
- **Production**: `health_check: "1d"`, `checkpoint: "7d"`, `vacuum: "30d"`
- **High-Volume**: `health_check: "12h"`, `checkpoint: "3d"`, `vacuum: "14d"`

#### 2. Operation Toggles (`enabled`)

Enable/disable specific maintenance operations:

```python
class _Enabled(BaseModel):
    health_check: bool = True   # Monitor database health
    checkpoint: bool = True     # Perform WAL checkpoints
    vacuum: bool = True         # Run database vacuum
```

**Use Cases**:
- **CI/CD**: Disable `checkpoint` and `vacuum`, keep `health_check`
- **Debugging**: Disable all operations temporarily
- **Custom Schedules**: Disable auto-maintenance for manual control

#### 3. Performance Thresholds (`thresholds`)

Database size and performance warning levels:

```python
class _Thresholds(BaseModel):
    large_db_size_mb: int = 1000        # "Large" database threshold
    large_entry_count: int = 100000     # Entry count for optimization
    warning_db_size_mb: int = 2000      # Size warning threshold
    critical_db_size_mb: int = 5000     # Critical size threshold
```

**Threshold Effects**:
- `large_db_size_mb`: Triggers more frequent vacuum operations
- `large_entry_count`: Enables additional optimization strategies
- `warning_db_size_mb`: Generates warning logs and metrics
- `critical_db_size_mb`: Triggers alerts and emergency maintenance

#### 4. Notification Settings (`notifications`)

Control maintenance logging and output:

```python
class _Notifications(BaseModel):
    enable_console_output: bool = True
    enable_file_logging: bool = True
    log_file: str = "serena/database/maintenance.log"
```

**Logging Behavior**:
- **Console Output**: Real-time maintenance status in terminal
- **File Logging**: Persistent logs for debugging and monitoring
- **Log Rotation**: Automatic log file management (if enabled)

#### 5. Performance Limits (`performance`)

Operation timeouts and optimization settings:

```python
class _Performance(BaseModel):
    max_checkpoint_duration_seconds: int = 300     # 5 minutes
    max_vacuum_duration_seconds: int = 1800        # 30 minutes
    auto_optimize_intervals: bool = True           # Adaptive intervals
```

**Performance Tuning**:
- **Timeouts**: Prevent long-running operations from blocking
- **Auto-optimization**: Adjusts intervals based on database growth patterns
- **Resource Limits**: Prevents maintenance from impacting application performance

#### 6. Backup Configuration (`backup`)

Optional database backup before maintenance:

```python
class _Backup(BaseModel):
    enable_pre_vacuum_backup: bool = False
    backup_directory: str = ".serena/backups"
    max_backup_files: int = 5
```

**Backup Strategy**:
- **Pre-vacuum**: Creates backup before potentially destructive operations
- **Retention**: Automatically removes old backups beyond `max_backup_files`
- **Storage**: Backups stored in configured directory with timestamps

## Environment-Specific Configurations

### Development Environment

```json
{
  "maintenance": {
    "intervals": {
      "health_check": "1h",
      "checkpoint": "1d",
      "vacuum": "7d"
    },
    "backup": {
      "enable_pre_vacuum_backup": true,
      "max_backup_files": 10
    },
    "notifications": {
      "enable_console_output": true,
      "enable_file_logging": true
    }
  }
}
```

**Rationale**: Frequent monitoring for quick issue detection, backups for safety, verbose logging for debugging.

### Production Environment

```json
{
  "maintenance": {
    "intervals": {
      "health_check": "1d",
      "checkpoint": "7d", 
      "vacuum": "30d"
    },
    "thresholds": {
      "large_db_size_mb": 2000,
      "warning_db_size_mb": 4000,
      "critical_db_size_mb": 8000
    },
    "performance": {
      "max_checkpoint_duration_seconds": 600,
      "max_vacuum_duration_seconds": 3600,
      "auto_optimize_intervals": true
    },
    "notifications": {
      "enable_console_output": false,
      "enable_file_logging": true
    }
  }
}
```

**Rationale**: Conservative intervals to minimize overhead, higher thresholds for larger datasets, disabled console output for cleaner logs.

### CI/CD Environment

```json
{
  "maintenance": {
    "enabled": {
      "health_check": true,
      "checkpoint": false,
      "vacuum": false
    },
    "intervals": {
      "health_check": "5m"
    },
    "notifications": {
      "enable_console_output": false,
      "enable_file_logging": false
    }
  }
}
```

**Rationale**: Only health checks for CI validation, no persistent operations, minimal logging to reduce CI noise.

### High-Volume Projects

```json
{
  "maintenance": {
    "intervals": {
      "health_check": "12h",
      "checkpoint": "3d",
      "vacuum": "14d"
    },
    "thresholds": {
      "large_db_size_mb": 3000,
      "large_entry_count": 500000,
      "warning_db_size_mb": 6000,
      "critical_db_size_mb": 12000
    },
    "backup": {
      "enable_pre_vacuum_backup": true,
      "max_backup_files": 3
    }
  }
}
```

**Rationale**: More frequent maintenance for heavy usage, higher thresholds for large datasets, backups for data safety.

## Monitoring and Operations

### Health Monitoring

Check maintenance status through multiple interfaces:

```bash
# CLI health check with maintenance status
python -m serena.cli health --verbose

# Direct maintenance status
python -m serena.cli maintenance

# View maintenance logs
tail -f serena/database/maintenance.log

# Check maintenance metrics via API
curl http://localhost:8000/maintenance/status
```

### Manual Maintenance Operations

```bash
# Trigger specific operations via API
curl -X POST http://localhost:8000/maintenance/run/checkpoint
curl -X POST http://localhost:8000/maintenance/run/vacuum
curl -X POST http://localhost:8000/maintenance/run/health_check

# Legacy script (deprecated, use API)
python -m serena.scripts.maintenance --status
```

### Key Metrics to Monitor

1. **Database Size**: Monitor growth trends and threshold breaches
2. **Operation Duration**: Track maintenance operation performance
3. **Error Rates**: Watch for failed maintenance operations
4. **Health Check Results**: Monitor database integrity scores
5. **Resource Usage**: CPU/memory impact during maintenance

### Log Analysis

Maintenance logs include structured information:

```
[2025-07-29 10:00:00] INFO: 🚀 Starting maintenance operation: health_check
[2025-07-29 10:00:01] INFO: ✅ Health check completed - Status: healthy
[2025-07-29 10:00:01] INFO: 📊 Database metrics: size=150MB, entries=25000, integrity=100%
```

**Key Log Patterns**:
- `🚀 Starting maintenance`: Operation initiation
- `✅ completed`: Successful operation completion
- `❌ failed`: Operation failure (investigate immediately)
- `⚠️ warning`: Non-critical issues requiring attention
- `📊 metrics`: Performance and health statistics

## Troubleshooting

### Common Issues

#### 1. Maintenance Operations Taking Too Long

**Symptoms**: Operations exceed configured timeouts
**Causes**: Large database size, insufficient resources, lock contention
**Solutions**:
- Increase timeout values in `performance` section
- Schedule maintenance during low-activity periods
- Consider more frequent vacuum operations to prevent buildup

#### 2. Database Size Growing Rapidly

**Symptoms**: Frequent size warnings, poor query performance
**Causes**: High indexing rate, infrequent vacuum operations, WAL file growth
**Solutions**:
- Reduce vacuum interval (e.g., from `30d` to `14d`)
- Enable more frequent checkpoints
- Monitor and optimize indexing patterns

#### 3. Maintenance Service Not Starting

**Symptoms**: No maintenance logs, operations not running
**Causes**: Configuration errors, database connectivity issues, permission problems
**Solutions**:
- Check configuration syntax and values
- Verify database file permissions
- Review server startup logs for errors

#### 4. High Resource Usage During Maintenance

**Symptoms**: System slowdown during maintenance operations
**Causes**: Aggressive vacuum settings, concurrent operations, insufficient resources
**Solutions**:
- Increase operation timeouts to reduce frequency
- Schedule maintenance during off-peak hours
- Consider resource-limited maintenance windows

### Debugging Steps

1. **Check Configuration**:
   ```bash
   python -c "from serena.settings import settings; print(settings.maintenance.model_dump_json(indent=2))"
   ```

2. **Verify Service Status**:
   ```bash
   curl http://localhost:8000/health | jq '.checks.maintenance'
   ```

3. **Analyze Recent Logs**:
   ```bash
   grep -E "(ERROR|WARNING|failed)" serena/database/maintenance.log | tail -20
   ```

4. **Manual Operation Test**:
   ```bash
   curl -X POST http://localhost:8000/maintenance/run/health_check
   ```

## Migration and Upgrades

### Upgrading Maintenance Configuration

When updating Serena versions, follow these steps:

1. **Backup Current Configuration**:
   ```bash
   cp serena/settings.py serena/settings.py.backup
   ```

2. **Review Configuration Changes**:
   - Check for new configuration options
   - Verify deprecated settings
   - Update threshold values if needed

3. **Test New Configuration**:
   ```bash
   python -m serena.cli maintenance --dry-run
   ```

4. **Monitor After Upgrade**:
   - Watch logs for configuration errors
   - Verify maintenance operations continue
   - Adjust settings based on performance

### Configuration Validation

The system validates configuration on startup:

```python
# Validation errors are logged and prevent startup
[ERROR] Invalid maintenance configuration: intervals.health_check must be a valid duration
[ERROR] Invalid maintenance configuration: thresholds.large_db_size_mb must be positive
```

## Best Practices

### General Guidelines

1. **Start Conservative**: Use default settings initially, then optimize based on usage patterns
2. **Monitor Actively**: Set up alerts for maintenance failures and performance degradation
3. **Test Changes**: Validate configuration changes in development before production
4. **Document Customizations**: Record reasons for non-default settings
5. **Regular Review**: Periodically assess and adjust maintenance schedules

### Performance Optimization

1. **Adaptive Intervals**: Enable `auto_optimize_intervals` for dynamic adjustment
2. **Resource Awareness**: Schedule intensive operations during low-activity periods
3. **Incremental Maintenance**: Prefer frequent, small operations over infrequent, large ones
4. **Monitoring Integration**: Use maintenance metrics in your monitoring stack

### Security Considerations

1. **Log Sensitivity**: Ensure maintenance logs don't contain sensitive information
2. **Backup Security**: Secure backup directory with appropriate permissions
3. **API Access**: Restrict maintenance API endpoints in production
4. **Resource Limits**: Configure timeouts to prevent resource exhaustion attacks

## Related Documentation

- [Database Patterns](./DATABASE_PATTERNS.md) - Database design and optimization patterns
- [Performance Guidelines](./PERFORMANCE_GUIDELINES.md) - General performance optimization
- [Architecture](./ARCHITECTURE.md) - Overall system architecture including Serena
- [Development Workflows](./DEVELOPMENT_WORKFLOWS.md) - Development and testing practices

---

For questions or issues with maintenance configuration, consult the troubleshooting section above or refer to the Serena CLI help: `python -m serena.cli --help`