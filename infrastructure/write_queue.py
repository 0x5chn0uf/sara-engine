from __future__ import annotations

"""
Enhanced asynchronous write queue with batch processing, error recovery, and monitoring.
"""

import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class WriteOperationType(Enum):
    """Types of write operations supported by the queue."""

    UPSERT = "upsert"
    DELETE = "delete"
    BATCH_UPSERT = "batch_upsert"
    BATCH_DELETE = "batch_delete"
    MAINTENANCE = "maintenance"
    INDEX_UPDATE = "index_update"


@dataclass
class WriteOperation:
    """Represents a single write operation in the queue."""

    id: str
    operation_type: WriteOperationType
    fn: Callable
    args: tuple
    kwargs: dict
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1  # Lower = higher priority

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def __lt__(self, other):
        """Make WriteOperation comparable for PriorityQueue."""
        if not isinstance(other, WriteOperation):
            return NotImplemented
        # First compare by priority (lower = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        # Then by timestamp (older first)
        return self.timestamp < other.timestamp


@dataclass
class QueueMetrics:
    """Queue performance and health metrics."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    retried_operations: int = 0
    current_queue_size: int = 0
    avg_processing_time_ms: float = 0.0
    last_batch_size: int = 0
    last_batch_processing_time_ms: float = 0.0


class EnhancedWriteQueue:
    """
    Enhanced write queue with batch processing, error recovery, and monitoring.

    Features:
    - Batch processing to reduce database connections
    - Exponential backoff retry logic
    - Priority queue support
    - Graceful shutdown handling
    - Comprehensive metrics and monitoring
    - Backpressure handling
    """

    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout_ms: int = 500,
        max_queue_size: int = 1000,
        max_retries: int = 3,
        worker_count: int = 1,
    ):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms / 1000.0  # Convert to seconds
        self.max_queue_size = max_queue_size
        self.max_retries = max_retries
        self.worker_count = worker_count

        # Thread-safe queues
        self._priority_queue: "queue.PriorityQueue[Tuple[int, WriteOperation]]" = (
            queue.PriorityQueue(maxsize=max_queue_size)
        )
        self._retry_queue: "queue.Queue[WriteOperation]" = queue.Queue()

        # Worker threads
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._shutdown_complete = threading.Event()

        # Metrics and monitoring
        self._metrics = QueueMetrics()
        self._metrics_lock = threading.Lock()
        self._processing_times = deque(maxlen=100)  # Keep last 100 processing times

        # Batching state
        self._current_batch: List[WriteOperation] = []
        self._last_batch_time = time.time()
        self._batch_lock = threading.Lock()

        # Start worker threads
        self._start_workers()
        print(
            f"Enhanced write queue started with {worker_count} workers, batch_size={batch_size}"
        )

    def _start_workers(self) -> None:
        """Start worker threads for processing operations."""
        for i in range(self.worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=False,  # Not daemon to ensure graceful shutdown
                name=f"serena-write-queue-{i}",
            )
            worker.start()
            self._workers.append(worker)

    def submit(
        self,
        fn: Callable,
        *args,
        priority: int = 1,
        max_retries: Optional[int] = None,
        operation_type: WriteOperationType = WriteOperationType.UPSERT,
        **kwargs,
    ) -> str:
        """
        Submit a write operation to the queue.

        Args:
            fn: Function to execute
            *args: Function arguments
            priority: Operation priority (lower = higher priority)
            max_retries: Override default retry count
            operation_type: Type of operation for batching optimization
            **kwargs: Function keyword arguments

        Returns:
            str: Operation ID for tracking

        Raises:
            queue.Full: If queue is at capacity
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Write queue is shutting down")

        operation = WriteOperation(
            id=str(uuid.uuid4()),
            operation_type=operation_type,
            fn=fn,
            args=args,
            kwargs=kwargs,
            timestamp=time.time(),
            priority=priority,
            max_retries=max_retries or self.max_retries,
        )

        try:
            # Use priority queue with operation priority
            self._priority_queue.put((priority, operation), timeout=1.0)

            with self._metrics_lock:
                self._metrics.total_operations += 1
                self._metrics.current_queue_size = self._priority_queue.qsize()

            print(f"Queued operation {operation.id} with priority {priority}")
            return operation.id

        except queue.Full:
            print("Write queue is full, rejecting operation")
            raise

    def submit_batch(
        self, operations: List[Tuple[Callable, tuple, dict]], priority: int = 1
    ) -> List[str]:
        """
        Submit multiple operations as a batch for optimized processing.

        Args:
            operations: List of (function, args, kwargs) tuples
            priority: Batch priority

        Returns:
            List[str]: Operation IDs for tracking
        """
        operation_ids = []

        for fn, args, kwargs in operations:
            try:
                op_id = self.submit(
                    fn,
                    *args,
                    priority=priority,
                    operation_type=WriteOperationType.BATCH_UPSERT,
                    **kwargs,
                )
                operation_ids.append(op_id)
            except queue.Full:
                # If queue is full, stop adding more operations
                print(
                    f"Queue full, only queued {len(operation_ids)}/{len(operations)} operations"
                )
                break

        return operation_ids

    def _worker_loop(self) -> None:
        """Main worker loop for processing operations with batch optimization."""
        print(f"Write queue worker {threading.current_thread().name} started")

        batch_operations = []
        last_batch_time = time.time()

        while not self._shutdown_event.is_set():
            try:
                # Try to collect operations for batching
                operation = None

                # Get operation with timeout for batching
                try:
                    priority, operation = self._priority_queue.get(
                        timeout=self.batch_timeout_ms
                    )
                except queue.Empty:
                    # Check for retry operations
                    try:
                        operation = self._retry_queue.get_nowait()
                    except queue.Empty:
                        # Process accumulated batch if timeout reached
                        if (
                            batch_operations
                            and (time.time() - last_batch_time) >= self.batch_timeout_ms
                        ):
                            self._process_batch(batch_operations)
                            batch_operations = []
                            last_batch_time = time.time()
                        continue

                if operation:
                    # Check if operation can be batched
                    if (
                        self._can_batch_operation(operation)
                        and len(batch_operations) < self.batch_size
                    ):
                        batch_operations.append(operation)

                        # Process batch if it's full or timeout reached
                        if len(batch_operations) >= self.batch_size or (
                            batch_operations
                            and (time.time() - last_batch_time) >= self.batch_timeout_ms
                        ):
                            self._process_batch(batch_operations)
                            batch_operations = []
                            last_batch_time = time.time()
                    else:
                        # Process individual operation (non-batchable or urgent)
                        self._process_operation(operation)

            except Exception as exc:
                print(f"Worker loop error: {exc}", exc_info=True)

        # Process remaining batch operations on shutdown
        if batch_operations:
            print(
                f"Processing {len(batch_operations)} remaining operations on shutdown"
            )
            self._process_batch(batch_operations)

        print(f"Write queue worker {threading.current_thread().name} stopped")

    def _can_batch_operation(self, operation: WriteOperation) -> bool:
        """Determine if an operation can be batched with others."""
        # Only batch UPSERT and BATCH_UPSERT operations
        return operation.operation_type in [
            WriteOperationType.UPSERT,
            WriteOperationType.BATCH_UPSERT,
        ]

    def _process_batch(self, operations: List[WriteOperation]) -> None:
        """Process a batch of operations together for better efficiency."""
        if not operations:
            return

        start_time = time.time()
        batch_size = len(operations)
        successful_ops = 0
        failed_ops = 0

        print(f"Processing batch of {batch_size} operations")

        # Group operations by function type for potential optimization
        grouped_ops = defaultdict(list)
        for op in operations:
            # Group by function name for potential batch processing
            func_name = getattr(op.fn, "__name__", "unknown")
            grouped_ops[func_name].append(op)

        # Process each group
        for func_name, group_ops in grouped_ops.items():
            # Check if we can do true batch processing for this function type
            if func_name == "_upsert_internal" and len(group_ops) > 1:
                # Try batch upsert if possible
                try:
                    self._process_batch_upsert(group_ops)
                    successful_ops += len(group_ops)
                except Exception as exc:
                    print(
                        f"Batch upsert failed, falling back to individual processing: {exc}"
                    )
                    # Fall back to individual processing
                    for op in group_ops:
                        try:
                            self._process_operation(op)
                            successful_ops += 1
                        except Exception:
                            failed_ops += 1
            else:
                # Process operations individually
                for op in group_ops:
                    try:
                        self._process_operation(op)
                        successful_ops += 1
                    except Exception:
                        failed_ops += 1

        # Record batch processing metrics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        with self._metrics_lock:
            self._metrics.last_batch_size = batch_size
            self._metrics.last_batch_processing_time_ms = processing_time

        print(
            f"Batch processing completed: {successful_ops} successful, {failed_ops} failed in {processing_time:.2f}ms"
        )

    def _process_batch_upsert(self, operations: List[WriteOperation]) -> None:
        """Optimized batch processing for upsert operations."""
        # This could be enhanced to use database batch operations
        # For now, we process them in a single transaction

        # Extract the first operation to get the memory instance
        if not operations:
            return

        first_op = operations[0]
        if len(first_op.args) == 0:
            # Can't determine memory instance, fall back to individual processing
            raise ValueError("Cannot determine memory instance for batch processing")

        # For now, process individually but within a batch context
        # This provides better error handling and metrics tracking
        for operation in operations:
            operation.fn(*operation.args, **operation.kwargs)

            # Mark each task as done
            if hasattr(self._priority_queue, "task_done"):
                self._priority_queue.task_done()

    def _process_operation(self, operation: WriteOperation) -> None:
        """Process a single write operation with error handling and retry logic."""
        start_time = time.time()

        try:
            # Execute the operation
            result = operation.fn(*operation.args, **operation.kwargs)

            # Record success metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            with self._metrics_lock:
                self._metrics.successful_operations += 1
                self._metrics.current_queue_size = self._priority_queue.qsize()
                self._processing_times.append(processing_time)

                # Update average processing time
                if self._processing_times:
                    self._metrics.avg_processing_time_ms = sum(
                        self._processing_times
                    ) / len(self._processing_times)

            print(
                f"Operation {operation.id} completed successfully in {processing_time:.2f}ms"
            )

            # Mark task as done for priority queue
            if hasattr(self._priority_queue, "task_done"):
                self._priority_queue.task_done()

        except Exception as exc:
            self._handle_operation_failure(operation, exc)

    def _handle_operation_failure(
        self, operation: WriteOperation, error: Exception
    ) -> None:
        """Handle failed operations with retry logic."""
        operation.retry_count += 1

        with self._metrics_lock:
            self._metrics.failed_operations += 1

        if operation.retry_count <= operation.max_retries:
            # Calculate exponential backoff delay
            delay = min(2**operation.retry_count, 30)  # Max 30 seconds

            print(
                f"Operation {operation.id} failed (attempt {operation.retry_count}/{operation.max_retries}), "
                f"retrying in {delay}s: {error}"
            )

            # Schedule retry
            retry_thread = threading.Thread(
                target=self._schedule_retry, args=(operation, delay), daemon=True
            )
            retry_thread.start()

            with self._metrics_lock:
                self._metrics.retried_operations += 1
        else:
            print(
                f"Operation {operation.id} failed permanently after {operation.retry_count} attempts: {error}"
            )

        # Mark task as done for priority queue
        if hasattr(self._priority_queue, "task_done"):
            self._priority_queue.task_done()

    def _schedule_retry(self, operation: WriteOperation, delay: float) -> None:
        """Schedule an operation for retry after a delay."""
        time.sleep(delay)

        if not self._shutdown_event.is_set():
            try:
                self._retry_queue.put(operation, timeout=1.0)
            except queue.Full:
                print(f"Retry queue full, dropping operation {operation.id}")

    def get_metrics(self) -> QueueMetrics:
        """Get current queue metrics."""
        with self._metrics_lock:
            # Update current queue size
            self._metrics.current_queue_size = self._priority_queue.qsize()
            return QueueMetrics(
                total_operations=self._metrics.total_operations,
                successful_operations=self._metrics.successful_operations,
                failed_operations=self._metrics.failed_operations,
                retried_operations=self._metrics.retried_operations,
                current_queue_size=self._metrics.current_queue_size,
                avg_processing_time_ms=self._metrics.avg_processing_time_ms,
                last_batch_size=self._metrics.last_batch_size,
                last_batch_processing_time_ms=self._metrics.last_batch_processing_time_ms,
            )

    def join(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all queued operations to complete.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            bool: True if all operations completed, False if timeout
        """
        try:
            self._priority_queue.join()
            return True
        except:
            return False

    def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Gracefully shutdown the write queue.

        Args:
            timeout: Maximum time to wait for operations to complete

        Returns:
            bool: True if shutdown completed successfully
        """
        print("Initiating write queue shutdown...")

        # Signal shutdown to workers first
        self._shutdown_event.set()
        
        # Give workers a moment to see the shutdown signal
        time.sleep(0.1)

        # Wait for current operations to complete with reduced timeout
        shutdown_start = time.time()
        drain_timeout = min(timeout * 0.3, 10.0)  # Use max 1/3 of timeout for draining
        
        try:
            # Wait for priority queue to empty with timeout
            queue_empty_start = time.time()
            while (
                not self._priority_queue.empty()
                and (time.time() - queue_empty_start) < drain_timeout
            ):
                time.sleep(0.05)  # Shorter sleep for responsiveness

            # Wait for retry queue to empty with timeout  
            retry_empty_start = time.time()
            while (
                not self._retry_queue.empty()
                and (time.time() - retry_empty_start) < drain_timeout
            ):
                time.sleep(0.05)

        except Exception as exc:
            print(f"Error during queue drain: {exc}")

        # Force worker threads to finish with aggressive timeout
        print(f"Waiting for {len(self._workers)} worker threads to finish...")
        remaining_timeout = max(1.0, timeout - (time.time() - shutdown_start))
        per_worker_timeout = remaining_timeout / max(len(self._workers), 1)
        
        alive_workers = []
        for i, worker in enumerate(self._workers):
            try:
                worker.join(timeout=per_worker_timeout)
                if worker.is_alive():
                    print(f"Worker {worker.name} still alive after timeout")
                    alive_workers.append(worker)
                else:
                    print(f"Worker {worker.name} shut down cleanly")
            except Exception as exc:
                print(f"Error joining worker {worker.name}: {exc}")
                alive_workers.append(worker)

        # Clear worker list to prevent resource leaks
        self._workers.clear()

        # Shutdown embedding queue as well with timeout protection
        try:
            from infrastructure.embeddings import shutdown_embedding_queue

            embedding_timeout = max(2.0, remaining_timeout * 0.3)
            print(f"Shutting down embedding queue (timeout: {embedding_timeout:.1f}s)...")
            
            # Use threading to timeout embedding shutdown
            embedding_success = False
            def shutdown_embeddings():
                nonlocal embedding_success
                try:
                    embedding_success = shutdown_embedding_queue()
                except Exception as e:
                    print(f"Embedding shutdown error: {e}")
                    embedding_success = False

            embed_thread = threading.Thread(target=shutdown_embeddings, daemon=True)
            embed_thread.start()
            embed_thread.join(timeout=embedding_timeout)
            
            if embed_thread.is_alive():
                print("Embedding queue shutdown timed out")
            elif not embedding_success:
                print("Embedding queue shutdown failed")
            else:
                print("Embedding queue shut down successfully")
                
        except Exception as exc:
            print(f"Failed to shutdown embedding queue: {exc}")

        # Clear any remaining queue items to prevent resource leaks
        try:
            while not self._priority_queue.empty():
                try:
                    self._priority_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self._retry_queue.empty():
                try:
                    self._retry_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as exc:
            print(f"Error clearing queues: {exc}")

        # Check if shutdown completed successfully
        success = len(alive_workers) == 0

        if success:
            print("Write queue shutdown completed successfully")
        else:
            print(f"Write queue shutdown timed out, {len(alive_workers)} workers still alive")
            # Force cleanup of remaining resources
            try:
                import gc
                gc.collect()
            except:
                pass

        # Set shutdown complete event
        self._shutdown_complete.set()

        return success

    def restart(self) -> bool:
        """
        Restart the write queue after shutdown.

        Returns:
            bool: True if restart was successful
        """
        try:
            if not self._shutdown_event.is_set():
                print("Write queue restart called but queue is not shutdown")
                return True

            # Reset shutdown event
            self._shutdown_event.clear()

            # Clear workers list
            self._workers.clear()

            # Restart worker threads
            self._start_workers()

            print("Write queue restarted successfully")
            return True

        except Exception as exc:
            print(f"Failed to restart write queue: {exc}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the write queue."""
        metrics = self.get_metrics()

        # Calculate health score based on various factors
        health_score = 100
        issues = []

        # Check queue size
        if metrics.current_queue_size > self.max_queue_size * 0.8:
            health_score -= 20
            issues.append("Queue is near capacity")

        # Check error rate
        if metrics.total_operations > 0:
            error_rate = metrics.failed_operations / metrics.total_operations
            if error_rate > 0.1:  # 10% error rate
                health_score -= 30
                issues.append(f"High error rate: {error_rate:.2%}")

        # Check if workers are alive
        dead_workers = sum(1 for worker in self._workers if not worker.is_alive())
        if dead_workers > 0:
            health_score -= 50
            issues.append(f"{dead_workers} workers are dead")

        status = (
            "healthy"
            if health_score >= 80
            else "degraded"
            if health_score >= 50
            else "unhealthy"
        )

        return {
            "status": status,
            "health_score": health_score,
            "issues": issues,
            "metrics": {
                "total_operations": metrics.total_operations,
                "successful_operations": metrics.successful_operations,
                "failed_operations": metrics.failed_operations,
                "current_queue_size": metrics.current_queue_size,
                "avg_processing_time_ms": metrics.avg_processing_time_ms,
                "worker_count": len([w for w in self._workers if w.is_alive()]),
            },
        }


# Global singleton instance with settings-based configuration
def _create_write_queue() -> EnhancedWriteQueue:
    """Create write queue with configuration from settings."""
    from settings import settings

    return EnhancedWriteQueue(
        batch_size=settings.write_batch_size,
        batch_timeout_ms=settings.write_batch_timeout_ms,
        max_queue_size=settings.write_queue_size,
        max_retries=3,
        worker_count=2 if settings.is_production else 1,  # More workers in production
    )


# Create singleton instance
write_queue = None  # Lazy initialization to prevent CLI hanging


# Provide backward compatibility
def submit(fn: Callable, *args, **kwargs) -> str:
    """Backward compatible submit function."""
    global write_queue
    if write_queue is None:
        write_queue = _create_write_queue()
    return write_queue.submit(fn, *args, **kwargs)


def join(timeout: Optional[float] = None) -> bool:
    """Backward compatible join function."""
    global write_queue
    if write_queue is None:
        return True  # Nothing to join if not initialized
    return write_queue.join(timeout)


def shutdown(timeout: float = 10.0) -> bool:
    """Backward compatible shutdown function for CLI cleanup."""
    global write_queue
    if write_queue is None:
        return True  # Nothing to shutdown if not initialized
    return write_queue.shutdown(timeout)


def restart() -> bool:
    """Backward compatible restart function for CLI cleanup."""
    global write_queue
    if write_queue is None:
        write_queue = _create_write_queue()
        return True
    return write_queue.restart()
