#!/usr/bin/env python3
"""
Demonstration script for the enhanced write queue reliability improvements.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from sara.infrastructure.write_queue import (EnhancedWriteQueue,
                                               WriteOperationType)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_database_write(task_id: str, content: str, delay: float = 0.01) -> str:
    """Simulate a database write operation."""
    time.sleep(delay)  # Simulate database latency
    result = f"Written task_{task_id}: {content[:20]}..."
    logger.debug(f"Database write completed: {result}")
    return result


def simulate_failing_operation() -> None:
    """Simulate an operation that fails sometimes."""
    import random

    if random.random() < 0.3:  # 30% failure rate
        raise RuntimeError("Simulated database error")
    time.sleep(0.01)


def demonstrate_basic_functionality():
    """Demonstrate basic write queue functionality."""
    print("\n🔧 Testing Basic Functionality")
    print("-" * 50)

    queue = EnhancedWriteQueue(
        batch_size=5, batch_timeout_ms=200, max_queue_size=100, worker_count=2
    )

    try:
        # Submit some operations
        results = []
        for i in range(10):
            op_id = queue.submit(
                simulate_database_write,
                f"task-{i}",
                f"Content for task {i}",
                operation_type=WriteOperationType.UPSERT,
                priority=1 if i < 5 else 2,  # First 5 have higher priority
            )
            results.append(op_id)

        # Wait for completion
        success = queue.join(timeout=5.0)
        print(f"✅ All operations completed: {success}")

        # Show metrics
        metrics = queue.get_metrics()
        print(f"📊 Metrics:")
        print(f"   - Total operations: {metrics.total_operations}")
        print(f"   - Successful: {metrics.successful_operations}")
        print(f"   - Failed: {metrics.failed_operations}")
        print(f"   - Average processing time: {metrics.avg_processing_time_ms:.2f}ms")

    finally:
        queue.shutdown(timeout=5.0)


def demonstrate_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n⚠️  Testing Error Handling & Retry Logic")
    print("-" * 50)

    queue = EnhancedWriteQueue(batch_size=3, max_retries=2, worker_count=1)

    try:
        # Submit operations that will fail sometimes
        for i in range(5):
            queue.submit(
                simulate_failing_operation,
                max_retries=2,
                operation_type=WriteOperationType.UPSERT,
            )

        # Wait for completion (including retries)
        time.sleep(3.0)

        # Show metrics
        metrics = queue.get_metrics()
        print(f"📊 Error Handling Metrics:")
        print(f"   - Total operations: {metrics.total_operations}")
        print(f"   - Successful: {metrics.successful_operations}")
        print(f"   - Failed: {metrics.failed_operations}")
        print(f"   - Retries attempted: {metrics.retried_operations}")

        # Show health status
        health = queue.health_check()
        print(f"🏥 Health Status: {health['status']} (score: {health['health_score']})")
        if health["issues"]:
            print(f"   - Issues: {', '.join(health['issues'])}")

    finally:
        queue.shutdown(timeout=5.0)


def demonstrate_concurrent_load():
    """Demonstrate concurrent load handling."""
    print("\n🚀 Testing Concurrent Load")
    print("-" * 50)

    queue = EnhancedWriteQueue(
        batch_size=10, batch_timeout_ms=100, max_queue_size=500, worker_count=3
    )

    def worker_thread(thread_id: int, operations_per_thread: int):
        """Submit operations from a worker thread."""
        for i in range(operations_per_thread):
            try:
                queue.submit(
                    simulate_database_write,
                    f"t{thread_id}-{i}",
                    f"Thread {thread_id} operation {i}",
                    0.001,  # Faster operations for load test
                    operation_type=WriteOperationType.BATCH_UPSERT,
                )
            except Exception as e:
                logger.warning(
                    f"Thread {thread_id} failed to submit operation {i}: {e}"
                )

    try:
        start_time = time.time()

        # Launch multiple threads submitting operations
        thread_count = 5
        ops_per_thread = 50
        total_expected = thread_count * ops_per_thread

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(worker_thread, i, ops_per_thread)
                for i in range(thread_count)
            ]

            # Wait for all submissions to complete
            for future in futures:
                future.result()

        print(
            f"⏱️  Submitted {total_expected} operations in {time.time() - start_time:.2f}s"
        )

        # Wait for processing
        processing_start = time.time()
        success = queue.join(timeout=10.0)
        processing_time = time.time() - processing_start

        print(f"✅ Processing completed: {success} in {processing_time:.2f}s")

        # Final metrics
        metrics = queue.get_metrics()
        throughput = (
            metrics.successful_operations / processing_time
            if processing_time > 0
            else 0
        )

        print(f"📊 Load Test Results:")
        print(f"   - Total submitted: {total_expected}")
        print(f"   - Successfully processed: {metrics.successful_operations}")
        print(f"   - Failed: {metrics.failed_operations}")
        print(f"   - Throughput: {throughput:.1f} ops/sec")
        print(f"   - Average latency: {metrics.avg_processing_time_ms:.2f}ms")

    finally:
        queue.shutdown(timeout=10.0)


def main():
    """Run all demonstrations."""
    print("🎭 Enhanced Write Queue Reliability Demonstration")
    print("=" * 60)

    # Run demonstrations
    demonstrate_basic_functionality()
    demonstrate_error_handling()
    demonstrate_concurrent_load()

    print("\n🎉 All demonstrations completed!")
    print("=" * 60)
    print("\n📋 Summary of Improvements:")
    print("✅ Batch processing for better throughput")
    print("✅ Priority queue for operation ordering")
    print("✅ Exponential backoff retry logic")
    print("✅ Comprehensive error handling")
    print("✅ Graceful shutdown mechanism")
    print("✅ Real-time metrics and health monitoring")
    print("✅ Concurrent load handling")
    print("✅ Backpressure protection")


if __name__ == "__main__":
    main()
