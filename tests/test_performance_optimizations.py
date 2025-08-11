"""
Performance tests for Serena search engine optimizations.

This module contains comprehensive performance tests and benchmarks for:
1. Async embedding queue performance
2. Search engine optimization
3. Embedding model lifecycle management
4. Batch vector operations
"""

import asyncio
import random
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sara.infrastructure.embeddings import (
    AsyncEmbeddingQueue, EmbeddingGenerator, batch_cosine_similarity,
    fast_similarity_with_precomputed, get_embedding_queue,
    optimized_batch_cosine_similarity, precompute_vector_norms,
    shutdown_embedding_queue)
from sara.infrastructure.search.search_core import SearchEngine


class TestAsyncEmbeddingQueue:
    """Test async embedding queue performance and functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Ensure clean state
        shutdown_embedding_queue()

    def teardown_method(self):
        """Clean up after tests."""
        shutdown_embedding_queue()

    @pytest.fixture
    def mock_generator(self):
        """Mock embedding generator for testing."""
        generator = Mock(spec=EmbeddingGenerator)
        generator.load_model_now.return_value = True
        generator.generate_embeddings_batch.return_value = [
            [0.1, 0.2, 0.3] * 128 for _ in range(10)  # 384-dim vectors
        ]
        return generator

    def test_embedding_queue_creation(self):
        """Test embedding queue creation and basic functionality."""
        queue = AsyncEmbeddingQueue(max_queue_size=100, worker_count=1)

        assert queue.max_queue_size == 100
        assert queue.worker_count == 1
        assert len(queue._workers) == 1
        assert all(worker.is_alive() for worker in queue._workers)

        # Test stats
        stats = queue.get_stats()
        assert stats["total_requests"] == 0
        assert stats["successful_embeddings"] == 0
        assert stats["queue_size"] == 0
        assert stats["workers_alive"] == 1

        # Clean shutdown
        assert queue.shutdown(timeout=5.0) is True

    @patch("sara.infrastructure.embeddings.get_default_generator")
    def test_embedding_request_processing(self, mock_get_generator):
        """Test embedding request processing performance."""
        # Setup mock
        mock_generator = Mock(spec=EmbeddingGenerator)
        mock_generator.load_model_now.return_value = True
        mock_generator.generate_embeddings_batch.return_value = [
            [0.1] * 384 for _ in range(5)
        ]
        mock_get_generator.return_value = mock_generator

        queue = AsyncEmbeddingQueue(max_queue_size=50, worker_count=2)

        # Submit multiple requests
        start_time = time.time()
        success_count = 0

        for i in range(10):
            chunks = [(f"test content {i}", 0) for _ in range(5)]
            if queue.submit_embedding_request(f"task_{i}", chunks):
                success_count += 1

        # Wait for processing
        time.sleep(2.0)

        processing_time = time.time() - start_time
        stats = queue.get_stats()

        assert success_count == 10
        assert stats["total_requests"] == 10
        assert processing_time < 5.0  # Should be fast

        # Cleanup
        queue.shutdown(timeout=5.0)

    def test_queue_overflow_handling(self):
        """Test queue behavior when at capacity."""
        queue = AsyncEmbeddingQueue(max_queue_size=2, worker_count=1)

        # Fill queue beyond capacity
        results = []
        for i in range(5):
            result = queue.submit_embedding_request(
                f"task_{i}", [("content", 0)], priority=1
            )
            results.append(result)

        # Some requests should fail due to queue being full
        assert sum(results) < 5

        stats = queue.get_stats()
        assert stats["queue_full_errors"] > 0

        queue.shutdown(timeout=5.0)


class TestBatchVectorOperations:
    """Test batch vector operation performance improvements."""

    def generate_test_vectors(self, count: int, dim: int = 384) -> List[List[float]]:
        """Generate test vectors for benchmarking."""
        return [[random.random() for _ in range(dim)] for _ in range(count)]

    def test_batch_similarity_performance(self):
        """Test performance of different batch similarity implementations."""
        # Generate test data
        query_vector = [random.random() for _ in range(384)]

        # Test with different vector set sizes
        for vector_count in [100, 500, 1000]:
            vectors = self.generate_test_vectors(vector_count)

            # Test standard batch similarity
            start_time = time.time()
            standard_results = batch_cosine_similarity(query_vector, vectors)
            standard_time = time.time() - start_time

            # Test optimized batch similarity
            start_time = time.time()
            optimized_results = optimized_batch_cosine_similarity(query_vector, vectors)
            optimized_time = time.time() - start_time

            # Results should be similar
            assert len(standard_results) == len(optimized_results) == vector_count

            # For large sets, optimized should be faster or similar
            if vector_count >= 500:
                assert optimized_time <= standard_time * 1.2  # Allow 20% variance

            # Verify numerical similarity (allowing for floating point differences)
            for std, opt in zip(standard_results[:10], optimized_results[:10]):
                assert abs(std - opt) < 0.001

    def test_precomputed_similarity_performance(self):
        """Test precomputed similarity calculation performance."""
        query_vector = [random.random() for _ in range(384)]
        vectors = self.generate_test_vectors(1000)

        # Precompute normalized vectors
        start_time = time.time()
        normalized_vectors, norms = precompute_vector_norms(vectors)
        precompute_time = time.time() - start_time

        # Test multiple queries with precomputed vectors
        queries = [self.generate_test_vectors(1, 384)[0] for _ in range(10)]

        # Standard approach (recompute each time)
        start_time = time.time()
        standard_results = []
        for query in queries:
            standard_results.append(batch_cosine_similarity(query, vectors))
        standard_total_time = time.time() - start_time

        # Precomputed approach
        start_time = time.time()
        precomputed_results = []
        for query in queries:
            precomputed_results.append(
                fast_similarity_with_precomputed(query, normalized_vectors, norms)
            )
        precomputed_total_time = time.time() - start_time

        # Precomputed should be faster for multiple queries
        total_precomputed_time = precompute_time + precomputed_total_time

        # For multiple queries, precomputed approach should be more efficient
        assert len(precomputed_results) == len(standard_results) == 10
        assert precomputed_total_time < standard_total_time

        print(f"Standard time: {standard_total_time:.4f}s")
        print(f"Precomputed time: {total_precomputed_time:.4f}s (including precompute)")
        print(f"Speedup: {standard_total_time / precomputed_total_time:.2f}x")

    def test_memory_efficiency(self):
        """Test memory efficiency of batch operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Generate large vector set
        large_vectors = self.generate_test_vectors(5000, 384)
        query_vector = [random.random() for _ in range(384)]

        # Memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Process with optimized batch similarity
        results = optimized_batch_cosine_similarity(
            query_vector, large_vectors, batch_size=1000
        )

        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        assert len(results) == 5000
        assert memory_increase < 500  # Should not increase memory by more than 500MB

        print(f"Memory increase: {memory_increase:.1f}MB for 5000 vectors")


class TestSearchEngineOptimizations:
    """Test search engine performance optimizations."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session for testing."""
        session = Mock()
        session.query.return_value.filter_by.return_value.delete.return_value = None
        session.commit.return_value = None
        session.execute.return_value.fetchall.return_value = []
        return session

    @patch("sara.infrastructure.search.search_core.get_session")
    def test_text_only_search_fallback(self, mock_get_session):
        """Test text-only search fallback performance."""
        # Mock database session
        mock_session = Mock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_get_session.return_value.__exit__.return_value = None

        # Mock query results
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("task1", "Test Title", "archive", "completed", "2023-01-01", "/path", 0.8),
            (
                "task2",
                "Another Task",
                "archive",
                "pending",
                "2023-01-02",
                "/path2",
                0.7,
            ),
        ]
        mock_session.execute.return_value = mock_result

        search_engine = SearchEngine()

        # Test text-only search
        start_time = time.time()
        results = search_engine._text_only_search("test query", k=10)
        search_time = time.time() - start_time

        assert len(results) == 2
        assert search_time < 1.0  # Should be fast
        assert all(result.task_id in ["task1", "task2"] for result in results)

    def test_search_with_empty_embeddings(self):
        """Test search behavior when no embeddings are available."""
        with patch(
            "sara.infrastructure.search.search_core.get_session"
        ) as mock_get_session:
            # Mock empty embeddings
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_get_session.return_value.__exit__.return_value = None

            mock_result = Mock()
            mock_result.fetchall.return_value = []
            mock_session.execute.return_value = mock_result

            search_engine = SearchEngine()

            # Should fall back to text-only search
            with patch.object(search_engine, "_text_only_search") as mock_text_search:
                mock_text_search.return_value = []

                results = search_engine.search("test query")

                assert mock_text_search.called
                assert results == []


class TestEmbeddingModelLifecycle:
    """Test embedding model lifecycle management improvements."""

    def test_adaptive_cleanup_timer(self):
        """Test adaptive cleanup timer functionality."""
        generator = EmbeddingGenerator()

        # Simulate different usage patterns
        generator._usage_stats["total_requests"] = 100
        generator._last_used = time.time()
        generator._usage_stats["total_processing_time"] = 10.0

        # Test adaptive timeout calculation
        adaptive_timeout = generator._calculate_adaptive_timeout()
        base_timeout = generator._idle_timeout

        # Should be between 0.5x and 2x base timeout
        assert base_timeout * 0.5 <= adaptive_timeout <= base_timeout * 2

    def test_memory_monitoring(self):
        """Test memory monitoring and aggressive cleanup."""
        with patch("psutil.Process") as mock_process:
            # Mock high memory usage
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = (
                3 * 1024 * 1024 * 1024
            )  # 3GB
            mock_process.return_value = mock_process_instance

            generator = EmbeddingGenerator()
            generator._model = Mock()  # Mock model for cleanup testing

            # This should trigger aggressive cleanup
            with patch.object(
                generator, "_schedule_aggressive_cleanup"
            ) as mock_cleanup:
                generator._update_usage_stats(0.1, is_batch=False)
                mock_cleanup.assert_called_once()

    def test_usage_stats_tracking(self):
        """Test usage statistics tracking accuracy."""
        generator = EmbeddingGenerator()

        # Simulate processing
        initial_stats = generator.get_usage_stats()
        assert initial_stats["total_requests"] == 0

        # Mock model for testing
        with patch.object(generator, "model") as mock_model:
            mock_model.encode.return_value = np.array([0.1] * 384)

            # Process some embeddings
            generator.generate_embedding("test text")

            updated_stats = generator.get_usage_stats()
            assert updated_stats["total_requests"] == 1
            assert updated_stats["total_processing_time"] > 0


class TestIntegratedPerformance:
    """Test integrated performance of all optimizations."""

    @pytest.mark.performance
    def test_end_to_end_performance(self):
        """Test end-to-end performance improvements."""
        # This test requires actual embedding models and database
        # Mark as performance test to run separately

        # Setup test data
        test_tasks = [
            f"Task {i}: This is test content for performance testing. "
            f"It contains various keywords and phrases to test search functionality. "
            f"The content is designed to be realistic and varied."
            for i in range(100)
        ]

        # Simulate write operations with async embedding queue
        with patch(
            "sara.infrastructure.embeddings.get_default_generator"
        ) as mock_gen:
            mock_generator = Mock()
            mock_generator.load_model_now.return_value = True
            mock_generator.generate_embeddings_batch.return_value = [
                [random.random() for _ in range(384)] for _ in range(10)
            ]
            mock_gen.return_value = mock_generator

            queue = get_embedding_queue()

            # Measure write performance
            start_time = time.time()

            for i, content in enumerate(test_tasks):
                chunks = [(content, 0)]
                queue.submit_embedding_request(f"task_{i}", chunks)

            write_time = time.time() - start_time

            # Should be very fast since writes are async
            assert write_time < 1.0

            # Wait for processing
            time.sleep(2.0)

            stats = queue.get_stats()
            assert stats["total_requests"] == 100

            shutdown_embedding_queue()

    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate sustained load
        generator = EmbeddingGenerator()

        with patch.object(generator, "model") as mock_model:
            mock_model.encode.return_value = np.array([[0.1] * 384] * 32)

            # Process many batches
            for i in range(50):
                texts = [f"test text {j}" for j in range(32)]
                generator.generate_embeddings_batch(texts)

                # Check memory periodically
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory

                    # Memory should not grow excessively
                    assert memory_increase < 1000  # Less than 1GB increase

        # Force cleanup
        generator.force_cleanup()

        final_memory = process.memory_info().rss / 1024 / 1024
        final_increase = final_memory - initial_memory

        print(f"Final memory increase: {final_increase:.1f}MB")
        assert final_increase < 500  # Should be reasonable after cleanup


if __name__ == "__main__":
    # Run basic performance tests
    print("Running Serena Performance Tests...")

    # Test batch similarity performance
    test = TestBatchVectorOperations()
    test.test_batch_similarity_performance()
    test.test_precomputed_similarity_performance()

    print("Performance tests completed successfully!")
