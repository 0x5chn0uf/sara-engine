"""Embedding generation and utilities.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from settings import settings


class EmbeddingGenerator:
    """Handles embedding generation using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model: "Optional[sentence_transformers.SentenceTransformer]" = None
        self._loading_thread: Optional[threading.Thread] = None
        self.embedding_dim = 384  # Default for MiniLM
        self._device = device or self._detect_optimal_device()
        self._lock = threading.Lock()  # Thread safety for model loading

        # Model lifecycle management
        self._last_used = time.time()
        self._cleanup_timer: Optional[threading.Timer] = None
        self._idle_timeout = 900  # 15 minutes idle timeout
        self._cleanup_enabled = True

        # Performance monitoring
        self._usage_stats = {
            "total_requests": 0,
            "batch_requests": 0,
            "total_processing_time": 0.0,
            "last_cleanup": None,
            "memory_peak_mb": 0.0,
        }

        print(
            f"EmbeddingGenerator initialized with device: {self._device}, idle_timeout: {self._idle_timeout}s"
        )

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------
    def _detect_optimal_device(self) -> str:
        """Detect the optimal device for embedding generation."""
        # Check settings override
        if settings.device:
            print("Using device from settings: %s", settings.device)
            return settings.device

        # Check for CUDA availability
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print("GPU detected: %s (count: %d)", gpu_name, gpu_count)
                return "cuda"
        except ImportError:
            print("PyTorch not available, checking for MPS...")
        except Exception as exc:
            print("CUDA check failed: %s", exc)

        # Check for MPS (Apple Silicon) availability
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("MPS (Apple Silicon) detected")
                return "mps"
        except (ImportError, AttributeError):
            print("MPS not available")
        except Exception as exc:
            print("MPS check failed: %s", exc)

        print("Using CPU fallback")
        return "cpu"

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    @property
    def model(self):  # noqa: D401
        if self._model is None:
            with self._lock:  # Thread-safe model loading
                if self._model is None:  # Double-check after acquiring lock
                    # Avoid blocking event loop
                    in_loop = False
                    try:
                        asyncio.get_running_loop()
                        in_loop = True
                    except RuntimeError:
                        pass

                    if in_loop:
                        if (
                            not self._loading_thread
                            or not self._loading_thread.is_alive()
                        ):
                            self._loading_thread = threading.Thread(
                                target=self._load_model_sync,
                                daemon=True,
                                name="embedding-model-loader",
                            )
                            self._loading_thread.start()
                        return None
                    self._load_model_sync()
        return self._model

    def load_model_now(self) -> bool:
        """Force synchronous model loading, waiting for completion if needed."""
        if self._model is not None:
            return True  # Already loaded

        # If we're in an async context and threading is being used
        if self._loading_thread and self._loading_thread.is_alive():
            print("Waiting for background model loading to complete...")
            self._loading_thread.join(timeout=30)  # Wait up to 30 seconds

        # If still not loaded, force synchronous loading
        if self._model is None:
            self._load_model_sync()

        return self._model is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model_sync(self) -> None:  # noqa: WPS213
        try:
            import numpy as _np

            if int(_np.__version__.split(".")[0]) >= 2:
                print(
                    "NumPy %s incompatible with sentence-transformers", _np.__version__
                )
                return
        except Exception:
            pass

        try:
            from pathlib import Path

            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            # Use local cache directory for faster loading
            cache_dir = Path.home() / ".cache" / "serena" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)

            print(f"Loading embedding model: {self.model_name}")
            print(f"Using cache directory: {cache_dir}")

            # Load with local cache and device optimization
            print(f"Loading model on device: {self._device}")
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(cache_dir),
                device=self._device,
            )

            # Verify device assignment
            actual_device = getattr(self._model.device, "type", str(self._model.device))
            if actual_device != self._device and self._device != "cpu":
                print(
                    f"Model loaded on {actual_device} instead of requested {self._device}, falling back to CPU"
                )
                # Reload on CPU if GPU failed
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(cache_dir),
                    device="cpu",
                )
                self._device = "cpu"

            print(f"Model successfully loaded on device: {self._device}")
        except ImportError:
            print("sentence-transformers not installed – embeddings disabled")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to load embedding model: {exc}")

    # ------------------------------------------------------------------
    # Model lifecycle management
    # ------------------------------------------------------------------
    def _update_usage_stats(
        self, processing_time: float, is_batch: bool = False
    ) -> None:
        """Update usage statistics and trigger cleanup timer reset."""
        self._last_used = time.time()
        self._usage_stats["total_requests"] += 1
        self._usage_stats["total_processing_time"] += processing_time

        if is_batch:
            self._usage_stats["batch_requests"] += 1

        # Reset cleanup timer with adaptive timeout based on usage
        self._reset_cleanup_timer_adaptive()

        # Monitor memory usage (optional, lightweight check)
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self._usage_stats["memory_peak_mb"]:
                self._usage_stats["memory_peak_mb"] = memory_mb

            # Trigger aggressive cleanup if memory usage is very high
            if memory_mb > 2048:  # 2GB threshold
                print(
                    f"High memory usage detected: {memory_mb:.1f}MB, scheduling cleanup"
                )
                self._schedule_aggressive_cleanup()

        except ImportError:
            pass  # psutil not available, skip memory monitoring
        except Exception:
            pass  # Ignore memory monitoring errors

    def _reset_cleanup_timer(self) -> None:
        """Reset the model cleanup timer."""
        if not self._cleanup_enabled:
            return

        # Cancel existing timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        # Start new timer
        self._cleanup_timer = threading.Timer(
            self._idle_timeout, self._cleanup_model_if_idle
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _reset_cleanup_timer_adaptive(self) -> None:
        """Reset the model cleanup timer with adaptive timeout based on usage patterns."""
        if not self._cleanup_enabled:
            return

        # Cancel existing timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        # Calculate adaptive timeout based on usage frequency
        adaptive_timeout = self._calculate_adaptive_timeout()

        # Start new timer with adaptive timeout
        self._cleanup_timer = threading.Timer(
            adaptive_timeout, self._cleanup_model_if_idle
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _calculate_adaptive_timeout(self) -> float:
        """Calculate adaptive cleanup timeout based on usage patterns."""
        # Base timeout
        base_timeout = self._idle_timeout

        # Adjust based on request frequency
        total_requests = self._usage_stats["total_requests"]

        if total_requests == 0:
            return base_timeout

        # Calculate requests per minute over lifetime
        uptime_minutes = (
            time.time() - (self._last_used - self._usage_stats["total_processing_time"])
        ) / 60
        requests_per_minute = total_requests / max(uptime_minutes, 1)

        # High usage (>10 req/min): longer timeout to avoid frequent reloading
        if requests_per_minute > 10:
            return base_timeout * 2
        # Medium usage (2-10 req/min): standard timeout
        elif requests_per_minute > 2:
            return base_timeout
        # Low usage (<2 req/min): shorter timeout to free memory sooner
        else:
            return base_timeout * 0.5

    def _schedule_aggressive_cleanup(self) -> None:
        """Schedule aggressive cleanup for high memory usage situations."""

        def aggressive_cleanup():
            if self._model is not None:
                print("Performing aggressive cleanup due to high memory usage")

                # Try to free GPU memory more aggressively
                try:
                    if hasattr(self._model, "device") and "cuda" in str(
                        self._model.device
                    ):
                        import torch

                        # Clear all caches
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        # Force garbage collection
                        import gc

                        gc.collect()
                except Exception as exc:
                    print(f"Aggressive GPU cleanup failed: {exc}")

                # Force cleanup regardless of idle time
                self.force_cleanup()

        # Schedule cleanup in a separate thread to avoid blocking
        cleanup_thread = threading.Thread(
            target=aggressive_cleanup, daemon=True, name="aggressive-cleanup"
        )
        cleanup_thread.start()

    def _cleanup_model_if_idle(self) -> None:
        """Clean up model if it has been idle for too long."""
        if not self._cleanup_enabled:
            return

        current_time = time.time()
        idle_time = current_time - self._last_used

        # Only cleanup if truly idle and cleanup is still enabled
        if idle_time >= self._idle_timeout and self._model is not None:
            print(
                f"Cleaning up embedding model after {idle_time / 60:.1f} minutes of inactivity"
            )

            with self._lock:
                # Double-check in case model was used between timer fire and lock acquisition
                if current_time - self._last_used >= self._idle_timeout:
                    try:
                        # Free model memory
                        if self._model is not None:
                            # Try to free GPU memory if applicable
                            try:
                                if hasattr(self._model, "device") and "cuda" in str(
                                    self._model.device
                                ):
                                    import torch

                                    torch.cuda.empty_cache()
                            except Exception:
                                pass  # Ignore GPU cleanup errors

                            # Clear model reference
                            self._model = None

                        # Record cleanup time
                        self._usage_stats["last_cleanup"] = current_time

                        print("Embedding model cleanup completed")

                    except Exception as exc:
                        print(f"Model cleanup failed: {exc}")

    def get_usage_stats(self) -> dict:
        """Get current usage statistics."""
        current_time = time.time()
        return {
            **self._usage_stats,
            "model_loaded": self._model is not None,
            "idle_time_seconds": current_time - self._last_used,
            "device": self._device,
            "cleanup_enabled": self._cleanup_enabled,
        }

    def disable_cleanup(self) -> None:
        """Disable automatic model cleanup (useful for high-frequency usage)."""
        self._cleanup_enabled = False
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
        print("Model cleanup disabled")

    def enable_cleanup(self, idle_timeout: Optional[int] = None) -> None:
        """Enable automatic model cleanup with optional timeout override."""
        self._cleanup_enabled = True
        if idle_timeout is not None:
            self._idle_timeout = idle_timeout
        self._reset_cleanup_timer()
        print(f"Model cleanup enabled with {self._idle_timeout} second timeout")

    def force_cleanup(self) -> bool:
        """Force immediate model cleanup, returns True if model was cleaned up."""
        if self._model is None:
            return False

        with self._lock:
            if self._model is not None:
                try:
                    # Cancel any pending cleanup timer
                    if self._cleanup_timer:
                        try:
                            self._cleanup_timer.cancel()
                            self._cleanup_timer = None
                        except Exception as e:
                            print(f"Timer cleanup error: {e}")

                    # Wait for any loading thread to complete
                    if self._loading_thread and self._loading_thread.is_alive():
                        try:
                            self._loading_thread.join(timeout=2.0)
                        except Exception as e:
                            print(f"Loading thread cleanup error: {e}")

                    # Free GPU memory if applicable
                    try:
                        if hasattr(self._model, "device") and "cuda" in str(
                            self._model.device
                        ):
                            import torch
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception as e:
                        print(f"CUDA cleanup error: {e}")

                    # Clear the model reference
                    self._model = None
                    self._usage_stats["last_cleanup"] = time.time()

                    # Force garbage collection to free memory
                    import gc
                    gc.collect()

                    print("Forced embedding model cleanup completed")
                    return True

                except Exception as exc:
                    print(f"Forced cleanup failed: {exc}")
                    # Still try to clear the model reference
                    try:
                        self._model = None
                    except:
                        pass

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_embedding(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self.embedding_dim

        if self.model is None:
            # Embedding model was expected but failed to load – abort early.
            raise RuntimeError("Embedding model not loaded; aborting pipeline")

        # Track processing time
        start_time = time.time()

        # Happy-path: model present
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)  # type: ignore[attr-defined]
        result = vec.tolist()
        del vec  # Explicit cleanup of numpy array

        # Update usage statistics
        processing_time = time.time() - start_time
        self._update_usage_stats(processing_time, is_batch=False)

        return result

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.model is None:
            raise RuntimeError("Embedding model not loaded; aborting pipeline")

        # Track processing time
        start_time = time.time()

        # Happy-path batch encode – any exception bubbles up to caller
        # Removed verbose logging: Starting batch encoding for {len(texts)} texts
        vecs = self.model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)  # type: ignore[attr-defined]
        # Removed verbose logging: Batch encode successful (shape={getattr(vecs, 'shape', 'unknown')})

        # Convert to list and explicitly delete numpy array to free memory
        result = vecs.tolist()
        del vecs  # Explicit cleanup of large numpy array

        # Update usage statistics
        processing_time = time.time() - start_time
        self._update_usage_stats(processing_time, is_batch=True)

        return result


# ------------------------------------------------------------------
# Helper functions (unchanged)
# ------------------------------------------------------------------


def chunk_content(content: str, max_chunk_size: int = 4096) -> List[tuple]:
    if len(content) <= max_chunk_size:
        return [(content, 0)]

    paragraphs = content.split("\n\n")
    chunks: list[tuple[str, int]] = []
    current = ""
    pos = 0
    for para in paragraphs:
        if len(para) > max_chunk_size:
            for sentence in para.split(". "):
                if len(current) + len(sentence) > max_chunk_size and current:
                    chunks.append((current.strip(), pos))
                    pos += len(current)
                    current = sentence + ". "
                else:
                    current += sentence + ". "
        else:
            if len(current) + len(para) > max_chunk_size and current:
                chunks.append((current.strip(), pos))
                pos += len(current)
                current = para + "\n\n"
            else:
                current += para + "\n\n"
    if current.strip():
        chunks.append((current.strip(), pos))
    return chunks


def preprocess_content(content: str) -> str:
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]
    content = (
        content.replace("```", "").replace("**", "").replace("*", "").replace("#", "")
    )
    return " ".join(line.strip() for line in content.split("\n") if line.strip())


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    try:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        n1 = np.linalg.norm(a)
        n2 = np.linalg.norm(b)
        if n1 == 0 or n2 == 0:
            # Cleanup arrays before returning
            del a, b
            return 0.0
        result = float(np.dot(a, b) / (n1 * n2))
        # Explicit cleanup of numpy arrays
        del a, b
        return result
    except Exception as exc:
        print("Cosine similarity error: %s", exc)
        return 0.0


def batch_cosine_similarity(
    query_vec: List[float], vectors: List[List[float]]
) -> List[float]:
    try:
        q = np.array(query_vec, dtype=np.float32)
        mat = np.array(vectors, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            # Cleanup arrays before returning
            del q, mat
            return [0.0] * len(vectors)
        q /= qn
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1
        mat = mat / norms[:, np.newaxis]
        result = np.dot(mat, q).tolist()
        # Explicit cleanup of numpy arrays
        del q, mat, norms
        return result
    except Exception as exc:
        print("Batch cosine similarity error: %s", exc)
        return [0.0] * len(vectors)


def optimized_batch_cosine_similarity(
    query_vec: List[float], vectors: List[List[float]], batch_size: int = 1000
) -> List[float]:
    """
    Optimized batch cosine similarity with memory-efficient processing.

    Processes large vector sets in batches to prevent memory overflow
    while maintaining high performance through vectorized operations.
    """
    try:
        if not vectors:
            return []

        # Convert query to normalized numpy array once
        q = np.array(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)

        if qn == 0:
            del q
            return [0.0] * len(vectors)

        q = q / qn  # Normalize query vector

        # Process vectors in batches for memory efficiency
        all_similarities = []

        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i : i + batch_size]

            # Convert batch to numpy array
            batch_mat = np.array(batch_vectors, dtype=np.float32)

            # Compute norms and normalize batch
            batch_norms = np.linalg.norm(batch_mat, axis=1)
            # Avoid division by zero
            batch_norms[batch_norms == 0] = 1
            batch_mat = batch_mat / batch_norms[:, np.newaxis]

            # Compute similarities for this batch
            batch_similarities = np.dot(batch_mat, q)
            all_similarities.extend(batch_similarities.tolist())

            # Explicit cleanup for large arrays
            del batch_mat, batch_norms, batch_similarities

        # Cleanup query vector
        del q

        return all_similarities

    except Exception as exc:
        print("Optimized batch cosine similarity error: %s", exc)
        return [0.0] * len(vectors)


def precompute_vector_norms(
    vectors: List[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute normalized vectors and their norms for repeated similarity calculations.

    This is useful when the same set of vectors will be compared against multiple queries.
    """
    try:
        if not vectors:
            return np.array([]), np.array([])

        # Convert to numpy array
        mat = np.array(vectors, dtype=np.float32)

        # Compute norms
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero

        # Normalize vectors
        normalized_mat = mat / norms[:, np.newaxis]

        # Clean up original matrix
        del mat

        return normalized_mat, norms

    except Exception as exc:
        print("Vector normalization error: %s", exc)
        return np.array([]), np.array([])


def fast_similarity_with_precomputed(
    query_vec: List[float], normalized_vectors: np.ndarray, original_norms: np.ndarray
) -> List[float]:
    """
    Fast similarity calculation using precomputed normalized vectors.
    """
    try:
        if normalized_vectors.size == 0:
            return []

        # Normalize query vector
        q = np.array(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)

        if qn == 0:
            del q
            return [0.0] * len(normalized_vectors)

        q = q / qn

        # Fast dot product with precomputed normalized vectors
        similarities = np.dot(normalized_vectors, q)
        result = similarities.tolist()

        # Cleanup
        del q, similarities

        return result

    except Exception as exc:
        print("Fast similarity calculation error: %s", exc)
        return [0.0] * len(normalized_vectors)


class AsyncEmbeddingQueue:
    """
    Async embedding queue to decouple embedding generation from write operations.

    This allows write operations to complete immediately while embeddings are generated
    in the background, enabling fallback to text-only search when embeddings are not ready.
    """

    def __init__(self, max_queue_size: int = 500, worker_count: int = 2):
        self.max_queue_size = max_queue_size
        self.worker_count = worker_count

        # Thread-safe queue for embedding requests
        self._embedding_queue: "queue.Queue[EmbeddingRequest]" = queue.Queue(
            maxsize=max_queue_size
        )
        self._shutdown_event = threading.Event()
        self._workers: List[threading.Thread] = []

        # Statistics and monitoring
        self._stats_lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "queue_full_errors": 0,
            "average_processing_time_ms": 0.0,
            "last_processed": None,
        }

        # Start worker threads
        self._start_workers()
        print(f"AsyncEmbeddingQueue started with {worker_count} workers")

    def _start_workers(self) -> None:
        """Start embedding worker threads."""
        for i in range(self.worker_count):
            worker = threading.Thread(
                target=self._worker_loop, daemon=False, name=f"embedding-worker-{i}"
            )
            worker.start()
            self._workers.append(worker)

    def submit_embedding_request(
        self,
        task_id: str,
        content_chunks: List[Tuple[str, int]],
        callback: Optional[Callable[[str, List[List[float]]], None]] = None,
        priority: int = 1,
    ) -> bool:
        """
        Submit an embedding generation request.

        Args:
            task_id: Task identifier
            content_chunks: List of (chunk_text, position) tuples
            callback: Optional callback function for results
            priority: Request priority (lower = higher priority)

        Returns:
            bool: True if request was queued, False if queue is full
        """
        if self._shutdown_event.is_set():
            print("Embedding queue is shutting down, rejecting request")
            return False

        request = EmbeddingRequest(
            id=str(uuid.uuid4()),
            task_id=task_id,
            content_chunks=content_chunks,
            callback=callback,
            priority=priority,
            timestamp=time.time(),
        )

        try:
            self._embedding_queue.put(
                request, timeout=0.1
            )  # Non-blocking with short timeout

            with self._stats_lock:
                self._stats["total_requests"] += 1

            # Removed verbose logging: Queued embedding request for task {task_id}
            return True

        except queue.Full:
            with self._stats_lock:
                self._stats["queue_full_errors"] += 1

            print(f"Embedding queue is full, skipping embeddings for task {task_id}")
            return False

    def _worker_loop(self) -> None:
        """Worker loop for processing embedding requests."""
        worker_name = threading.current_thread().name
        print(f"Embedding worker {worker_name} started")

        generator = get_default_generator()

        while not self._shutdown_event.is_set():
            try:
                # Get embedding request with timeout
                try:
                    request = self._embedding_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Process the embedding request
                self._process_embedding_request(request, generator)

                # Mark task as done
                self._embedding_queue.task_done()

            except Exception as exc:
                print(f"Embedding worker {worker_name} error: {exc}", exc_info=True)

        print(f"Embedding worker {worker_name} stopped")

    def _process_embedding_request(
        self, request: "EmbeddingRequest", generator: EmbeddingGenerator
    ) -> None:
        """Process a single embedding request."""
        start_time = time.time()

        try:
            # Ensure model is loaded
            if not generator.load_model_now():
                raise RuntimeError("Failed to load embedding model")

            # Generate embeddings for all chunks
            chunk_texts = [chunk_text for chunk_text, _ in request.content_chunks]
            embeddings = generator.generate_embeddings_batch(chunk_texts)

            # Update database with embeddings
            self._store_embeddings(request.task_id, request.content_chunks, embeddings)

            # Call callback if provided
            if request.callback:
                try:
                    request.callback(request.task_id, embeddings)
                except Exception as callback_exc:
                    print(
                        f"Embedding callback failed for task {request.task_id}: {callback_exc}"
                    )

            # Update success statistics
            processing_time_ms = (time.time() - start_time) * 1000

            with self._stats_lock:
                self._stats["successful_embeddings"] += 1
                self._stats["last_processed"] = time.time()

                # Update rolling average processing time
                current_avg = self._stats["average_processing_time_ms"]
                successful_count = self._stats["successful_embeddings"]
                self._stats["average_processing_time_ms"] = (
                    current_avg * (successful_count - 1) + processing_time_ms
                ) / successful_count

            print(
                f"✅ Embeddings generated for {request.task_id} ({processing_time_ms:.0f}ms)"
            )

        except Exception as exc:
            with self._stats_lock:
                self._stats["failed_embeddings"] += 1

            print(f"❌ Failed to generate embeddings for task {request.task_id}: {exc}")

    def _store_embeddings(
        self,
        task_id: str,
        content_chunks: List[Tuple[str, int]],
        embeddings: List[List[float]],
    ) -> None:
        """Store generated embeddings in the database."""
        from core.models import EmbeddingRecord, Embedding
        from infrastructure.database import get_session

        try:
            with get_session() as session:
                # Delete existing embeddings for this task
                session.query(Embedding).filter_by(task_id=task_id).delete()

                # Add new embeddings
                for (chunk_text, chunk_pos), vector in zip(content_chunks, embeddings):
                    embedding_record = EmbeddingRecord.from_vector(
                        task_id=task_id,
                        vector=vector,
                        chunk_id=chunk_pos,
                        position=0,
                    )

                    embedding = Embedding(
                        task_id=embedding_record.task_id,
                        chunk_id=embedding_record.chunk_id,
                        position=embedding_record.position,
                        vector=embedding_record.vector,
                    )
                    session.add(embedding)

                session.commit()
                # Embeddings stored successfully

        except Exception as exc:
            print(f"❌ Failed to store embeddings for task {task_id}: {exc}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        with self._stats_lock:
            return {
                **self._stats,
                "queue_size": self._embedding_queue.qsize(),
                "workers_alive": sum(1 for w in self._workers if w.is_alive()),
                "is_shutdown": self._shutdown_event.is_set(),
            }

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Shutdown the embedding queue gracefully."""
        print("Shutting down AsyncEmbeddingQueue...")

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for queue to empty
        shutdown_start = time.time()
        try:
            while (
                not self._embedding_queue.empty()
                and (time.time() - shutdown_start) < timeout
            ):
                time.sleep(0.1)
        except Exception as exc:
            print(f"Error during embedding queue drain: {exc}")

        # Wait for workers to finish
        remaining_timeout = max(0, timeout - (time.time() - shutdown_start))
        for worker in self._workers:
            worker.join(timeout=remaining_timeout / len(self._workers))

        # Check success
        success = all(not worker.is_alive() for worker in self._workers)

        if success:
            print("AsyncEmbeddingQueue shutdown completed successfully")
        else:
            print("AsyncEmbeddingQueue shutdown timed out")

        return success


@dataclass
class EmbeddingRequest:
    """Represents an embedding generation request."""

    id: str
    task_id: str
    content_chunks: List[Tuple[str, int]]  # (chunk_text, position)
    callback: Optional[Callable[[str, List[List[float]]], None]]
    priority: int
    timestamp: float


# Global embedding queue instance
_global_embedding_queue: Optional[AsyncEmbeddingQueue] = None


def get_embedding_queue() -> AsyncEmbeddingQueue:
    """Get or create the global embedding queue instance."""
    global _global_embedding_queue

    if _global_embedding_queue is None:
        _global_embedding_queue = AsyncEmbeddingQueue()

    return _global_embedding_queue


def shutdown_embedding_queue() -> bool:
    """Shutdown the global embedding queue."""
    global _global_embedding_queue

    if _global_embedding_queue is not None:
        success = _global_embedding_queue.shutdown()
        _global_embedding_queue = None
        return success

    return True


@lru_cache(maxsize=1)
def get_default_generator() -> EmbeddingGenerator:
    return EmbeddingGenerator()


@lru_cache(maxsize=1024)
def generate_embedding(text: str) -> List[float]:
    return get_default_generator().generate_embedding(text)
