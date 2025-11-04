"""
Test: Thread safety of concurrent operations

CRITICAL: Your code uses threading extensively - must test!

Validates:
- Capture contexts are thread-local and isolated
- Factory interceptor thread-local state works
- Concurrent graph building doesn't corrupt data
- Analysis cache is thread-safe
"""

import threading
import torch
import pytest
import time
import logging
import genie

logger = logging.getLogger(__name__)


class TestThreadSafety:
    """Test thread safety of concurrent operations."""
    
    def test_concurrent_captures_isolated(self):
        """Test multiple threads can capture independently without interference."""
        
        results = {}
        errors = []
        
        def worker(thread_id):
            try:
                with genie.capture():
                    # Each thread creates different operations
                    x = torch.randn(10, 10)
                    y = x + thread_id
                    z = y @ y
                
                # Materialize
                result = z.cpu()
                results[thread_id] = result.shape
            except Exception as e:
                errors.append((thread_id, e))
        
        # Run 10 concurrent threads
        threads = [threading.Thread(target=worker, args=(i,)) 
                   for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # HARD ASSERTIONS
        assert len(errors) == 0, \
            f"Thread failures: {errors}"
        assert len(results) == 10, \
            f"Not all threads completed: {len(results)}/10"
        
        # All should have correct shape
        for thread_id, shape in results.items():
            assert shape == torch.Size([10, 10]), \
                f"Thread {thread_id} wrong shape: {shape}"
        
        print(f"✅ {len(results)} concurrent captures succeeded")
    
    def test_capture_context_thread_local(self):
        """Test capture context is truly thread-local."""
        
        results = {'inside': [], 'outside': []}
        
        def thread1():
            with genie.capture():
                time.sleep(0.1)  # Ensure overlap with thread2
                x = torch.randn(5, 5)
                results['inside'].append(isinstance(x, genie.LazyTensor))
        
        def thread2():
            time.sleep(0.05)  # Start in middle of thread1's capture
            x = torch.randn(5, 5)
            results['outside'].append(not isinstance(x, genie.LazyTensor))
        
        t1 = threading.Thread(target=thread1)
        t2 = threading.Thread(target=thread2)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # HARD ASSERTIONS
        assert len(results['inside']) == 1 and results['inside'][0], \
            "Thread 1 capture context not working!"
        assert len(results['outside']) == 1 and results['outside'][0], \
            "Thread 2 affected by thread 1 capture!"
        
        print("✅ Capture contexts properly isolated between threads")
    
    def test_factory_interceptor_thread_local(self):
        """Test factory interceptor thread-local state."""
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                # Create LazyTensor (triggers factory interceptor)
                with genie.capture():
                    x = torch.randn(10, 10)
                    y = x @ x
                
                result = y.cpu()
                results.append((thread_id, result.shape))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = [threading.Thread(target=worker, args=(i,)) 
                   for i in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # HARD ASSERTIONS
        assert len(errors) == 0, \
            f"Factory interceptor thread safety failed: {errors}"
        assert len(results) == 20, \
            f"Not all threads completed: {len(results)}/20"
        
        print(f"✅ Factory interceptor thread-safe ({len(results)} threads)")
    
    def test_graph_builder_concurrent_access(self):
        """Test graph builder handles concurrent access safely."""
        
        graphs = []
        errors = []
        
        def worker(thread_id):
            try:
                with genie.capture():
                    x = torch.randn(5, 5)
                    for _ in range(10):
                        x = x + 1
                
                graph = genie.get_graph()
                graphs.append((thread_id, len(list(graph.nodes()))))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = [threading.Thread(target=worker, args=(i,)) 
                   for i in range(15)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # HARD ASSERTIONS
        assert len(errors) == 0, \
            f"Graph builder concurrent access failed: {errors}"
        assert len(graphs) == 15, \
            f"Not all graphs created: {len(graphs)}/15"
        
        print(f"✅ Graph builder thread-safe ({len(graphs)} graphs)")
    
    def test_semantic_analysis_concurrent(self):
        """Test semantic analysis is thread-safe."""
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                with genie.capture():
                    x = torch.randn(20, 30)
                    y = torch.randn(30, 40)
                    z = x @ y
                
                graph = genie.get_graph()
                annotated = genie.annotate_graph(graph)
                
                results.append((thread_id, annotated.costs['total_compute_flops']))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = [threading.Thread(target=worker, args=(i,)) 
                   for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # HARD ASSERTIONS
        assert len(errors) == 0, \
            f"Semantic analysis not thread-safe: {errors}"
        assert len(results) == 10, \
            f"Not all analyses completed: {len(results)}/10"
        
        # All should have non-zero costs
        for thread_id, flops in results:
            assert flops > 0, \
                f"Thread {thread_id} got zero FLOPs!"
        
        print(f"✅ Semantic analysis thread-safe ({len(results)} analyses)")
    
    def test_race_condition_stress(self):
        """Stress test for race conditions."""
        
        errors = []
        success_count = [0]
        lock = threading.Lock()
        
        def worker():
            try:
                # Rapid capture/release cycles
                for _ in range(50):
                    with genie.capture():
                        x = torch.randn(3, 3)
                        y = x + 1
                    result = y.cpu()
                
                with lock:
                    success_count[0] += 1
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # HARD ASSERTIONS
        assert len(errors) == 0, \
            f"Race conditions detected: {errors[:5]}"  # Show first 5
        assert success_count[0] == 10, \
            f"Only {success_count[0]}/10 threads succeeded!"
        
        print(f"✅ No race conditions in stress test (500 captures)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
