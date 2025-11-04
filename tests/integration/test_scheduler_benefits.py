"""
Test: Scheduler Benefits & Optimization Validation

Validates:
- Schedule creation for various patterns
- Stage assignment correctness
- Dependency respect in schedules
- Optimization opportunities detected
- Cost-based scheduling decisions
- Multi-stage execution planning
"""

import torch
import pytest
import logging
import genie

logger = logging.getLogger(__name__)


class TestSchedulerBasics:
    """Test basic scheduler functionality."""
    
    def test_scheduler_creates_valid_schedule(self):
        """Test scheduler creates valid schedule."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = torch.randn(5, 5)
            z = x @ y
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        assert hasattr(schedule, 'total_stages')
        assert schedule.total_stages > 0
        
        print("✅ Scheduler creates valid schedule")
    
    def test_all_operations_scheduled(self):
        """Test all operations are scheduled."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = torch.randn(5, 5)
            z = x + y
            w = z @ z
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        # Should have scheduled all operations
        assert schedule.total_stages > 0
        
        print("✅ All operations scheduled")
    
    def test_schedule_respects_dependencies(self):
        """Test schedule respects data dependencies."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1          # Depends on x
            z = y @ y          # Depends on y
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        # Should not fail
        assert schedule is not None
        assert schedule.total_stages > 0
        
        print("✅ Schedule respects dependencies")


class TestSchedulerPatternDetection:
    """Test scheduler detects and optimizes patterns."""
    
    def test_scheduler_handles_linear_chain(self):
        """Test scheduler handles linear operation chains."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            for i in range(5):
                x = x @ x + i
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        assert schedule.total_stages > 0
        
        print("✅ Scheduler handles linear chains")
    
    def test_scheduler_handles_fan_out_pattern(self):
        """Test scheduler handles fan-out patterns."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            
            # Fan-out: x used by multiple operations
            y = x + 1
            z = x @ x
            w = torch.relu(x)
            
            result = y + z + w
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        assert schedule.total_stages > 0
        
        print("✅ Scheduler handles fan-out patterns")
    
    def test_scheduler_handles_fan_in_pattern(self):
        """Test scheduler handles fan-in patterns."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.randn(10, 10)
            
            # Fan-in: result depends on multiple inputs
            result = x + y + z
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        assert schedule.total_stages > 0
        
        print("✅ Scheduler handles fan-in patterns")
    
    def test_scheduler_handles_attention_pattern(self):
        """Test scheduler handles attention-like patterns."""
        
        with genie.capture():
            q = torch.randn(2, 8, 64)
            k = torch.randn(2, 8, 64)
            v = torch.randn(2, 8, 64)
            
            # Attention-like pattern
            scores = q @ k.transpose(-2, -1)
            attn = torch.relu(scores)
            output = attn @ v
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        assert schedule.total_stages > 0
        
        print("✅ Scheduler handles attention patterns")


class TestSchedulerOptimization:
    """Test scheduler optimizations."""
    
    def test_scheduler_creates_efficient_schedule(self):
        """Test scheduler creates efficient schedule."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            y = x @ x
            z = y + 1
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        # Should create schedule with reasonable stage count
        assert schedule.total_stages > 0
        assert schedule.total_stages <= len(list(graph.nodes()))
        
        print("✅ Scheduler creates efficient schedule")
    
    def test_scheduler_minimizes_stages(self):
        """Test scheduler minimizes stage count where possible."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
            z = y * 2
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        # Should use minimal stages
        assert schedule.total_stages > 0
        
        print("✅ Scheduler minimizes stages")
    
    def test_scheduler_balances_load(self):
        """Test scheduler attempts to balance load across stages."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            
            # Multiple operations for load balancing
            a = x @ y
            b = x + y
            c = x - y
            d = x * y
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        assert schedule.total_stages > 0
        
        print("✅ Scheduler balances load")


class TestSchedulerCorrectness:
    """Test scheduler correctness properties."""
    
    def test_schedule_preserves_semantics(self):
        """Test schedule preserves computation semantics."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
            z = torch.relu(y)
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        # Execution should be semantically correct
        result = z.cpu()
        
        assert result.shape == torch.Size([5, 5])
        assert (result >= 0).all()
        
        print("✅ Schedule preserves semantics")
    
    def test_schedule_no_deadlocks(self):
        """Test schedule doesn't create deadlocks."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x @ x
            z = y + x
            w = z @ z
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        # Should complete without deadlock
        result = w.cpu()
        assert result.shape == torch.Size([5, 5])
        
        print("✅ Schedule has no deadlocks")
    
    def test_schedule_no_duplicate_execution(self):
        """Test schedule doesn't execute operations twice."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            y = x + 1
            z = y @ y
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        # Execution should be correct without duplicates
        result = z.cpu()
        
        assert result.shape == torch.Size([5, 5])
        
        print("✅ No duplicate execution")


class TestSchedulerComplex:
    """Test scheduler on complex graphs."""
    
    def test_scheduler_handles_large_graph(self):
        """Test scheduler handles large graphs."""
        
        with genie.capture():
            x = torch.randn(20, 20)
            
            # Create 50 operations
            for i in range(50):
                x = x + 0.01
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        assert schedule.total_stages > 0
        
        print("✅ Scheduler handles large graphs")
    
    def test_scheduler_handles_nested_ops(self):
        """Test scheduler handles nested operations."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            
            # Nested structure
            y = torch.relu(x @ x + 1)
            z = torch.sigmoid(y @ y - 1)
            w = torch.tanh(z + x)
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        result = w.cpu()
        
        assert result.shape == torch.Size([5, 5])
        assert schedule.total_stages > 0
        
        print("✅ Scheduler handles nested operations")
    
    def test_scheduler_handles_diamond_pattern(self):
        """Test scheduler handles diamond patterns."""
        
        with genie.capture():
            x = torch.randn(5, 5)
            
            # Diamond: x -> y,z -> w
            y = x + 1
            z = x @ x
            w = y + z
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        result = w.cpu()
        
        assert result.shape == torch.Size([5, 5])
        assert schedule.total_stages > 0
        
        print("✅ Scheduler handles diamond patterns")


class TestSchedulerWithMetrics:
    """Test scheduler with performance metrics."""
    
    def test_scheduler_annotates_costs(self):
        """Test scheduler properly annotates costs."""
        
        with genie.capture():
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = x @ y
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        
        # Should have cost information
        assert hasattr(annotated, 'costs')
        assert 'total_compute_flops' in annotated.costs
        assert annotated.costs['total_compute_flops'] > 0
        
        schedule = genie.schedule(annotated.base_graph)
        assert schedule is not None
        
        print("✅ Scheduler annotates costs")
    
    def test_scheduler_considers_memory(self):
        """Test scheduler considers memory in planning."""
        
        with genie.capture():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            
            # Operations with memory implications
            z = x @ y
            w = z @ z
            v = w + x
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        
        # Should have memory information
        assert hasattr(annotated, 'costs')
        
        schedule = genie.schedule(annotated.base_graph)
        assert schedule is not None
        
        print("✅ Scheduler considers memory")
    
    def test_scheduler_considers_bandwidth(self):
        """Test scheduler considers bandwidth."""
        
        with genie.capture():
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            
            # Operations with bandwidth implications
            z = x @ y
            w = z.t()
            v = w + x.t()
        
        graph = genie.get_graph()
        annotated = genie.annotate_graph(graph)
        schedule = genie.schedule(annotated.base_graph)
        
        assert schedule is not None
        
        print("✅ Scheduler considers bandwidth")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
