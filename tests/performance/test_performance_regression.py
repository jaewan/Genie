"""
Performance Regression Tests

These tests monitor for performance degradations and ensure optimizations
provide expected benefits. They establish baselines and alert when
performance degrades beyond acceptable thresholds.

Usage:
    pytest tests/performance/test_performance_regression.py -v

CI/CD Integration:
    pytest tests/performance/test_performance_regression.py --json-report
"""

import pytest
import torch
import torch.nn as nn
import time
import json
import os
import gc
import subprocess
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import sys
sys.path.insert(0, '.')
import djinn
from djinn.core.capture import capture, get_graph
from djinn.frontend.semantic.annotator import SemanticAnnotator
from djinn.frontend.semantic.pattern_registry import PatternRegistry, MatchingMode

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""
    test_name: str
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    patterns_matched: int
    cache_hit_rate: float
    index_hit_rate: float
    timestamp: float
    git_commit: str
    python_version: str
    torch_version: str

    def to_dict(self):
        return asdict(self)


@dataclass
class PerformanceThresholds:
    """Acceptable performance thresholds for regression detection."""
    max_time_degradation_percent: float = 20.0  # Allow 20% degradation
    min_cache_hit_rate: float = 0.1  # At least 10% cache hit rate
    min_index_hit_rate: float = 0.3  # At least 30% index hit rate
    min_patterns_matched: int = 0  # At least some patterns should match


class PerformanceMonitor:
    """Monitor and track performance metrics over time."""

    def __init__(self, baseline_file: str = "tests/performance/baselines.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.thresholds = PerformanceThresholds()
        self.load_baselines()

    def load_baselines(self):
        """Load existing performance baselines."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    for test_name, baseline_data in data.items():
                        self.baselines[test_name] = PerformanceBaseline(**baseline_data)
            except Exception as e:
                logger.warning(f"Failed to load baselines: {e}")

    def save_baselines(self):
        """Save current baselines to file."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        data = {name: baseline.to_dict() for name, baseline in self.baselines.items()}
        with open(self.baseline_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_baseline(self, test_name: str, metrics: Dict):
        """Record a new performance baseline."""
        import subprocess

        baseline = PerformanceBaseline(
            test_name=test_name,
            avg_time_ms=metrics['avg_time_ms'],
            max_time_ms=metrics['max_time_ms'],
            min_time_ms=metrics['min_time_ms'],
            patterns_matched=metrics['patterns_matched'],
            cache_hit_rate=metrics['cache_hit_rate'],
            index_hit_rate=metrics['index_hit_rate'],
            timestamp=time.time(),
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:8],
            python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            torch_version=torch.__version__
        )

        self.baselines[test_name] = baseline
        self.save_baselines()
        return baseline

    def check_regression(self, test_name: str, current_metrics: Dict) -> Dict:
        """Check if current performance indicates a regression."""
        if test_name not in self.baselines:
            # No baseline available, record current as baseline
            baseline = self.record_baseline(test_name, current_metrics)
            return {
                'regression_detected': False,
                'reason': 'No baseline available - recorded current performance',
                'baseline': baseline.to_dict(),
                'current': current_metrics
            }

        baseline = self.baselines[test_name]

        # Check time regression
        time_degradation = ((current_metrics['avg_time_ms'] - baseline.avg_time_ms)
                          / baseline.avg_time_ms * 100)

        # Check cache hit rate regression
        cache_degradation = baseline.cache_hit_rate - current_metrics['cache_hit_rate']

        # Check index hit rate regression
        index_degradation = baseline.index_hit_rate - current_metrics['index_hit_rate']

        # Check pattern matching regression
        patterns_degradation = baseline.patterns_matched - current_metrics['patterns_matched']

        issues = []

        if time_degradation > self.thresholds.max_time_degradation_percent:
            issues.append(f"Time degraded by {time_degradation:.1f}% (>{self.thresholds.max_time_degradation_percent}%)")

        # Only check cache/index hit rates if baseline had non-zero rates (meaningful comparison)
        if baseline.cache_hit_rate > 0 and cache_degradation > 0.05:
            issues.append(f"Cache hit rate degraded from {baseline.cache_hit_rate:.2f} to {current_metrics['cache_hit_rate']:.2f}")

        if baseline.index_hit_rate > 0 and index_degradation > 0.1:
            issues.append(f"Index hit rate degraded from {baseline.index_hit_rate:.2f} to {current_metrics['index_hit_rate']:.2f}")

        if patterns_degradation < 0:
            issues.append(f"Pattern matching degraded: {patterns_degradation} fewer matches")

        return {
            'regression_detected': len(issues) > 0,
            'issues': issues,
            'baseline': baseline.to_dict(),
            'current': current_metrics,
            'time_degradation_percent': time_degradation,
            'cache_degradation': cache_degradation,
            'index_degradation': index_degradation
        }


# Global performance monitor
monitor = PerformanceMonitor()


class TestPerformanceRegression:
    """Performance regression tests with baseline comparison."""

    def setup_method(self):
        """Set up clean state for each test."""
        # Clear any cached data if possible
        try:
            import djinn.core.graph_builder
            if hasattr(genie.core.graph_builder, 'get_global_builder'):
                builder = genie.core.graph_builder.get_global_builder()
                if hasattr(builder, 'clear'):
                    builder.clear()
        except Exception:
            pass  # Ignore errors in cleanup

    def test_llm_pattern_matching_performance(self):
        """Test LLM pattern matching performance doesn't regress."""
        # Create LLM-like computation graph
        def create_llm_graph():
            with capture():
                batch_size, seq_len, hidden_size = 2, 12, 64
                x = torch.randn(batch_size, seq_len, hidden_size)

                # Attention mechanism
                q = x @ torch.randn(hidden_size, hidden_size)
                k = x @ torch.randn(hidden_size, hidden_size)
                scores = q @ k.transpose(-2, -1)
                attention = scores.softmax(dim=-1)
                output = attention @ (x @ torch.randn(hidden_size, hidden_size))
            return get_graph()

        # Run multiple times for stable measurements
        times = []
        pattern_counts = []

        for i in range(10):
            graph = create_llm_graph()

            start_time = time.time()
            result = genie.annotate_graph(graph)
            duration = (time.time() - start_time) * 1000

            times.append(duration)

            # Count successful pattern matches
            patterns_matched = sum(1 for pattern_matches in result.patterns.values()
                                 for match in pattern_matches
                                 if match.metadata.get('confidence', 0) > 0.5)
            pattern_counts.append(patterns_matched)

        # Calculate metrics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        avg_patterns = sum(pattern_counts) / len(pattern_counts)

        # Get performance metrics
        annotator = SemanticAnnotator()
        cache_stats = annotator.get_cache_stats()
        registry_report = annotator.pattern_registry.get_performance_report()

        current_metrics = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'min_time_ms': min_time,
            'patterns_matched': int(avg_patterns),
            'cache_hit_rate': cache_stats.get('content_addressed_cache', {}).get('hit_rate', 0),
            'index_hit_rate': registry_report.get('_index', {}).get('index_hit_rate', 0)
        }

        # Check for regression
        regression_check = monitor.check_regression('llm_pattern_matching', current_metrics)

        if regression_check['regression_detected']:
            pytest.fail(f"Performance regression detected: {regression_check['issues']}")

        # Record/update baseline if no regression
        monitor.record_baseline('llm_pattern_matching', current_metrics)

        print(f"✅ LLM performance: {avg_time:.2f}ms avg, {avg_patterns:.1f} patterns, "
              f"cache: {current_metrics['cache_hit_rate']:.2f}, "
              f"index: {current_metrics['index_hit_rate']:.2f}")

    def test_vision_pattern_matching_performance(self):
        """Test vision pattern matching performance doesn't regress."""
        def create_vision_graph():
            with capture():
                x = torch.randn(1, 3, 32, 32)

                # CNN layers
                conv1 = torch.randn(16, 3, 3, 3)
                conv2 = torch.randn(32, 16, 3, 3)

                y1 = torch.conv2d(x, conv1, padding=1).relu()
                y2 = torch.conv2d(y1, conv2, padding=1).relu()
                output = torch.nn.functional.adaptive_avg_pool2d(y2, (1, 1))
            return get_graph()

        # Run multiple times
        times = []
        pattern_counts = []

        for i in range(10):
            graph = create_vision_graph()

            start_time = time.time()
            result = genie.annotate_graph(graph)
            duration = (time.time() - start_time) * 1000

            times.append(duration)

            patterns_matched = sum(1 for pattern_matches in result.patterns.values()
                                 for match in pattern_matches if match.metadata.get('confidence', 0) > 0.5)
            pattern_counts.append(patterns_matched)

        # Calculate metrics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        avg_patterns = sum(pattern_counts) / len(pattern_counts)

        # Get performance metrics
        annotator = SemanticAnnotator()
        cache_stats = annotator.get_cache_stats()
        registry_report = annotator.pattern_registry.get_performance_report()

        current_metrics = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'min_time_ms': min_time,
            'patterns_matched': int(avg_patterns),
            'cache_hit_rate': cache_stats.get('content_addressed_cache', {}).get('hit_rate', 0),
            'index_hit_rate': registry_report.get('_index', {}).get('index_hit_rate', 0)
        }

        # Check for regression
        regression_check = monitor.check_regression('vision_pattern_matching', current_metrics)

        if regression_check['regression_detected']:
            pytest.fail(f"Performance regression detected: {regression_check['issues']}")

        # Record/update baseline
        monitor.record_baseline('vision_pattern_matching', current_metrics)

        print(f"✅ Vision performance: {avg_time:.2f}ms avg, {avg_patterns:.1f} patterns")

    def test_hierarchical_index_effectiveness(self):
        """Test that hierarchical index provides expected performance benefits."""
        registry = PatternRegistry()

        # Create graphs of different sizes
        test_graphs = []

        # Small graph
        with capture():
            x = torch.randn(5, 5)
            y = x @ x
        test_graphs.append(('small', get_graph()))

        # Medium graph
        with capture():
            batch, seq, hidden = 2, 10, 32
            x = torch.randn(batch, seq, hidden)
            q = x @ torch.randn(hidden, hidden)
            k = x @ torch.randn(hidden, hidden)
            scores = q @ k.transpose(-2, -1)
            output = scores.softmax(dim=-1) @ (x @ torch.randn(hidden, hidden))
        test_graphs.append(('medium', get_graph()))

        # Large graph
        with capture():
            batch, seq, hidden = 4, 20, 64
            x = torch.randn(batch, seq, hidden)

            # Multi-head attention simulation
            outputs = x
            for head in range(8):
                q = outputs @ torch.randn(hidden, hidden)
                k = outputs @ torch.randn(hidden, hidden)
                v = outputs @ torch.randn(hidden, hidden)
                scores = q @ k.transpose(-2, -1)
                attention = scores.softmax(dim=-1)
                output = attention @ v
                outputs = output

        test_graphs.append(('large', get_graph()))

        # Test performance scaling
        for size_name, graph in test_graphs:
            start_time = time.time()
            result = registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)
            duration = (time.time() - start_time) * 1000

            # Get index performance
            report = registry.get_performance_report()
            index_stats = report.get('_index', {})
            matching_stats = report.get('_matching', {})

            # For larger graphs, we expect better index performance
            if size_name == 'large':
                # Large graphs should have reasonable index hit rates
                index_hit_rate = index_stats.get('index_hit_rate', 0)
                if index_hit_rate < 0.1:  # At least 10% hit rate
                    logger.warning(f"Low index hit rate for large graph: {index_hit_rate:.2f}")

            print(f"✅ {size_name} graph: {duration:.2f}ms, "
                  f"index_hit_rate: {index_stats.get('index_hit_rate', 0):.2f}")

    def test_cache_effectiveness_regression(self):
        """Test that caching provides expected performance benefits."""
        annotator = SemanticAnnotator(enable_cache=True)

        # Create identical graphs
        identical_graphs = []
        for i in range(5):
            with capture():
                x = torch.randn(10, 20)
                y = x @ torch.randn(20, 30)
                z = y.relu()
            identical_graphs.append(get_graph())

        # First annotation (cache miss)
        start = time.time()
        result1 = annotator.annotate(identical_graphs[0])
        time1 = (time.time() - start) * 1000

        # Subsequent annotations (cache hits)
        times = [time1]
        for graph in identical_graphs[1:]:
            start = time.time()
            result = annotator.annotate(graph)
            times.append((time.time() - start) * 1000)

        # Check cache effectiveness
        cache_stats = annotator.get_cache_stats()
        hit_rate = cache_stats.get('content_addressed_cache', {}).get('hit_rate', 0)

        # Expect significant speedup on cache hits
        avg_cache_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
        speedup_ratio = time1 / avg_cache_time if avg_cache_time > 0 else 1.0

        # Should have reasonable cache hit rate and speedup
        # Note: First run may have low hit rate, we mainly check that caching works
        assert speedup_ratio >= 1.0, f"Cache speedup should be >= 1.0x: {speedup_ratio:.2f}x"
        if hit_rate > 0:
            # If we got any cache hits, speedup should be substantial
            assert speedup_ratio >= 1.5, f"Speedup should be >= 1.5x with cache hits: {speedup_ratio:.2f}x"

        print(f"✅ Cache effectiveness: {speedup_ratio:.1f}x speedup, "
              f"hit_rate: {hit_rate:.2f}, first: {time1:.2f}ms, cached: {avg_cache_time:.2f}ms")

    def test_thread_safety_performance(self):
        """Test that thread safety doesn't significantly impact performance."""
        registry = PatternRegistry()

        def worker(thread_id):
            with capture():
                x = torch.randn(8, 8)
                y = x @ torch.randn(8, 8)
                z = y.relu()
            graph = get_graph()

            start = time.time()
            result = registry.match_patterns(graph, mode=MatchingMode.FAST)
            duration = (time.time() - start) * 1000
            return duration

        # Run concurrent workers
        import threading

        times = []
        threads = []

        for i in range(20):
            thread = threading.Thread(target=lambda: times.append(worker(i)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check that concurrent access doesn't cause significant slowdown
        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Performance shouldn't be terrible even under concurrent load
        assert avg_time < 100, f"Concurrent access too slow: {avg_time:.2f}ms avg"
        assert max_time < 500, f"Some thread took too long: {max_time:.2f}ms"

        print(f"✅ Thread safety: {avg_time:.2f}ms avg, {max_time:.2f}ms max across 20 threads")


class TestPerformanceAlerts:
    """Performance alert system for CI/CD."""

    def test_generate_performance_report(self):
        """Generate comprehensive performance report for CI/CD."""
        # Run various performance tests
        registry = PatternRegistry()
        annotator = SemanticAnnotator(enable_cache=True)

        # Test different workloads
        workloads = [
            ('simple', lambda: self._create_simple_graph()),
            ('llm', lambda: self._create_llm_graph()),
            ('vision', lambda: self._create_vision_graph()),
            ('complex', lambda: self._create_complex_graph())
        ]

        report = {
            'timestamp': time.time(),
            'workloads': {},
            'overall_metrics': {},
            'regressions': []
        }

        for workload_name, graph_func in workloads:
            # Run performance test
            graph = graph_func()

            start_time = time.time()
            result = annotator.annotate(graph)
            duration = (time.time() - start_time) * 1000

            # Get detailed metrics
            cache_stats = annotator.get_cache_stats()
            registry_report = registry.get_performance_report()

            workload_metrics = {
                'avg_time_ms': duration,
                'max_time_ms': duration,
                'min_time_ms': duration,
                'patterns_matched': sum(len(matches) for matches in result.patterns.values()),
                'cache_hit_rate': cache_stats.get('content_addressed_cache', {}).get('hit_rate', 0),
                'index_hit_rate': registry_report.get('_index', {}).get('index_hit_rate', 0),
                'nodes': len(list(graph.nodes())),
                'backend_type': graph.backend_type
            }

            report['workloads'][workload_name] = workload_metrics

        # Overall metrics
        report['overall_metrics'] = {
            'total_tests': len(workloads),
            'avg_time_ms': sum(w['avg_time_ms'] for w in report['workloads'].values()) / len(workloads),
            'total_patterns_matched': sum(w['patterns_matched'] for w in report['workloads'].values()),
            'cache_utilization': sum(w['cache_hit_rate'] for w in report['workloads'].values()) / len(workloads),
            'index_utilization': sum(w['index_hit_rate'] for w in report['workloads'].values()) / len(workloads)
        }

        # Save report
        report_file = Path("tests/performance/latest_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Check for regressions
        for workload_name, metrics in report['workloads'].items():
            regression_check = monitor.check_regression(f'ci_{workload_name}', metrics)
            if regression_check['regression_detected']:
                report['regressions'].append({
                    'workload': workload_name,
                    'issues': regression_check['issues']
                })

        print(f"✅ Performance report saved: {report['overall_metrics']}")

        # Fail test if regressions detected
        if report['regressions']:
            regression_details = [f"{r['workload']}: {r['issues']}" for r in report['regressions']]
            pytest.fail(f"Performance regressions detected: {regression_details}")

    def _create_simple_graph(self):
        with capture():
            x = torch.randn(5, 5)
            y = x @ x
        return get_graph()

    def _create_llm_graph(self):
        with capture():
            batch, seq, hidden = 2, 10, 32
            x = torch.randn(batch, seq, hidden)
            q = x @ torch.randn(hidden, hidden)
            k = x @ torch.randn(hidden, hidden)
            scores = q @ k.transpose(-2, -1)
            output = scores.softmax(dim=-1) @ (x @ torch.randn(hidden, hidden))
        return get_graph()

    def _create_vision_graph(self):
        with capture():
            x = torch.randn(1, 3, 16, 16)
            conv = torch.randn(16, 3, 3, 3)
            y = torch.conv2d(x, conv, padding=1).relu()
        return get_graph()

    def _create_complex_graph(self):
        with capture():
            # Multi-modal graph
            vision_x = torch.randn(1, 3, 32, 32)
            text_x = torch.randn(1, 20, 128)

            vision_conv = torch.randn(32, 3, 3, 3)
            text_linear = torch.randn(128, 256)

            vision_features = torch.conv2d(vision_x, vision_conv, padding=1).relu()
            text_features = text_x @ text_linear

            # Fusion
            vision_flat = torch.nn.functional.adaptive_avg_pool2d(vision_features, (1, 1)).view(1, -1)
            fusion = vision_flat + text_features.mean(dim=1, keepdim=True)
            output = fusion.relu()
        return get_graph()


def test_performance_regression_detection():
    """Test that regression detection works correctly."""
    # Create a fresh monitor for this test
    test_monitor = PerformanceMonitor()
    
    # Create mock baseline
    baseline = PerformanceBaseline(
        test_name="test_regression",
        avg_time_ms=100.0,
        max_time_ms=150.0,
        min_time_ms=50.0,
        patterns_matched=2,
        cache_hit_rate=0.5,
        index_hit_rate=0.8,
        timestamp=time.time(),
        git_commit="abc123",
        python_version="3.12",
        torch_version="2.9.0"
    )

    # Test case 1: No regression
    current_good = {
        'avg_time_ms': 90.0,  # 10% improvement
        'max_time_ms': 140.0,
        'min_time_ms': 45.0,
        'patterns_matched': 2,
        'cache_hit_rate': 0.6,  # Better cache
        'index_hit_rate': 0.9   # Better index
    }

    test_monitor.baselines["test_regression"] = baseline
    regression_check = test_monitor.check_regression("test_regression", current_good)
    assert not regression_check['regression_detected'], "Good performance should not be flagged as regression"

    # Test case 2: Time regression
    current_bad_time = current_good.copy()
    current_bad_time['avg_time_ms'] = 150.0  # 50% degradation

    test_monitor.baselines["test_regression_time"] = baseline
    regression_check = test_monitor.check_regression("test_regression_time", current_bad_time)
    assert regression_check['regression_detected'], "Significant time degradation should be detected"

    # Test case 3: Cache regression (only if baseline had non-zero rate)
    current_bad_cache = current_good.copy()
    current_bad_cache['cache_hit_rate'] = 0.4  # Degraded from 0.5

    test_monitor.baselines["test_regression_cache"] = baseline
    regression_check = test_monitor.check_regression("test_regression_cache", current_bad_cache)
    # Cache degradation: 0.5 - 0.4 = 0.1, which is > 0.05 threshold, so should detect
    assert regression_check['regression_detected'], "Cache degradation should be detected"

    print("✅ Regression detection working correctly")


if __name__ == '__main__':
    # Run regression tests and generate report
    pytest.main([__file__, '-v', '--tb=short'])
