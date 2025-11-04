"""
Baseline Performance Measurement Tests

Establish TCP baseline performance for all supported operations.
This provides the foundation for comparing with DPDK optimizations in Phase 2.

Tests:
- Operation latency percentiles (p50, p95, p99)
- Throughput measurement under sustained load
- Component timing breakdown with profiling
- Memory usage patterns
- Network bandwidth utilization
"""

import pytest
import pytest_asyncio
import asyncio
import torch
import logging
import time
import json
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def profiled_coordinator():
    """Create coordinator with profiling enabled for detailed measurements."""
    from genie.core.coordinator import GenieCoordinator, CoordinatorConfig

    config = CoordinatorConfig(
        node_id='perf-test-client',
        control_port=16555,
        data_port=16556,
        tcp_fallback=True,
        enable_profiling=True  # Enable detailed profiling
    )

    coordinator = GenieCoordinator(config)
    await coordinator.start()

    logger.info("✅ Profiled coordinator started")

    yield coordinator

    # Cleanup
    await coordinator.stop()
    logger.info("✅ Profiled coordinator stopped")


@pytest_asyncio.fixture
async def performance_server():
    """Start performance test server."""
    from genie.server.server import GenieServer, ServerConfig

    config = ServerConfig(
        node_id='perf-test-server',
        control_port=16555,
        data_port=16556,
        gpu_indices=[0] if torch.cuda.is_available() else None,
        prefer_dpdk=False,
        tcp_fallback=True
    )

    server = GenieServer(config)
    success = await server.start()

    if not success:
        pytest.fail("Performance test server failed to start")

    # Wait for server to be ready
    await asyncio.sleep(2)

    logger.info("✅ Performance test server started")

    yield server

    # Cleanup
    await server.stop()
    logger.info("✅ Performance test server stopped")


class TestBaselinePerformance:
    """Comprehensive baseline performance measurement."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_operation_latencies(self, performance_server, profiled_coordinator):
        """
        Measure latency for all supported operations.

        This establishes TCP baseline for comparison with DPDK later.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST: Operation Latency Measurement")
        logger.info("="*80)

        # Define test cases with different tensor sizes and operation types
        operations_to_test = [
            # Unary operations (single input)
            ('aten::relu', [torch.randn(1000, 1000)], 'unary'),
            ('aten::sigmoid', [torch.randn(1000, 1000)], 'unary'),
            ('aten::tanh', [torch.randn(1000, 1000)], 'unary'),
            ('aten::sum', [torch.randn(1000, 1000)], 'reduction'),
            ('aten::mean', [torch.randn(1000, 1000)], 'reduction'),

            # Binary operations (two inputs)
            ('aten::add', [torch.randn(1000, 1000), torch.randn(1000, 1000)], 'binary'),
            ('aten::mul', [torch.randn(1000, 1000), torch.randn(1000, 1000)], 'binary'),
            ('aten::matmul', [torch.randn(1000, 1000), torch.randn(1000, 1000)], 'binary'),

            # Different tensor sizes for scaling analysis
            ('aten::relu', [torch.randn(100, 100)], 'unary_small'),
            ('aten::relu', [torch.randn(5000, 5000)], 'unary_large'),
            ('aten::matmul', [torch.randn(500, 500), torch.randn(500, 500)], 'binary_medium'),
        ]

        results = {}

        for operation, inputs, op_type in operations_to_test:
            logger.info(f"Measuring {op_type} operation: {operation} with shape {inputs[0].shape}")

            # Warmup phase
            logger.debug("  Warmup phase...")
            for _ in range(5):
                try:
                    await profiled_coordinator.execute_remote_operation(
                        operation, inputs, 'localhost:16556', timeout=10.0
                    )
                except RuntimeError:
                    pass  # Expected to fail sometimes during warmup

            # Measurement phase
            logger.debug("  Measurement phase...")
            latencies = []
            successful_runs = 0

            for i in range(100):  # 100 measurements per operation
                try:
                    start = time.perf_counter()
                    result = await profiled_coordinator.execute_remote_operation(
                        operation, inputs, 'localhost:16556', timeout=10.0
                    )
                    latency = time.perf_counter() - start
                    latencies.append(latency * 1000)  # Convert to ms
                    successful_runs += 1

                    if (i + 1) % 25 == 0:
                        logger.debug(f"    Completed {i + 1}/100 measurements")

                except RuntimeError as e:
                    logger.debug(f"    Measurement {i + 1} failed: {e}")
                    continue

            if not latencies:
                logger.warning(f"  No successful measurements for {operation}")
                continue

            # Compute percentiles
            results[operation] = {
                'operation_type': op_type,
                'input_shapes': [list(t.shape) for t in inputs],
                'input_dtypes': [str(t.dtype) for t in inputs],
                'measurements': len(latencies),
                'successful_runs': successful_runs,
                'success_rate': successful_runs / 100,
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'total_bytes': sum(t.numel() * t.element_size() for t in inputs),
                'bytes_per_ms': (sum(t.numel() * t.element_size() for t in inputs) / np.mean(latencies)) if latencies else 0
            }

            logger.info(f"  ✅ {operation}: p50={results[operation]['p50']:.2f}ms  "
                       f"p95={results[operation]['p95']:.2f}ms  "
                       f"p99={results[operation]['p99']:.2f}ms  "
                       f"({successful_runs}/100 successful)")

        # Generate comprehensive report
        self._generate_latency_report(results)

        # Save detailed results
        with open('results/tcp_baseline_latency_detailed.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("✅ Operation latency measurement complete")

        # Assertions for CI
        for operation, stats in results.items():
            assert stats['success_rate'] > 0.8, f"{operation} success rate too low: {stats['success_rate']:.1%}"
            assert stats['p95'] < 1000, f"{operation} p95 latency too high: {stats['p95']:.2f}ms"

        return results

    def _generate_latency_report(self, results: Dict):
        """Generate formatted performance report."""
        report = []
        report.append("\n" + "="*80)
        report.append("TCP Baseline Performance (1000x1000 tensors unless noted)")
        report.append("="*80)

        # Group by operation type
        by_type = {}
        for op, stats in results.items():
            op_type = stats['operation_type']
            if op_type not in by_type:
                by_type[op_type] = []
            by_type[op_type].append((op, stats))

        for op_type, operations in by_type.items():
            report.append(f"\n{op_type.upper()} OPERATIONS:")
            report.append("-" * 50)

            for op, stats in operations:
                report.append(f"{op:20s}: p50={stats['p50']:6.2f}ms  p95={stats['p95']:6.2f}ms  "
                           f"p99={stats['p99']:6.2f}ms  ({stats['success_rate']:.0%})")

        report.append("\n" + "="*80)

        # Print to console and save to file
        report_text = "\n".join(report)
        logger.info(report_text)

        with open('results/tcp_baseline_latency.txt', 'w') as f:
            f.write(report_text)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_sustained_throughput(self, performance_server, profiled_coordinator):
        """Measure sustained throughput under continuous load."""
        logger.info("\n" + "="*80)
        logger.info("TEST: Sustained Throughput Measurement")
        logger.info("="*80)

        duration_seconds = 30  # 30 second test
        operation_count = 0
        successful_operations = 0

        start_time = time.time()
        end_time = start_time + duration_seconds

        logger.info(f"Measuring throughput for {duration_seconds}s...")

        while time.time() < end_time:
            # Batch of concurrent operations
            batch_size = 5
            tasks = []

            for _ in range(batch_size):
                tensor = torch.randn(100, 100)  # Small tensors for throughput test
                tasks.append(
                    profiled_coordinator.execute_remote_operation(
                        'aten::relu', [tensor], 'localhost:16556', timeout=5.0
                    )
                )

            # Execute batch concurrently
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                operation_count += batch_size
                successful_operations += sum(1 for r in batch_results
                                           if not isinstance(r, Exception))

                # Progress logging
                elapsed = time.time() - start_time
                if operation_count % 50 == 0:
                    rate = operation_count / elapsed
                    success_rate = successful_operations / operation_count
                    logger.info(f"  Progress: {operation_count} ops in {elapsed:.1f}s "
                              f"({rate:.1f} ops/sec, {success_rate:.1%} success)")

            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                continue

        total_elapsed = time.time() - start_time
        throughput = operation_count / total_elapsed
        success_rate = successful_operations / operation_count if operation_count > 0 else 0

        throughput_results = {
            'duration_seconds': total_elapsed,
            'total_operations': operation_count,
            'successful_operations': successful_operations,
            'throughput_ops_per_sec': throughput,
            'success_rate': success_rate,
            'avg_latency_ms': (total_elapsed / operation_count) * 1000 if operation_count > 0 else 0
        }

        logger.info(f"\nThroughput Results:")
        logger.info(f"  Duration: {total_elapsed:.2f}s")
        logger.info(f"  Operations: {operation_count} total, {successful_operations} successful")
        logger.info(f"  Throughput: {throughput:.1f} ops/sec")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Avg latency: {throughput_results['avg_latency_ms']:.2f}ms per operation")

        # Save results
        with open('results/tcp_baseline_throughput.json', 'w') as f:
            json.dump(throughput_results, f, indent=2)

        # Assertions
        assert throughput > 10, f"Throughput too low: {throughput:.1f} ops/sec"
        assert success_rate > 0.9, f"Success rate too low: {success_rate:.1%}"

        logger.info("✅ Sustained throughput measurement complete")
        return throughput_results

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_component_timing_breakdown(self, performance_server, profiled_coordinator):
        """Measure detailed component timing breakdown."""
        logger.info("\n" + "="*80)
        logger.info("TEST: Component Timing Breakdown")
        logger.info("="*80)

        # Run operations with profiling enabled
        test_operations = [
            ('aten::relu', [torch.randn(1000, 1000)]),
            ('aten::matmul', [torch.randn(500, 500), torch.randn(500, 500)]),
            ('aten::add', [torch.randn(2000, 2000), torch.randn(2000, 2000)]),
        ]

        logger.info("Running operations with detailed profiling...")

        for operation, inputs in test_operations:
            logger.info(f"  Profiling {operation}...")
            try:
                await profiled_coordinator.execute_remote_operation(
                    operation, inputs, 'localhost:16556', timeout=10.0
                )
            except RuntimeError:
                logger.warning(f"  {operation} failed, continuing with other operations")

        # Generate profiling report
        report = profiled_coordinator.profiler.generate_report()
        logger.info(f"\n{report}")

        # Save detailed profiling data
        profiled_coordinator.profiler.save_report('results/tcp_profiling_detailed.json')

        # Analyze bottlenecks
        bottleneck_analysis = profiled_coordinator.profiler.get_bottleneck_analysis()
        logger.info("Bottleneck Analysis:")
        for component, data in bottleneck_analysis['component_bottlenecks'].items():
            if data['is_bottleneck']:
                logger.info(f"  🚨 BOTTLENECK: {component} ({data['percentage']:.1f}% of time)")
            else:
                logger.info(f"  ✅ OK: {component} ({data['percentage']:.1f}% of time)")

        # Save bottleneck analysis
        with open('results/tcp_bottleneck_analysis.json', 'w') as f:
            json.dump(bottleneck_analysis, f, indent=2)

        logger.info("✅ Component timing breakdown complete")

        # Assertions
        assert len(profiled_coordinator.profiler.measurements) > 0, "No profiling measurements collected"
        assert 'component_bottlenecks' in bottleneck_analysis, "Bottleneck analysis failed"

        return bottleneck_analysis

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_patterns(self, performance_server, profiled_coordinator):
        """Measure memory usage patterns during operations."""
        logger.info("\n" + "="*80)
        logger.info("TEST: Memory Usage Patterns")
        logger.info("="*80)

        # Test different tensor sizes to understand memory scaling
        tensor_sizes = [100, 500, 1000, 2000]

        memory_results = {}

        for size in tensor_sizes:
            logger.info(f"Testing memory usage with {size}x{size} tensors...")

            # Measure memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run operations
            operations = [
                ('aten::relu', [torch.randn(size, size)]),
                ('aten::add', [torch.randn(size, size), torch.randn(size, size)]),
            ]

            operation_times = []
            for operation, inputs in operations:
                try:
                    start = time.perf_counter()
                    result = await profiled_coordinator.execute_remote_operation(
                        operation, inputs, 'localhost:16556', timeout=10.0
                    )
                    operation_time = time.perf_counter() - start
                    operation_times.append(operation_time)

                    # Force garbage collection
                    import gc
                    gc.collect()

                except RuntimeError as e:
                    logger.debug(f"  {operation} failed: {e}")
                    operation_times.append(None)

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before

            memory_results[f"{size}x{size}"] = {
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_delta_mb': memory_delta,
                'tensor_bytes': size * size * 4,  # float32 = 4 bytes per element
                'operations_successful': sum(1 for t in operation_times if t is not None),
                'avg_operation_time': np.mean([t for t in operation_times if t is not None]) if operation_times else 0
            }

            logger.info(f"  Memory: {memory_before:.1f}MB → {memory_after:.1f}MB "
                       f"(Δ{memory_delta:+.1f}MB)")
            logger.info(f"  Operations: {memory_results[f'{size}x{size}']['operations_successful']}/2 successful")
            logger.info(f"  Avg time: {memory_results[f'{size}x{size}']['avg_operation_time']*1000:.2f}ms")

        # Save memory analysis
        with open('results/tcp_memory_usage_patterns.json', 'w') as f:
            json.dump(memory_results, f, indent=2)

        # Generate memory scaling report
        self._generate_memory_scaling_report(memory_results)

        logger.info("✅ Memory usage patterns measurement complete")
        return memory_results

    def _generate_memory_scaling_report(self, memory_results: Dict):
        """Generate memory scaling analysis report."""
        report = []
        report.append("\n" + "="*80)
        report.append("Memory Usage Scaling Analysis")
        report.append("="*80)

        sizes = sorted([int(k.split('x')[0]) for k in memory_results.keys()])
        memory_deltas = [memory_results[f"{s}x{s}"]['memory_delta_mb'] for s in sizes]

        report.append("Tensor Size → Memory Overhead (MB):")
        report.append("-" * 40)

        for size, delta in zip(sizes, memory_deltas):
            tensor_mb = memory_results[f"{size}x{size}"]['tensor_bytes'] / 1024 / 1024
            report.append(f"{size:4d}x{size:4d} ({tensor_mb:6.1f}MB) → Δ{delta:6.1f}MB")

        # Calculate scaling factor
        if len(sizes) >= 2:
            scaling_factor = memory_deltas[-1] / memory_deltas[0] if memory_deltas[0] != 0 else 0
            expected_linear = sizes[-1] / sizes[0]
            report.append(f"\nScaling factor: {scaling_factor:.2f}x (expected linear: {expected_linear:.2f}x)")

            if scaling_factor > expected_linear * 1.2:
                report.append("⚠️  Memory usage scales super-linearly - potential memory leak")
            elif scaling_factor < expected_linear * 0.8:
                report.append("✅ Memory usage scales sub-linearly - good memory efficiency")
            else:
                report.append("✅ Memory usage scales linearly - expected behavior")

        report.append("="*80)

        report_text = "\n".join(report)
        logger.info(report_text)

        with open('results/tcp_memory_scaling_analysis.txt', 'w') as f:
            f.write(report_text)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_network_bandwidth_utilization(self, performance_server, profiled_coordinator):
        """Measure network bandwidth utilization."""
        logger.info("\n" + "="*80)
        logger.info("TEST: Network Bandwidth Utilization")
        logger.info("="*80)

        # Test different data sizes to understand bandwidth scaling
        test_configs = [
            {'size': (100, 100), 'count': 100, 'name': 'small_frequent'},
            {'size': (1000, 1000), 'count': 10, 'name': 'large_infrequent'},
            {'size': (2000, 2000), 'count': 5, 'name': 'very_large_rare'},
        ]

        bandwidth_results = {}

        for config in test_configs:
            logger.info(f"Testing {config['name']}: {config['size']} tensors, {config['count']} operations")

            total_bytes = 0
            total_time = 0

            for i in range(config['count']):
                tensor = torch.randn(*config['size'])

                try:
                    start = time.perf_counter()
                    result = await profiled_coordinator.execute_remote_operation(
                        'aten::relu', [tensor], 'localhost:16556', timeout=15.0
                    )
                    operation_time = time.perf_counter() - start
                    total_time += operation_time
                    total_bytes += tensor.numel() * tensor.element_size()

                except RuntimeError as e:
                    logger.debug(f"  Operation {i + 1} failed: {e}")
                    continue

            if total_time > 0:
                bandwidth_mbps = (total_bytes * 2) / total_time / 1024 / 1024  # *2 for send + receive
                avg_latency_ms = (total_time / config['count']) * 1000

                bandwidth_results[config['name']] = {
                    'total_bytes': total_bytes,
                    'total_time': total_time,
                    'operations': config['count'],
                    'bandwidth_mbps': bandwidth_mbps,
                    'avg_latency_ms': avg_latency_ms,
                    'bytes_per_operation': total_bytes / config['count']
                }

                logger.info(f"  Bandwidth: {bandwidth_mbps:.1f} Mbps")
                logger.info(f"  Avg latency: {avg_latency_ms:.2f} ms")
                logger.info(f"  Bytes per op: {total_bytes / config['count'] / 1024:.1f} KB")

        # Generate bandwidth report
        self._generate_bandwidth_report(bandwidth_results)

        # Save results
        with open('results/tcp_network_bandwidth.json', 'w') as f:
            json.dump(bandwidth_results, f, indent=2)

        logger.info("✅ Network bandwidth utilization measurement complete")
        return bandwidth_results

    def _generate_bandwidth_report(self, bandwidth_results: Dict):
        """Generate network bandwidth analysis report."""
        report = []
        report.append("\n" + "="*80)
        report.append("Network Bandwidth Analysis")
        report.append("="*80)

        report.append("Configuration → Bandwidth (Mbps):")
        report.append("-" * 50)

        for config_name, results in bandwidth_results.items():
            report.append(f"{config_name:15s}: {results['bandwidth_mbps']:6.1f} Mbps "
                         f"({results['avg_latency_ms']:5.1f}ms avg)")

        # Calculate average bandwidth
        if bandwidth_results:
            avg_bandwidth = np.mean([r['bandwidth_mbps'] for r in bandwidth_results.values()])
            report.append(f"\nAverage bandwidth: {avg_bandwidth:.1f} Mbps")

            # Estimate theoretical maximum (assuming 1 Gbps network)
            theoretical_max = 1000  # 1 Gbps
            efficiency = (avg_bandwidth / theoretical_max) * 100
            report.append(f"Network efficiency: {efficiency:.1f}% of theoretical maximum")

            if efficiency < 50:
                report.append("⚠️  Network efficiency is low - potential protocol overhead")
            elif efficiency > 80:
                report.append("✅ Network efficiency is high - good protocol design")
            else:
                report.append("✅ Network efficiency is acceptable")

        report.append("="*80)

        report_text = "\n".join(report)
        logger.info(report_text)

        with open('results/tcp_network_bandwidth_analysis.txt', 'w') as f:
            f.write(report_text)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_connection_pool_efficiency(self, performance_server, profiled_coordinator):
        """Test connection pool efficiency under various loads."""
        logger.info("\n" + "="*80)
        logger.info("TEST: Connection Pool Efficiency")
        logger.info("="*80)

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]

        pool_results = {}

        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")

            duration = 10  # 10 seconds per concurrency level
            start_time = time.time()
            end_time = start_time + duration

            operation_count = 0
            successful_operations = 0

            while time.time() < end_time:
                # Execute batch of concurrent operations
                tasks = []
                for _ in range(concurrency):
                    tensor = torch.randn(50, 50)  # Small tensors for connection stress test
                    tasks.append(
                        profiled_coordinator.execute_remote_operation(
                            'aten::relu', [tensor], 'localhost:16556', timeout=5.0
                        )
                    )

                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    operation_count += concurrency
                    successful_operations += sum(1 for r in results
                                               if not isinstance(r, Exception))

                except Exception as e:
                    logger.debug(f"Batch failed at concurrency {concurrency}: {e}")
                    break

            elapsed = time.time() - start_time
            throughput = operation_count / elapsed if elapsed > 0 else 0
            success_rate = successful_operations / operation_count if operation_count > 0 else 0

            # Get connection pool stats
            pool_stats = profiled_coordinator.transports['tcp'].get_connection_pool_stats()

            pool_results[concurrency] = {
                'duration': elapsed,
                'operations': operation_count,
                'successful': successful_operations,
                'throughput': throughput,
                'success_rate': success_rate,
                'pool_stats': pool_stats
            }

            logger.info(f"  Concurrency {concurrency}: {throughput:.1f} ops/sec, "
                       f"{success_rate:.1%} success")
            logger.info(f"  Pool: {pool_stats['hit_rate']:.1%} hit rate, "
                       f"{pool_stats['created']} created, {pool_stats['reused']} reused")

        # Generate pool efficiency report
        self._generate_pool_efficiency_report(pool_results)

        # Save results
        with open('results/tcp_connection_pool_efficiency.json', 'w') as f:
            json.dump(pool_results, f, indent=2)

        logger.info("✅ Connection pool efficiency measurement complete")
        return pool_results

    def _generate_pool_efficiency_report(self, pool_results: Dict):
        """Generate connection pool efficiency analysis."""
        report = []
        report.append("\n" + "="*80)
        report.append("Connection Pool Efficiency Analysis")
        report.append("="*80)

        report.append("Concurrency → Throughput (ops/sec) → Pool Efficiency:")
        report.append("-" * 60)

        for concurrency, results in sorted(pool_results.items()):
            throughput = results['throughput']
            hit_rate = results['pool_stats']['hit_rate']
            report.append(f"{concurrency:3d} concurrent: {throughput:6.1f} ops/sec "
                         f"(hit rate: {hit_rate:.1%})")

        # Analyze scaling
        concurrencies = sorted(pool_results.keys())
        throughputs = [pool_results[c]['throughput'] for c in concurrencies]

        if len(concurrencies) >= 2:
            scaling_efficiency = throughputs[-1] / throughputs[0] if throughputs[0] > 0 else 0
            ideal_scaling = concurrencies[-1] / concurrencies[0]
            scaling_ratio = scaling_efficiency / ideal_scaling

            report.append(f"\nScaling efficiency: {scaling_efficiency:.2f}x "
                         f"(ideal: {ideal_scaling:.2f}x)")
            report.append(f"Scaling ratio: {scaling_ratio:.2f}")

            if scaling_ratio > 0.8:
                report.append("✅ Excellent scaling - system handles concurrency well")
            elif scaling_ratio > 0.5:
                report.append("⚠️  Good scaling - some contention but acceptable")
            else:
                report.append("⚠️  Poor scaling - significant contention bottlenecks")

        # Pool performance summary
        final_concurrency = max(pool_results.keys())
        final_stats = pool_results[final_concurrency]['pool_stats']

        report.append(f"\nFinal pool stats at {final_concurrency} concurrency:")
        report.append(f"  Hit rate: {final_stats['hit_rate']:.1%}")
        report.append(f"  Connections created: {final_stats['created']}")
        report.append(f"  Connections reused: {final_stats['reused']}")
        report.append(f"  Total errors: {final_stats['errors']}")

        if final_stats['hit_rate'] > 0.8:
            report.append("✅ Connection pool is highly efficient")
        elif final_stats['hit_rate'] > 0.5:
            report.append("✅ Connection pool is reasonably efficient")
        else:
            report.append("⚠️  Connection pool efficiency could be improved")

        report.append("="*80)

        report_text = "\n".join(report)
        logger.info(report_text)

        with open('results/tcp_connection_pool_analysis.txt', 'w') as f:
            f.write(report_text)


if __name__ == '__main__':
    # Run performance tests directly
    logger.info("Running TCP Baseline Performance Tests...")

    # Configure logging for performance tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )

    pytest.main([__file__, '-v', '-s', '--tb=short'])
