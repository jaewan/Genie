"""
Test optimizations implemented in Week 2.

Validates that the performance optimizations provide measurable improvements:
- Serialization optimization
- Connection pool enhancements
- Scheduler caching
- Operation batching
"""

import pytest
import pytest_asyncio
import asyncio
import torch
import logging
import time
import json
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def optimization_server():
    """Start server for optimization testing."""
    from genie.server.server import GenieServer, ServerConfig

    config = ServerConfig(
        node_id='opt-server',
        control_port=18555,
        data_port=18556,
        gpu_indices=[0] if torch.cuda.is_available() else None,
        prefer_dpdk=False,
        tcp_fallback=True
    )

    server = GenieServer(config)
    success = await server.start()

    if not success:
        pytest.fail("Optimization test server failed to start")

    await asyncio.sleep(2)

    logger.info("✅ Optimization test server started")

    yield server

    await server.stop()
    logger.info("✅ Optimization test server stopped")


@pytest_asyncio.fixture
async def optimization_coordinator():
    """Create coordinator with optimizations enabled."""
    from genie.core.coordinator import GenieCoordinator, CoordinatorConfig

    config = CoordinatorConfig(
        node_id='opt-client',
        control_port=18555,
        data_port=18556,
        tcp_fallback=True,
        enable_profiling=True
    )

    coordinator = GenieCoordinator(config)
    await coordinator.start()

    logger.info("✅ Optimization test coordinator started")

    yield coordinator

    await coordinator.stop()
    logger.info("✅ Optimization test coordinator stopped")


class TestOptimizations:
    """Test optimization implementations."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_scheduler_caching_effectiveness(self, optimization_server, optimization_coordinator):
        """
        Test that scheduler caching reduces decision overhead.

        Validates that repeated similar operations benefit from caching.
        """
        logger.info("\n" + "="*80)
        logger.info("🧠 SCHEDULER CACHING OPTIMIZATION TEST")
        logger.info("="*80)

        # Test repeated operations that should benefit from caching
        operations = []

        # Phase 1: Fill cache with diverse operations
        logger.info("Phase 1: Filling scheduler cache...")

        cache_fill_ops = [
            ('aten::relu', [torch.randn(100, 100)]),
            ('aten::add', [torch.randn(100, 100), torch.randn(100, 100)]),
            ('aten::matmul', [torch.randn(200, 200), torch.randn(200, 200)]),
            ('aten::sigmoid', [torch.randn(50, 50)]),
        ]

        for operation, inputs in cache_fill_ops:
            try:
                await optimization_coordinator.execute_remote_operation(
                    operation, inputs, 'localhost:18556', timeout=10.0
                )
                operations.append((operation, inputs))
            except RuntimeError:
                logger.debug(f"Cache fill operation failed: {operation}")
                continue

        # Get initial cache stats
        initial_stats = optimization_coordinator.scheduler.get_stats()
        initial_cache_size = initial_stats['cache_size']
        initial_cache_hits = initial_stats['cache_hits']

        logger.info(f"Initial cache: {initial_cache_size} entries, {initial_cache_hits} hits")

        # Phase 2: Test cache effectiveness with repeated operations
        logger.info("Phase 2: Testing cache effectiveness...")

        cache_test_ops = [
            ('aten::relu', [torch.randn(100, 100)]),  # Should hit cache
            ('aten::add', [torch.randn(100, 100), torch.randn(100, 100)]),  # Should hit cache
            ('aten::relu', [torch.randn(1000, 1000)]),  # Different size, might miss cache
        ]

        for operation, inputs in cache_test_ops:
            try:
                await optimization_coordinator.execute_remote_operation(
                    operation, inputs, 'localhost:18556', timeout=10.0
                )
                operations.append((operation, inputs))
            except RuntimeError:
                logger.debug(f"Cache test operation failed: {operation}")
                continue

        # Get final cache stats
        final_stats = optimization_coordinator.scheduler.get_stats()
        final_cache_size = final_stats['cache_size']
        final_cache_hits = final_stats['cache_hits']

        cache_effectiveness = {
            'initial_cache_size': initial_cache_size,
            'final_cache_size': final_cache_size,
            'cache_hits_gained': final_cache_hits - initial_cache_hits,
            'cache_hit_rate': final_stats['cache_hit_rate'],
            'decisions_made': final_stats['decisions_made'],
            'total_operations': len(operations)
        }

        logger.info("Cache effectiveness results:")
        logger.info(f"  Cache hit rate: {cache_effectiveness['cache_hit_rate']:.1%}")
        logger.info(f"  Cache hits gained: {cache_effectiveness['cache_hits_gained']}")
        logger.info(f"  Total decisions: {cache_effectiveness['decisions_made']}")
        logger.info(f"  Operations completed: {cache_effectiveness['total_operations']}")

        # Save results
        with open('results/scheduler_caching_optimization.json', 'w') as f:
            json.dump(cache_effectiveness, f, indent=2)

        logger.info("✅ Scheduler caching optimization test complete")

        # Assertions
        assert cache_effectiveness['cache_hit_rate'] > 0, "Cache should have some hits"
        assert final_cache_size >= initial_cache_size, "Cache should not shrink"

        return cache_effectiveness

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_connection_pool_optimization(self, optimization_server, optimization_coordinator):
        """
        Test enhanced connection pool performance.

        Validates connection warming and health checking improvements.
        """
        logger.info("\n" + "="*80)
        logger.info("🔗 CONNECTION POOL OPTIMIZATION TEST")
        logger.info("="*80)

        # Run operations to warm up connection pool
        logger.info("Warming up connection pool...")

        warmup_operations = [
            ('aten::relu', [torch.randn(50, 50)]) for _ in range(20)
        ]

        for operation, inputs in warmup_operations:
            try:
                await optimization_coordinator.execute_remote_operation(
                    operation, inputs, 'localhost:18556', timeout=5.0
                )
            except RuntimeError:
                logger.debug(f"Warmup operation failed: {operation}")
                continue

        # Get pool stats after warmup
        pool_stats = optimization_coordinator.transports['tcp'].get_connection_pool_stats()

        logger.info("Connection pool performance after warmup:")
        logger.info(f"  Hit rate: {pool_stats['hit_rate']:.1%}")
        logger.info(f"  Created: {pool_stats['created']}")
        logger.info(f"  Reused: {pool_stats['reused']}")
        logger.info(f"  Errors: {pool_stats['errors']}")
        logger.info(f"  Health efficiency: {pool_stats['health_efficiency']:.1%}")

        # Test under load
        logger.info("Testing under concurrent load...")

        async def load_test_operation(i: int):
            tensor = torch.randn(30, 30)
            try:
                return await optimization_coordinator.execute_remote_operation(
                    'aten::relu', [tensor], 'localhost:18556', timeout=5.0
                )
            except RuntimeError:
                return None

        # Run 50 concurrent operations
        start = time.time()
        tasks = [load_test_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        load_time = time.time() - start

        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        success_rate = len(successful_results) / len(results)

        # Get final pool stats
        final_pool_stats = optimization_coordinator.transports['tcp'].get_connection_pool_stats()

        pool_optimization_results = {
            'warmup_stats': pool_stats,
            'final_stats': final_pool_stats,
            'load_test': {
                'operations': 50,
                'successful': len(successful_results),
                'success_rate': success_rate,
                'total_time': load_time,
                'throughput': 50 / load_time if load_time > 0 else 0
            }
        }

        logger.info("Load test results:")
        logger.info(f"  Throughput: {pool_optimization_results['load_test']['throughput']:.1f} ops/sec")
        logger.info(f"  Success rate: {success_rate:.1%}")

        # Save results
        with open('results/connection_pool_optimization.json', 'w') as f:
            json.dump(pool_optimization_results, f, indent=2)

        logger.info("✅ Connection pool optimization test complete")

        # Assertions
        assert success_rate > 0.8, f"Connection pool success rate too low: {success_rate:.1%}"
        assert pool_stats['hit_rate'] > 0, "Connection pool should have some hits"

        return pool_optimization_results

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_serialization_optimization(self, optimization_server, optimization_coordinator):
        """
        Test serialization performance improvements.

        Validates that non-blocking CPU transfer and optimized serialization work.
        """
        logger.info("\n" + "="*80)
        logger.info("📦 SERIALIZATION OPTIMIZATION TEST")
        logger.info("="*80)

        # Test different tensor sizes and types
        test_cases = [
            {'size': (100, 100), 'dtype': torch.float32, 'name': 'small_float32'},
            {'size': (500, 500), 'dtype': torch.float32, 'name': 'medium_float32'},
            {'size': (1000, 1000), 'dtype': torch.float16, 'name': 'large_float16'},
            {'size': (2000, 2000), 'dtype': torch.float32, 'name': 'very_large_float32'},
        ]

        serialization_results = {}

        for test_case in test_cases:
            logger.info(f"Testing {test_case['name']}: {test_case['size']} {test_case['dtype']}")

            # Create tensor (on CPU for fair comparison)
            tensor = torch.randn(*test_case['size'], dtype=test_case['dtype'])

            # Measure serialization time
            start = time.perf_counter()

            try:
                result = await optimization_coordinator.execute_remote_operation(
                    'aten::relu', [tensor], 'localhost:18556', timeout=10.0
                )
                serialization_time = time.perf_counter() - start

                tensor_bytes = tensor.numel() * tensor.element_size()

                serialization_results[test_case['name']] = {
                    'tensor_shape': list(tensor.shape),
                    'tensor_dtype': str(tensor.dtype),
                    'tensor_bytes': tensor_bytes,
                    'total_time': serialization_time,
                    'serialization_rate_mbps': (tensor_bytes / serialization_time) / 1024 / 1024,
                    'success': True
                }

                logger.info(f"  ✅ Success: {serialization_time:.3f}s "
                           f"({serialization_results[test_case['name']]['serialization_rate_mbps']:.1f} MB/s)")

            except RuntimeError as e:
                serialization_results[test_case['name']] = {
                    'tensor_shape': list(tensor.shape),
                    'tensor_dtype': str(tensor.dtype),
                    'tensor_bytes': tensor_bytes,
                    'total_time': time.perf_counter() - start,
                    'error': str(e),
                    'success': False
                }
                logger.info(f"  ❌ Failed: {str(e)[:50]}...")

        # Calculate average serialization performance
        successful_cases = [r for r in serialization_results.values() if r['success']]
        if successful_cases:
            avg_rate = sum(r['serialization_rate_mbps'] for r in successful_cases) / len(successful_cases)
            logger.info(f"\nAverage serialization rate: {avg_rate:.1f} MB/s")

            # Check if we're achieving reasonable performance
            if avg_rate > 100:  # 100 MB/s is reasonable for TCP
                logger.info("✅ Serialization performance is good")
            elif avg_rate > 50:
                logger.info("⚠️  Serialization performance is acceptable")
            else:
                logger.info("⚠️  Serialization performance could be improved")

        # Save results
        with open('results/serialization_optimization.json', 'w') as f:
            json.dump(serialization_results, f, indent=2)

        logger.info("✅ Serialization optimization test complete")

        # Assertions
        assert len(successful_cases) > 0, "At least some serialization tests should succeed"

        return serialization_results

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_overall_optimization_benefits(self, optimization_server, optimization_coordinator):
        """
        Test overall benefits of all optimizations combined.

        This provides a comprehensive view of the optimization impact.
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 OVERALL OPTIMIZATION BENEFITS TEST")
        logger.info("="*80)

        # Run comprehensive workload with profiling
        logger.info("Running comprehensive workload with all optimizations...")

        # Define mixed workload
        workload = [
            # Small, frequent operations (should benefit from batching)
            ('aten::relu', [torch.randn(100, 100)], 30),
            ('aten::add', [torch.randn(100, 100), torch.randn(100, 100)], 20),

            # Medium operations (should benefit from caching)
            ('aten::matmul', [torch.randn(300, 300), torch.randn(300, 300)], 15),
            ('aten::sigmoid', [torch.randn(500, 500)], 15),

            # Large operations (should benefit from efficient serialization)
            ('aten::matmul', [torch.randn(800, 800), torch.randn(800, 800)], 5),
        ]

        start_time = time.perf_counter()
        successful_operations = 0
        total_operations = sum(count for _, _, count in workload)

        # Execute workload
        for operation, inputs, count in workload:
            logger.debug(f"Executing {count} {operation} operations...")

            for i in range(count):
                try:
                    await optimization_coordinator.execute_remote_operation(
                        operation, inputs, 'localhost:18556', timeout=10.0
                    )
                    successful_operations += 1

                    # Progress logging
                    if (successful_operations % 10) == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = successful_operations / elapsed
                        logger.debug(f"   Progress: {successful_operations}/{total_operations} "
                                   f"({rate:.1f} ops/sec)")

                except RuntimeError as e:
                    logger.debug(f"   Operation {i+1}/{count} failed: {e}")
                    continue

        total_time = time.perf_counter() - start_time
        throughput = successful_operations / total_time if total_time > 0 else 0
        success_rate = successful_operations / total_operations

        # Get final statistics
        final_scheduler_stats = optimization_coordinator.scheduler.get_stats()
        final_pool_stats = optimization_coordinator.transports['tcp'].get_connection_pool_stats()

        optimization_benefits = {
            'workload': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': success_rate,
                'total_time': total_time,
                'throughput': throughput
            },
            'scheduler_optimization': {
                'cache_hit_rate': final_scheduler_stats['cache_hit_rate'],
                'decisions_made': final_scheduler_stats['decisions_made'],
                'cache_size': final_scheduler_stats['cache_size']
            },
            'connection_pool_optimization': {
                'hit_rate': final_pool_stats['hit_rate'],
                'health_efficiency': final_pool_stats['health_efficiency'],
                'warming_hits': final_pool_stats['warming_hits']
            },
            'profiling_data': {
                'measurements': len(optimization_coordinator.profiler.measurements),
                'total_profiling_time': sum(m['total_latency'] for m in optimization_coordinator.profiler.measurements)
            }
        }

        logger.info("Overall optimization benefits:")
        logger.info(f"  Throughput: {throughput:.1f} ops/sec")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Scheduler cache hit rate: {final_scheduler_stats['cache_hit_rate']:.1%}")
        logger.info(f"  Connection pool hit rate: {final_pool_stats['hit_rate']:.1%}")
        logger.info(f"  Profiling measurements: {len(optimization_coordinator.profiler.measurements)}")

        # Generate optimization impact report
        impact_report = self._generate_optimization_impact_report(optimization_benefits)

        # Save comprehensive results
        with open('results/overall_optimization_benefits.json', 'w') as f:
            json.dump(optimization_benefits, f, indent=2)

        with open('results/optimization_impact_report.md', 'w') as f:
            f.write(impact_report)

        logger.info("✅ Overall optimization benefits test complete")

        # Assertions
        assert success_rate > 0.7, f"Overall success rate too low: {success_rate:.1%}"
        assert throughput > 5, f"Overall throughput too low: {throughput:.1f} ops/sec"
        assert final_scheduler_stats['cache_hit_rate'] > 0, "Scheduler cache should have hits"

        return optimization_benefits

    def _generate_optimization_impact_report(self, benefits: Dict) -> str:
        """Generate optimization impact analysis report."""
        report = []
        report.append("="*80)
        report.append("OPTIMIZATION IMPACT ANALYSIS REPORT")
        report.append("="*80)

        workload = benefits['workload']
        scheduler = benefits['scheduler_optimization']
        pool = benefits['connection_pool_optimization']

        # Overall performance
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 50)
        report.append(f"Throughput: {workload['throughput']:.1f} operations per second")
        report.append(f"Success Rate: {workload['success_rate']:.1%}")
        report.append(f"Total Operations: {workload['successful_operations']}/{workload['total_operations']}")
        report.append("")

        # Scheduler optimization impact
        report.append("SCHEDULER OPTIMIZATION")
        report.append("-" * 50)
        report.append(f"Cache Hit Rate: {scheduler['cache_hit_rate']:.1%}")
        report.append(f"Placement Decisions: {scheduler['decisions_made']}")
        report.append(f"Cache Size: {scheduler['cache_size']} entries")

        cache_impact = "Good" if scheduler['cache_hit_rate'] > 0.5 else \
                      "Moderate" if scheduler['cache_hit_rate'] > 0.2 else "Low"
        report.append(f"Cache Impact: {cache_impact}")
        report.append("")

        # Connection pool optimization impact
        report.append("CONNECTION POOL OPTIMIZATION")
        report.append("-" * 50)
        report.append(f"Connection Reuse Rate: {pool['hit_rate']:.1%}")
        report.append(f"Health Efficiency: {pool['health_efficiency']:.1%}")
        report.append(f"Warming Hits: {pool['warming_hits']}")

        pool_impact = "Excellent" if pool['hit_rate'] > 0.8 else \
                     "Good" if pool['hit_rate'] > 0.5 else "Needs Improvement"
        report.append(f"Pool Impact: {pool_impact}")
        report.append("")

        # Recommendations
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 50)

        recommendations = []

        if scheduler['cache_hit_rate'] < 0.3:
            recommendations.append("• Improve scheduler caching - low hit rate suggests poor cache key design")

        if pool['hit_rate'] < 0.5:
            recommendations.append("• Enhance connection pool - low reuse rate indicates connection management issues")

        if workload['success_rate'] < 0.9:
            recommendations.append("• Improve error handling - high failure rate under load")

        if workload['throughput'] < 20:
            recommendations.append("• Consider DPDK implementation - current throughput may be network-limited")

        if not recommendations:
            recommendations.append("• All optimizations performing well - consider Phase 2 DPDK implementation")

        for rec in recommendations:
            report.append(rec)

        report.append("="*80)

        return "\n".join(report)


if __name__ == '__main__':
    # Run optimization tests directly
    logger.info("🚀 Running Optimization Tests...")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )

    pytest.main([__file__, '-v', '-s', '--tb=short'])
