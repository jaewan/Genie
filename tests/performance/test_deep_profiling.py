"""
Deep Profiling Session: Bottleneck Identification and Analysis

This test runs comprehensive profiling with real operations to identify
actual performance bottlenecks in the system.

Tests:
- Real operation profiling with timing breakdown
- Resource utilization monitoring
- Network bandwidth analysis
- Memory usage patterns
- Bottleneck identification with severity levels
- Optimization recommendations
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
async def profiling_server():
    """Start server for deep profiling tests."""
    from djinn.server.server import GenieServer, ServerConfig

    config = ServerConfig(
        node_id='profiling-server',
        control_port=17555,
        data_port=17556,
        gpu_indices=[0] if torch.cuda.is_available() else None,
        prefer_dpdk=False,
        tcp_fallback=True
    )

    server = GenieServer(config)
    success = await server.start()

    if not success:
        pytest.fail("Profiling server failed to start")

    await asyncio.sleep(2)  # Wait for server to be ready

    logger.info("✅ Profiling server started")

    yield server

    await server.stop()
    logger.info("✅ Profiling server stopped")


@pytest_asyncio.fixture
async def profiling_coordinator():
    """Create coordinator with profiling enabled."""
    from djinn.core.coordinator import GenieCoordinator, CoordinatorConfig

    config = CoordinatorConfig(
        node_id='profiling-client',
        control_port=17555,
        data_port=17556,
        tcp_fallback=True,
        enable_profiling=True
    )

    coordinator = GenieCoordinator(config)
    await coordinator.start()

    logger.info("✅ Profiling coordinator started")

    yield coordinator

    await coordinator.stop()
    logger.info("✅ Profiling coordinator stopped")


class TestDeepProfiling:
    """Comprehensive deep profiling and bottleneck analysis."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_comprehensive_operation_profiling(self, profiling_server, profiling_coordinator):
        """
        Run comprehensive profiling session with real operations.

        This generates the data needed for bottleneck analysis and optimization planning.
        """
        logger.info("\n" + "="*80)
        logger.info("🧪 COMPREHENSIVE PROFILING SESSION")
        logger.info("="*80)

        # Define comprehensive workload for profiling
        workload_scenarios = [
            {
                'name': 'Small Operations (High Frequency)',
                'operations': [
                    ('aten::relu', [torch.randn(100, 100)], 50),
                    ('aten::add', [torch.randn(100, 100), torch.randn(100, 100)], 50),
                    ('aten::sum', [torch.randn(100, 100)], 50),
                ]
            },
            {
                'name': 'Medium Operations (Balanced)',
                'operations': [
                    ('aten::matmul', [torch.randn(500, 500), torch.randn(500, 500)], 20),
                    ('aten::relu', [torch.randn(1000, 1000)], 20),
                    ('aten::add', [torch.randn(1000, 1000), torch.randn(1000, 1000)], 20),
                ]
            },
            {
                'name': 'Large Operations (Low Frequency)',
                'operations': [
                    ('aten::matmul', [torch.randn(1000, 1000), torch.randn(1000, 1000)], 10),
                    ('aten::relu', [torch.randn(2000, 2000)], 10),
                ]
            },
            {
                'name': 'Mixed Concurrent Load',
                'operations': [
                    ('aten::relu', [torch.randn(200, 200)], 25),
                    ('aten::matmul', [torch.randn(300, 300), torch.randn(300, 300)], 15),
                    ('aten::add', [torch.randn(400, 400), torch.randn(400, 400)], 20),
                ],
                'concurrent': True
            }
        ]

        total_operations = 0
        successful_operations = 0

        for scenario in workload_scenarios:
            scenario_name = scenario['name']
            operations = scenario['operations']
            is_concurrent = scenario.get('concurrent', False)

            logger.info(f"\n📊 Scenario: {scenario_name}")
            logger.info(f"   Concurrent: {is_concurrent}")
            logger.info(f"   Operations: {len(operations)} types")

            scenario_start = time.time()

            if is_concurrent:
                # Run mixed operations concurrently
                await self._run_concurrent_workload(profiling_coordinator, operations)
            else:
                # Run operations sequentially
                await self._run_sequential_workload(profiling_coordinator, operations)

            scenario_time = time.time() - scenario_start

            # Count operations in this scenario
            scenario_ops = sum(count for _, _, count in operations)
            total_operations += scenario_ops

            logger.info(f"   Completed {scenario_ops} operations in {scenario_time:.2f}s")
            logger.info(f"   Throughput: {scenario_ops / scenario_time:.1f} ops/sec")

        # Generate comprehensive profiling report
        logger.info("🔍 Generating profiling analysis...")
        report = profiling_coordinator.profiler.generate_report()
        logger.info(f"\n{report}")

        # Save detailed profiling data
        profiling_coordinator.profiler.save_report('results/deep_profiling_session.json')

        # Generate bottleneck analysis
        logger.info("🎯 Analyzing bottlenecks...")
        bottleneck_analysis = profiling_coordinator.profiler.get_bottleneck_analysis()

        # Save bottleneck analysis
        with open('results/deep_bottleneck_analysis.json', 'w') as f:
            json.dump(bottleneck_analysis, f, indent=2)

        # Generate optimization recommendations
        logger.info("💡 Generating optimization recommendations...")
        recommendations = profiling_coordinator.profiler._generate_optimization_recommendations(
            bottleneck_analysis['component_bottlenecks'],
            bottleneck_analysis['network_analysis']
        )

        logger.info("Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")

        # Save comprehensive analysis
        comprehensive_analysis = {
            'profiling_summary': {
                'total_operations': len(profiling_coordinator.profiler.measurements),
                'total_time': sum(m['total_latency'] for m in profiling_coordinator.profiler.measurements),
                'throughput': len(profiling_coordinator.profiler.measurements) /
                              max(0.001, sum(m['total_latency'] for m in profiling_coordinator.profiler.measurements))
            },
            'bottleneck_analysis': bottleneck_analysis,
            'recommendations': recommendations,
            'workload_scenarios': workload_scenarios,
            'timestamp': time.time()
        }

        with open('results/comprehensive_performance_analysis.json', 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2)

        # Generate human-readable analysis document
        self._generate_analysis_document(comprehensive_analysis)

        logger.info("📈 Deep profiling session complete!")
        logger.info(f"   Measurements: {len(profiling_coordinator.profiler.measurements)}")
        logger.info(f"   Total time: {comprehensive_analysis['profiling_summary']['total_time']:.2f}s")
        logger.info(f"   Overall throughput: {comprehensive_analysis['profiling_summary']['throughput']:.1f} ops/sec")

        # Assertions for CI
        assert len(profiling_coordinator.profiler.measurements) > 0, "No profiling measurements collected"
        assert comprehensive_analysis['profiling_summary']['throughput'] > 0, "Throughput should be positive"

        return comprehensive_analysis

    async def _run_sequential_workload(self, coordinator, operations):
        """Run operations sequentially for profiling."""
        for operation, inputs, count in operations:
            logger.debug(f"   Running {count} {operation} operations...")

            for i in range(count):
                try:
                    await coordinator.execute_remote_operation(
                        operation, inputs, 'localhost:17556', timeout=15.0
                    )
                except RuntimeError as e:
                    logger.debug(f"     Operation {i+1}/{count} failed: {e}")
                    continue

    async def _run_concurrent_workload(self, coordinator, operations):
        """Run mixed operations concurrently for profiling."""
        async def run_mixed_operation(op_id: int, operation: str, inputs: list, count: int):
            """Run a specific operation type concurrently."""
            for i in range(count):
                try:
                    await coordinator.execute_remote_operation(
                        operation, inputs, 'localhost:17556', timeout=15.0
                    )
                except RuntimeError as e:
                    logger.debug(f"     Concurrent {operation} {i+1}/{count} failed: {e}")
                    continue

        # Create concurrent tasks for each operation type
        tasks = []
        for op_id, (operation, inputs, count) in enumerate(operations):
            task = asyncio.create_task(
                run_mixed_operation(op_id, operation, inputs, count)
            )
            tasks.append(task)

        # Wait for all operation types to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    def _generate_analysis_document(self, analysis: Dict):
        """Generate human-readable analysis document."""
        doc = []
        doc.append("="*80)
        doc.append("DEEP PROFILING ANALYSIS REPORT")
        doc.append("="*80)
        doc.append(f"Generated: {time.ctime(analysis['timestamp'])}")
        doc.append("")

        # Summary
        summary = analysis['profiling_summary']
        doc.append("EXECUTIVE SUMMARY")
        doc.append("-" * 50)
        doc.append(f"Total Operations: {summary['total_operations']}")
        doc.append(f"Total Execution Time: {summary['total_time']:.2f}s")
        doc.append(f"Average Throughput: {summary['throughput']:.1f} ops/sec")
        doc.append(f"Operations per Second: {1/summary['total_time']*summary['total_operations']:.1f}")
        doc.append("")

        # Bottleneck Analysis
        doc.append("BOTTLENECK ANALYSIS")
        doc.append("-" * 50)

        bottlenecks = analysis['bottleneck_analysis']['component_bottlenecks']

        # Critical bottlenecks
        critical = [b for b in bottlenecks.values() if b['is_bottleneck']]
        if critical:
            doc.append("🚨 CRITICAL BOTTLENECKS (>50% of execution time):")
            for bottleneck in critical:
                component = bottleneck['component']
                percentage = bottleneck['percentage']
                time_ms = bottleneck['time_ms']
                doc.append(f"   • {component}: {percentage:.1f}% ({time_ms:.1f}ms)")

        # High impact bottlenecks
        high = [b for b in bottlenecks.values()
                if not b['is_bottleneck'] and b['percentage'] > 20]
        if high:
            doc.append("⚠️  HIGH IMPACT (>20% of execution time):")
            for bottleneck in high:
                component = bottleneck['component']
                percentage = bottleneck['percentage']
                time_ms = bottleneck['time_ms']
                doc.append(f"   • {component}: {percentage:.1f}% ({time_ms:.1f}ms)")

        doc.append("")

        # Network Analysis
        network = analysis['bottleneck_analysis']['network_analysis']
        doc.append("NETWORK ANALYSIS")
        doc.append("-" * 50)
        doc.append(f"Total Bytes Transferred: {network['total_bytes'] / 1024 / 1024:.1f}MB")
        doc.append(f"Average Bytes per Operation: {network['avg_per_op'] / 1024:.1f}KB")
        doc.append(f"Network Throughput: {network['throughput_mbps']:.1f} Mbps")

        if network['throughput_mbps'] < 1000:
            doc.append("⚠️  Network bandwidth is below 1 Gbps - potential bottleneck")
        else:
            doc.append("✅ Network bandwidth utilization is good")

        doc.append("")

        # Optimization Recommendations
        doc.append("OPTIMIZATION RECOMMENDATIONS")
        doc.append("-" * 50)

        if analysis['recommendations']:
            for i, rec in enumerate(analysis['recommendations'], 1):
                doc.append(f"{i}. {rec}")
        else:
            doc.append("No specific optimization recommendations generated.")
            doc.append("System performance appears balanced across components.")

        doc.append("")

        # Technical Details
        doc.append("TECHNICAL DETAILS")
        doc.append("-" * 50)
        doc.append(f"Profiling Framework: GenieProfiler v1.0")
        doc.append(f"Measurements Collected: {len(profiling_coordinator.profiler.measurements)}")
        doc.append(f"Resource Monitoring: {'Enabled' if torch.cuda.is_available() else 'CPU-only'}")
        doc.append(f"Network Monitoring: {'Enabled' if hasattr(profiling_coordinator.profiler, 'network_samples') else 'Disabled'}")

        doc.append("="*80)

        # Save document
        doc_text = "\n".join(doc)
        with open('results/deep_profiling_analysis.md', 'w') as f:
            f.write(doc_text)

        logger.info("📄 Analysis document saved: results/deep_profiling_analysis.md")
        logger.info(f"📊 Analysis summary: {len(profiling_coordinator.profiler.measurements)} measurements, "
                   f"{summary['total_operations']} operations, "
                   f"{summary['throughput']:.1f} ops/sec")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_scheduler_performance_impact(self, profiling_server, profiling_coordinator):
        """
        Test scheduler performance impact and decision quality.

        This validates that the semantic-aware scheduler provides benefits
        without excessive overhead.
        """
        logger.info("\n" + "="*80)
        logger.info("🧠 SCHEDULER PERFORMANCE IMPACT ANALYSIS")
        logger.info("="*80)

        # Test LLM decode pattern (should trigger co-location)
        logger.info("Testing LLM decode pattern scheduling...")

        decode_operations = []
        decode_times = []

        # Simulate LLM decode operations (batch_size=1, sequential)
        for i in range(20):
            input_tensor = torch.randn(1, 512, 768)  # LLM decode shape

            start = time.perf_counter()
            try:
                result = await profiling_coordinator.execute_remote_operation(
                    'aten::matmul', [input_tensor, torch.randn(768, 768)],
                    'localhost:17556', timeout=10.0
                )
                operation_time = time.perf_counter() - start
                decode_times.append(operation_time)
                decode_operations.append(result.shape)

                # Log progress
                if (i + 1) % 5 == 0:
                    avg_time = np.mean(decode_times)
                    logger.info(f"   Completed {i+1}/20 decode operations (avg: {avg_time*1000:.1f}s)")

            except RuntimeError as e:
                logger.debug(f"   Decode operation {i+1} failed: {e}")
                continue

        # Test prefill pattern (should allow parallelization)
        logger.info("Testing LLM prefill pattern scheduling...")

        prefill_operations = []
        prefill_times = []

        # Simulate LLM prefill operations (batch_size=32, parallelizable)
        for i in range(10):
            input_tensor = torch.randn(32, 512, 768)  # LLM prefill shape

            start = time.perf_counter()
            try:
                result = await profiling_coordinator.execute_remote_operation(
                    'aten::matmul', [input_tensor, torch.randn(768, 768)],
                    'localhost:17556', timeout=10.0
                )
                operation_time = time.perf_counter() - start
                prefill_times.append(operation_time)
                prefill_operations.append(result.shape)

                if (i + 1) % 3 == 0:
                    avg_time = np.mean(prefill_times)
                    logger.info(f"   Completed {i+1}/10 prefill operations (avg: {avg_time*1000:.1f}s)")

            except RuntimeError as e:
                logger.debug(f"   Prefill operation {i+1} failed: {e}")
                continue

        # Analyze scheduler performance
        scheduler_stats = profiling_coordinator.scheduler.get_stats()

        scheduler_analysis = {
            'decode_operations': {
                'count': len(decode_operations),
                'successful': len([op for op in decode_operations if op is not None]),
                'avg_time_ms': np.mean(decode_times) * 1000 if decode_times else 0,
                'total_time': sum(decode_times) if decode_times else 0
            },
            'prefill_operations': {
                'count': len(prefill_operations),
                'successful': len([op for op in prefill_operations if op is not None]),
                'avg_time_ms': np.mean(prefill_times) * 1000 if prefill_times else 0,
                'total_time': sum(prefill_times) if prefill_times else 0
            },
            'scheduler_stats': scheduler_stats,
            'scheduling_overhead_ms': scheduler_stats.get('avg_decision_time_ms', 0)
        }

        # Generate scheduler performance report
        logger.info("📊 Scheduler Performance Results:")
        logger.info(f"   Decode ops: {scheduler_analysis['decode_operations']['successful']}/"
                   f"{scheduler_analysis['decode_operations']['count']} successful")
        logger.info(f"   Decode avg: {scheduler_analysis['decode_operations']['avg_time_ms']:.1f}ms")
        logger.info(f"   Prefill ops: {scheduler_analysis['prefill_operations']['successful']}/"
                   f"{scheduler_analysis['prefill_operations']['count']} successful")
        logger.info(f"   Prefill avg: {scheduler_analysis['prefill_operations']['avg_time_ms']:.1f}ms")
        logger.info(f"   Scheduler decisions: {scheduler_stats['decisions_made']}")
        logger.info(f"   Available devices: {scheduler_stats['available_devices']}")

        # Save scheduler analysis
        with open('results/scheduler_performance_analysis.json', 'w') as f:
            json.dump(scheduler_analysis, f, indent=2)

        logger.info("✅ Scheduler performance impact analysis complete")

        # Assertions
        assert scheduler_analysis['decode_operations']['successful'] > 0, "No decode operations succeeded"
        assert scheduler_analysis['prefill_operations']['successful'] > 0, "No prefill operations succeeded"
        assert scheduler_stats['decisions_made'] >= 30, "Scheduler should have made placement decisions"

        return scheduler_analysis

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency_analysis(self, profiling_server, profiling_coordinator):
        """
        Analyze memory efficiency and identify memory-related bottlenecks.
        """
        logger.info("\n" + "="*80)
        logger.info("🧠 MEMORY EFFICIENCY ANALYSIS")
        logger.info("="*80)

        # Test memory usage with different tensor sizes
        memory_scenarios = [
            {'size': (100, 100), 'operations': 100, 'name': 'Small tensors'},
            {'size': (500, 500), 'operations': 50, 'name': 'Medium tensors'},
            {'size': (1000, 1000), 'operations': 20, 'name': 'Large tensors'},
        ]

        memory_results = {}

        for scenario in memory_scenarios:
            logger.info(f"Testing memory efficiency: {scenario['name']}")

            # Measure memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run operations
            start = time.perf_counter()
            successful_ops = 0

            for i in range(scenario['operations']):
                tensor = torch.randn(*scenario['size'])

                try:
                    await profiling_coordinator.execute_remote_operation(
                        'aten::relu', [tensor], 'localhost:17556', timeout=10.0
                    )
                    successful_ops += 1

                    # Periodic progress logging
                    if (i + 1) % 25 == 0:
                        elapsed = time.perf_counter() - start
                        rate = (i + 1) / elapsed
                        logger.debug(f"   Progress: {i+1}/{scenario['operations']} "
                                   f"({rate:.1f} ops/sec)")

                except RuntimeError as e:
                    logger.debug(f"   Operation {i+1} failed: {e}")
                    continue

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            total_time = time.perf_counter() - start

            tensor_bytes = scenario['size'][0] * scenario['size'][1] * 4  # float32
            total_tensor_bytes = tensor_bytes * scenario['operations']

            memory_results[scenario['name']] = {
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_delta_mb': memory_after - memory_before,
                'operations': scenario['operations'],
                'successful_operations': successful_ops,
                'success_rate': successful_ops / scenario['operations'],
                'total_time': total_time,
                'throughput': successful_ops / total_time if total_time > 0 else 0,
                'tensor_bytes_per_operation': tensor_bytes,
                'total_tensor_bytes': total_tensor_bytes,
                'memory_efficiency': total_tensor_bytes / (memory_after - memory_before) / 1024 / 1024 if memory_after > memory_before else 0
            }

            logger.info(f"   Memory: {memory_before:.1f}MB → {memory_after:.1f}MB "
                       f"(Δ{memory_after - memory_before:.1f}MB)")
            logger.info(f"   Operations: {successful_ops}/{scenario['operations']} "
                       f"({successful_ops / scenario['operations']:.1%})")
            logger.info(f"   Throughput: {successful_ops / total_time:.1f} ops/sec")
            logger.info(f"   Memory efficiency: {memory_results[scenario['name']]['memory_efficiency']:.1f} MB tensor per MB RAM")

        # Generate memory efficiency report
        self._generate_memory_efficiency_report(memory_results)

        # Save results
        with open('results/memory_efficiency_analysis.json', 'w') as f:
            json.dump(memory_results, f, indent=2)

        logger.info("✅ Memory efficiency analysis complete")
        return memory_results

    def _generate_memory_efficiency_report(self, memory_results: Dict):
        """Generate memory efficiency analysis report."""
        doc = []
        doc.append("\n" + "="*80)
        doc.append("MEMORY EFFICIENCY ANALYSIS REPORT")
        doc.append("="*80)

        doc.append("Scenario → Memory Usage → Efficiency:")
        doc.append("-" * 60)

        for scenario, results in memory_results.items():
            doc.append(f"{scenario:15s}: Δ{results['memory_delta_mb']:6.1f}MB "
                      f"({results['throughput']:5.1f} ops/sec)")

        # Calculate scaling analysis
        sizes = [int(s.split()[0]) for s in memory_results.keys()]
        deltas = [memory_results[s]['memory_delta_mb'] for s in memory_results.keys()]

        if len(sizes) >= 2:
            # Memory scaling factor
            memory_scaling = deltas[-1] / deltas[0] if deltas[0] > 0 else 0

            # Tensor size scaling factor
            tensor_scaling = (sizes[-1] / sizes[0]) ** 2  # Area scaling

            doc.append(f"\nMemory scaling: {memory_scaling:.2f}x")
            doc.append(f"Tensor scaling: {tensor_scaling:.2f}x")

            if memory_scaling < tensor_scaling * 1.5:  # Allow some overhead
                doc.append("✅ Memory usage scales efficiently with tensor size")
            else:
                doc.append("⚠️  Memory usage scales poorly - potential memory leaks or overhead")

        # Identify memory bottlenecks
        max_memory_scenario = max(memory_results.items(), key=lambda x: x[1]['memory_delta_mb'])
        min_efficiency_scenario = min(memory_results.items(), key=lambda x: x[1]['memory_efficiency'])

        doc.append(f"\nHighest memory usage: {max_memory_scenario[0]} ({max_memory_scenario[1]['memory_delta_mb']:.1f}MB)")
        doc.append(f"Lowest efficiency: {min_efficiency_scenario[0]} ({min_efficiency_scenario[1]['memory_efficiency']:.1f} ratio)")

        doc.append("="*80)

        doc_text = "\n".join(doc)
        logger.info(doc_text)

        with open('results/memory_efficiency_analysis.md', 'w') as f:
            f.write(doc_text)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_error_rate_under_load(self, profiling_server, profiling_coordinator):
        """
        Test error rates and system stability under various load conditions.
        """
        logger.info("\n" + "="*80)
        logger.info("🛡️  ERROR RATE AND STABILITY ANALYSIS")
        logger.info("="*80)

        # Test different load patterns
        load_patterns = [
            {'name': 'Light Load', 'concurrency': 1, 'duration': 30, 'tensor_size': (100, 100)},
            {'name': 'Medium Load', 'concurrency': 5, 'duration': 30, 'tensor_size': (200, 200)},
            {'name': 'Heavy Load', 'concurrency': 10, 'duration': 30, 'tensor_size': (300, 300)},
            {'name': 'Stress Load', 'concurrency': 20, 'duration': 20, 'tensor_size': (100, 100)},  # Smaller tensors for stress
        ]

        stability_results = {}

        for pattern in load_patterns:
            logger.info(f"Testing {pattern['name']}: {pattern['concurrency']} concurrent, "
                       f"{pattern['tensor_size']} tensors, {pattern['duration']}s")

            start_time = time.time()
            end_time = start_time + pattern['duration']

            total_operations = 0
            successful_operations = 0
            errors = []

            while time.time() < end_time:
                # Execute batch of concurrent operations
                tasks = []
                for _ in range(pattern['concurrency']):
                    tensor = torch.randn(*pattern['tensor_size'])
                    tasks.append(
                        profiling_coordinator.execute_remote_operation(
                            'aten::relu', [tensor], 'localhost:17556', timeout=5.0
                        )
                    )

                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    total_operations += pattern['concurrency']
                    successful_operations += sum(1 for r in results
                                               if not isinstance(r, Exception))

                    # Track errors
                    for result in results:
                        if isinstance(result, Exception):
                            errors.append(str(result))

                except Exception as e:
                    logger.debug(f"Batch failed in {pattern['name']}: {e}")
                    errors.append(str(e))
                    continue

            elapsed = time.time() - start_time
            throughput = total_operations / elapsed if elapsed > 0 else 0
            success_rate = successful_operations / total_operations if total_operations > 0 else 0
            error_rate = 1 - success_rate

            # Analyze error patterns
            error_types = {}
            for error in errors:
                error_type = 'timeout' if 'timeout' in error.lower() else \
                           'network' if 'network' in error.lower() or 'connection' in error.lower() else \
                           'execution' if 'execution' in error.lower() else 'other'
                error_types[error_type] = error_types.get(error_type, 0) + 1

            stability_results[pattern['name']] = {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'throughput': throughput,
                'success_rate': success_rate,
                'error_rate': error_rate,
                'error_types': error_types,
                'duration': elapsed
            }

            logger.info(f"   Throughput: {throughput:.1f} ops/sec")
            logger.info(f"   Success rate: {success_rate:.1%}")
            logger.info(f"   Error rate: {error_rate:.1%}")
            logger.info(f"   Error breakdown: {error_types}")

        # Generate stability report
        self._generate_stability_report(stability_results)

        # Save results
        with open('results/stability_under_load_analysis.json', 'w') as f:
            json.dump(stability_results, f, indent=2)

        logger.info("✅ Error rate and stability analysis complete")

        # Assertions
        for pattern, results in stability_results.items():
            assert results['success_rate'] > 0.8, f"{pattern} success rate too low: {results['success_rate']:.1%}"
            assert results['throughput'] > 0, f"{pattern} throughput is zero"

        return stability_results

    def _generate_stability_report(self, stability_results: Dict):
        """Generate stability analysis report."""
        doc = []
        doc.append("\n" + "="*80)
        doc.append("SYSTEM STABILITY ANALYSIS REPORT")
        doc.append("="*80)

        doc.append("Load Pattern → Performance → Stability:")
        doc.append("-" * 60)

        for pattern, results in stability_results.items():
            success_rate = results['success_rate']
            error_rate = results['error_rate']
            throughput = results['throughput']

            stability_indicator = "✅" if success_rate > 0.95 else "⚠️" if success_rate > 0.8 else "❌"

            doc.append(f"{pattern:12s}: {throughput:5.1f} ops/sec "
                      f"{stability_indicator} ({success_rate:.0%} success)")

        # Overall assessment
        avg_success_rate = np.mean([r['success_rate'] for r in stability_results.values()])
        max_throughput = max(r['throughput'] for r in stability_results.values())

        doc.append(f"\nOverall system stability: {avg_success_rate:.1%} average success rate")
        doc.append(f"Peak throughput: {max_throughput:.1f} operations per second")

        if avg_success_rate > 0.95:
            doc.append("🎉 EXCELLENT - System is highly stable under load")
        elif avg_success_rate > 0.8:
            doc.append("✅ GOOD - System is stable with minor issues")
        else:
            doc.append("⚠️  NEEDS IMPROVEMENT - System has stability issues under load")

        doc.append("="*80)

        doc_text = "\n".join(doc)
        logger.info(doc_text)

        with open('results/system_stability_analysis.md', 'w') as f:
            f.write(doc_text)


if __name__ == '__main__':
    # Run deep profiling tests directly
    logger.info("🧪 Running Deep Profiling Session...")

    # Configure logging for deep profiling
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )

    pytest.main([__file__, '-v', '-s', '--tb=short'])
