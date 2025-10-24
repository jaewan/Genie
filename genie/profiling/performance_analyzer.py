"""
Performance analyzer for Genie profiling data.

Provides high-level analysis, visualization, and optimization recommendations
based on collected profiling measurements.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes profiling data to identify performance bottlenecks and optimization opportunities.

    Provides:
    - Bottleneck identification with severity levels
    - Optimization recommendations with expected impact
    - Performance regression detection
    - Comparative analysis between different configurations
    """

    def __init__(self, profiler):
        self.profiler = profiler
        self.baselines = {}

    def load_baseline(self, filename: str):
        """Load baseline performance data for comparison."""
        try:
            with open(filename, 'r') as f:
                self.baselines = json.load(f)
            logger.info(f"Baseline loaded from {filename}")
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")

    def save_baseline(self, filename: str, tag: str = "baseline"):
        """Save current measurements as baseline."""
        try:
            self.profiler.save_report(filename)
            logger.info(f"Baseline saved to {filename} with tag {tag}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")

    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Compare current performance with baseline."""
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)

            if not self.profiler.measurements:
                return {'error': 'No current measurements to compare'}

            current_analysis = self.profiler.get_bottleneck_analysis()
            baseline_measurements = baseline_data.get('measurements', [])

            if not baseline_measurements:
                return {'error': 'No baseline measurements found'}

            # Calculate performance regression/improvement
            current_total = sum(m['total_latency'] for m in self.profiler.measurements)
            baseline_total = sum(m['total_latency'] for m in baseline_measurements)

            comparison = {
                'total_latency_change': ((current_total - baseline_total) / baseline_total) * 100,
                'operation_count_current': len(self.profiler.measurements),
                'operation_count_baseline': len(baseline_measurements),
                'component_changes': {},
                'recommendations': []
            }

            # Compare component timings
            current_components = {}
            baseline_components = {}

            for m in self.profiler.measurements:
                for component, duration in m.get('timings', {}).items():
                    current_components[component] = current_components.get(component, 0) + duration

            for m in baseline_measurements:
                for component, duration in m.get('timings', {}).items():
                    baseline_components[component] = baseline_components.get(component, 0) + duration

            for component in set(current_components.keys()) | set(baseline_components.keys()):
                current_time = current_components.get(component, 0)
                baseline_time = baseline_components.get(component, 0)

                if baseline_time > 0:
                    change = ((current_time - baseline_time) / baseline_time) * 100
                    comparison['component_changes'][component] = change

                    if change > 20:  # Significant degradation
                        comparison['recommendations'].append(f"{component} degraded by {change:.1f}%")
                    elif change < -20:  # Significant improvement
                        comparison['recommendations'].append(f"{component} improved by {abs(change):.1f}%")

            return comparison

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {'error': str(e)}

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks with severity levels."""
        if not self.profiler.measurements:
            return {'error': 'No measurements available'}

        analysis = self.profiler.get_bottleneck_analysis()

        bottlenecks = {
            'critical': [],  # >50% of total time
            'high': [],      # 20-50% of total time
            'medium': [],    # 10-20% of total time
            'low': []        # <10% of total time
        }

        for component, data in analysis['component_bottlenecks'].items():
            percentage = data['percentage']

            if percentage > 50:
                bottlenecks['critical'].append({
                    'component': component,
                    'percentage': percentage,
                    'time_ms': data['time_ms'],
                    'impact': 'Very High - Major bottleneck'
                })
            elif percentage > 20:
                bottlenecks['high'].append({
                    'component': component,
                    'percentage': percentage,
                    'time_ms': data['time_ms'],
                    'impact': 'High - Significant performance impact'
                })
            elif percentage > 10:
                bottlenecks['medium'].append({
                    'component': component,
                    'percentage': percentage,
                    'time_ms': data['time_ms'],
                    'impact': 'Medium - Moderate optimization opportunity'
                })
            else:
                bottlenecks['low'].append({
                    'component': component,
                    'percentage': percentage,
                    'time_ms': data['time_ms'],
                    'impact': 'Low - Minor optimization opportunity'
                })

        return {
            'bottlenecks': bottlenecks,
            'network_analysis': analysis['network_analysis'],
            'overall_recommendations': analysis['recommendations']
        }

    def generate_optimization_plan(self) -> Dict[str, Any]:
        """Generate prioritized optimization plan based on profiling data."""
        bottlenecks = self.identify_bottlenecks()

        optimization_plan = {
            'priority_order': [],
            'estimated_impact': {},
            'implementation_effort': {},
            'detailed_recommendations': []
        }

        # Prioritize based on impact
        all_bottlenecks = (
            bottlenecks['bottlenecks']['critical'] +
            bottlenecks['bottlenecks']['high'] +
            bottlenecks['bottlenecks']['medium'] +
            bottlenecks['bottlenecks']['low']
        )

        # Sort by percentage impact (descending)
        all_bottlenecks.sort(key=lambda x: x['percentage'], reverse=True)

        for i, bottleneck in enumerate(all_bottlenecks):
            optimization_plan['priority_order'].append({
                'rank': i + 1,
                'component': bottleneck['component'],
                'percentage': bottleneck['percentage'],
                'time_ms': bottleneck['time_ms'],
                'impact': bottleneck['impact']
            })

            # Estimate implementation effort (rough heuristic)
            effort = self._estimate_implementation_effort(bottleneck['component'])
            optimization_plan['implementation_effort'][bottleneck['component']] = effort

            # Estimate impact
            potential_improvement = self._estimate_optimization_impact(bottleneck['component'], bottleneck['percentage'])
            optimization_plan['estimated_impact'][bottleneck['component']] = potential_improvement

            # Generate detailed recommendation
            recommendation = self._generate_detailed_recommendation(
                bottleneck['component'],
                bottleneck['percentage'],
                potential_improvement,
                effort
            )
            optimization_plan['detailed_recommendations'].append(recommendation)

        return optimization_plan

    def _estimate_implementation_effort(self, component: str) -> str:
        """Estimate implementation effort for optimizing a component."""
        effort_map = {
            'network_send': 'High - Requires transport layer changes',
            'serialize': 'Medium - May need zero-copy implementation',
            'wait_result': 'Medium - Server-side optimization',
            'scheduler_time': 'Low - Algorithm optimization',
            'deserialize': 'Low - Minor protocol changes',
            'queue_wait_time': 'Medium - Connection pooling improvements'
        }

        return effort_map.get(component, 'Medium - Unknown component')

    def _estimate_optimization_impact(self, component: str, current_percentage: float) -> str:
        """Estimate potential performance improvement for a component."""
        # Rough estimates based on component type
        impact_map = {
            'network_send': 'High - 5-10x improvement possible with DPDK',
            'serialize': 'Medium - 2-3x improvement with zero-copy',
            'wait_result': 'Medium - 1.5-2x improvement with server optimization',
            'scheduler_time': 'Low - 1.2-1.5x improvement with better algorithms',
            'deserialize': 'Low - 1.1-1.3x improvement',
            'queue_wait_time': 'Medium - 2-3x improvement with better pooling'
        }

        return impact_map.get(component, 'Medium - Unknown impact')

    def _generate_detailed_recommendation(self, component: str, percentage: float,
                                        impact: str, effort: str) -> Dict[str, Any]:
        """Generate detailed optimization recommendation."""
        recommendations = {
            'network_send': {
                'title': 'Implement DPDK Zero-Copy Transport',
                'description': 'Replace TCP transport with DPDK for GPU-to-GPU zero-copy transfers',
                'implementation_steps': [
                    'Set up DPDK with GPUDev support',
                    'Implement zero-copy tensor serialization',
                    'Add GPUDirect RDMA support',
                    'Update connection management'
                ],
                'expected_improvement': '5-10x reduction in network latency',
                'risk_level': 'High - Requires system-level changes'
            },
            'serialize': {
                'title': 'Optimize Tensor Serialization',
                'description': 'Implement zero-copy tensor transfer and GPU memory mapping',
                'implementation_steps': [
                    'Use pinned GPU memory for transfers',
                    'Implement CUDA IPC for same-node transfers',
                    'Add memory registration caching',
                    'Optimize tensor metadata encoding'
                ],
                'expected_improvement': '2-3x reduction in serialization time',
                'risk_level': 'Medium - Memory management complexity'
            },
            'wait_result': {
                'title': 'Optimize Server-Side Execution',
                'description': 'Improve server execution efficiency and parallel processing',
                'implementation_steps': [
                    'Implement CUDA streams for overlapping execution',
                    'Add operation batching for small operations',
                    'Optimize GPU kernel selection',
                    'Add result prefetching'
                ],
                'expected_improvement': '1.5-2x reduction in execution time',
                'risk_level': 'Medium - Requires server-side changes'
            },
            'scheduler_time': {
                'title': 'Optimize Scheduler Algorithms',
                'description': 'Improve placement decision speed and accuracy',
                'implementation_steps': [
                    'Cache placement decisions',
                    'Use pre-computed cost models',
                    'Implement incremental scheduling',
                    'Add workload prediction'
                ],
                'expected_improvement': '1.2-1.5x reduction in scheduling overhead',
                'risk_level': 'Low - Algorithm optimization only'
            }
        }

        return recommendations.get(component, {
            'title': f'Optimize {component}',
            'description': f'Generic optimization for {component} component',
            'implementation_steps': ['Analyze component behavior', 'Identify optimization opportunities', 'Implement improvements'],
            'expected_improvement': 'Unknown - requires analysis',
            'risk_level': 'Medium'
        })

    def generate_performance_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for performance dashboard visualization."""
        if not self.profiler.measurements:
            return {'error': 'No measurements available'}

        analysis = self.identify_bottlenecks()

        # Prepare data for visualization
        dashboard_data = {
            'summary': {
                'total_operations': len(self.profiler.measurements),
                'avg_latency_ms': np.mean([m['total_latency'] * 1000 for m in self.profiler.measurements]),
                'p95_latency_ms': np.percentile([m['total_latency'] * 1000 for m in self.profiler.measurements], 95),
                'throughput_ops_per_sec': len(self.profiler.measurements) / max(0.001, sum(m['total_latency'] for m in self.profiler.measurements))
            },

            'component_breakdown': [
                {
                    'component': component,
                    'percentage': data['percentage'],
                    'time_ms': data['time_ms'],
                    'severity': 'critical' if data['percentage'] > 50 else
                               'high' if data['percentage'] > 20 else
                               'medium' if data['percentage'] > 10 else 'low'
                }
                for component, data in analysis['component_bottlenecks'].items()
            ],

            'network_metrics': {
                'total_bytes': analysis['network_analysis']['total_bytes'],
                'avg_bytes_per_op': analysis['network_analysis']['avg_per_op'],
                'throughput_mbps': analysis['network_analysis']['throughput_mbps']
            },

            'optimization_opportunities': [
                {
                    'rank': i + 1,
                    'component': rec['component'],
                    'percentage': rec['percentage'],
                    'time_ms': rec['time_ms'],
                    'impact': rec['impact']
                }
                for i, rec in enumerate(self.generate_optimization_plan()['priority_order'][:5])
            ]
        }

        return dashboard_data

    def export_to_json(self, filename: str):
        """Export analysis results to JSON file."""
        try:
            analysis = {
                'bottleneck_analysis': self.identify_bottlenecks(),
                'optimization_plan': self.generate_optimization_plan(),
                'dashboard_data': self.generate_performance_dashboard_data(),
                'timestamp': time.time(),
                'measurements_count': len(self.profiler.measurements)
            }

            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)

            logger.info(f"Performance analysis exported to {filename}")

        except Exception as e:
            logger.error(f"Failed to export analysis: {e}")
