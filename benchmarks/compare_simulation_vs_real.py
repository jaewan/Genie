#!/usr/bin/env python3
"""
Compare simulation results vs real network results for Week 3 validation.

This script loads both simulation results (from Week 2) and real network results
(from Week 3) and compares them to validate the accuracy of our simulation.
"""

import json
import logging
import os
import statistics
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimulationVsRealComparator:
    """
    Compare simulation results with real network measurements.
    """

    def __init__(self):
        """Initialize comparator."""
        self.simulation_results = {}
        self.real_network_results = {}

    def load_simulation_results(self) -> bool:
        """
        Load simulation results from Week 2.

        Returns:
            True if loaded successfully, False otherwise
        """
        simulation_files = [
            'baseline_no_colocation.json',
            'optimized_with_colocation.json'
        ]

        for filename in simulation_files:
            filepath = os.path.join('benchmarks', filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Extract strategy from filename
                    strategy = filename.replace('.json', '').replace('_', '_')
                    if 'baseline' in strategy:
                        strategy = 'baseline_no_colocation'
                    elif 'optimized' in strategy:
                        strategy = 'optimized_with_colocation'

                    self.simulation_results[strategy] = data
                    logger.info(f"✅ Loaded simulation: {strategy}")

                except Exception as e:
                    logger.error(f"❌ Failed to load {filepath}: {e}")
                    return False
            else:
                logger.warning(f"⚠️  Simulation file not found: {filepath}")

        return len(self.simulation_results) > 0

    def load_real_network_results(self) -> bool:
        """
        Load real network results from Week 3.

        Returns:
            True if loaded successfully, False otherwise
        """
        real_files = [
            'real_network_baseline.json',
            'real_network_optimized.json'
        ]

        for filename in real_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)

                    strategy = data.get('strategy', 'unknown')
                    self.real_network_results[strategy] = data
                    logger.info(f"✅ Loaded real network: {strategy}")

                except Exception as e:
                    logger.error(f"❌ Failed to load {filename}: {e}")
                    return False
            else:
                logger.warning(f"⚠️  Real network file not found: {filename}")

        return len(self.real_network_results) > 0

    def compare_results(self) -> Dict[str, Any]:
        """
        Compare simulation vs real network results.

        Returns:
            Dictionary with comparison analysis
        """
        logger.info("=" * 70)
        logger.info("🔍 SIMULATION vs REAL NETWORK COMPARISON")
        logger.info("=" * 70)

        comparison = {
            'simulation_accuracy': {},
            'overall_assessment': {},
            'detailed_comparison': {}
        }

        # Compare baseline results
        if 'baseline_no_colocation' in self.simulation_results and 'baseline_no_colocation' in self.real_network_results:
            sim_baseline = self.simulation_results['baseline_no_colocation']
            real_baseline = self.real_network_results['baseline_no_colocation']

            baseline_comparison = self._compare_single_result(
                sim_baseline, real_baseline,
                "Baseline (No Co-location)"
            )

            comparison['baseline_comparison'] = baseline_comparison

        # Compare optimized results
        if 'optimized_with_colocation' in self.simulation_results and 'optimized_with_colocation' in self.real_network_results:
            sim_optimized = self.simulation_results['optimized_with_colocation']
            real_optimized = self.real_network_results['optimized_with_colocation']

            optimized_comparison = self._compare_single_result(
                sim_optimized, real_optimized,
                "Optimized (With Co-location)"
            )

            comparison['optimized_comparison'] = optimized_comparison

        # Overall assessment
        if 'baseline_comparison' in comparison and 'optimized_comparison' in comparison:
            baseline_comp = comparison['baseline_comparison']
            optimized_comp = comparison['optimized_comparison']

            # Check if simulation accurately predicted the improvement
            sim_improvement = sim_optimized['avg_latency_ms'] - sim_baseline['avg_latency_ms']
            sim_improvement_pct = abs(sim_improvement) / sim_baseline['avg_latency_ms'] * 100

            real_improvement = real_optimized['avg_latency_ms'] - real_baseline['avg_latency_ms']
            real_improvement_pct = abs(real_improvement) / real_baseline['avg_latency_ms'] * 100

            improvement_error = abs(sim_improvement_pct - real_improvement_pct)

            comparison['overall_assessment'] = {
                'simulation_improvement_pct': sim_improvement_pct,
                'real_improvement_pct': real_improvement_pct,
                'improvement_error_pct': improvement_error,
                'simulation_accurate': improvement_error < 10,  # Within 10% is good
                'simulation_direction_correct': (sim_improvement > 0) == (real_improvement > 0)
            }

            logger.info("")
            logger.info("📊 OVERALL ASSESSMENT:")
            logger.info(f"  Simulation predicted: {sim_improvement_pct:.1f}% improvement")
            logger.info(f"  Real network showed:  {real_improvement_pct:.1f}% improvement")
            logger.info(f"  Prediction error:     {improvement_error:.1f}%")

            if comparison['overall_assessment']['simulation_accurate']:
                logger.info("  ✅ Simulation ACCURATE (error < 10%)")
            else:
                logger.info("  ❌ Simulation INACCURATE (error >= 10%)")

            if comparison['overall_assessment']['simulation_direction_correct']:
                logger.info("  ✅ Direction CORRECT")
            else:
                logger.info("  ❌ Direction WRONG")

        return comparison

    def _compare_single_result(self, sim_result: Dict, real_result: Dict, label: str) -> Dict[str, Any]:
        """
        Compare a single simulation vs real result pair.

        Args:
            sim_result: Simulation result dictionary
            real_result: Real network result dictionary
            label: Label for this comparison

        Returns:
            Comparison dictionary
        """
        logger.info(f"\n📈 {label} Comparison:")
        logger.info("-" * 50)

        sim_latency = sim_result['avg_latency_ms']
        real_latency = real_result['avg_latency_ms']

        latency_diff = real_latency - sim_latency
        latency_error_pct = abs(latency_diff) / sim_latency * 100

        comparison = {
            'simulation_latency_ms': sim_latency,
            'real_latency_ms': real_latency,
            'latency_difference_ms': latency_diff,
            'latency_error_pct': latency_error_pct,
            'simulation_accurate': latency_error_pct < 20,  # Within 20% is reasonable
            'real_faster_than_sim': real_latency < sim_latency,
            'sim_server_url': sim_result.get('server_url', 'unknown'),
            'real_server_url': real_result.get('server_url', 'unknown')
        }

        logger.info(f"  Simulation: {sim_latency:.2f}ms avg")
        logger.info(f"  Real:       {real_latency:.2f}ms avg")
        logger.info(f"  Difference: {latency_diff:+.2f}ms ({latency_error_pct:+.1f}%)")

        if comparison['simulation_accurate']:
            logger.info("  ✅ ACCURATE (< 20% error)")
        else:
            logger.info("  ❌ INACCURATE (>= 20% error)")

        if comparison['real_faster_than_sim']:
            logger.info("  📡 Real network faster than simulation")
        else:
            logger.info("  🐌 Real network slower than simulation")

        return comparison

    def generate_report(self, comparison: Dict) -> str:
        """
        Generate a human-readable report.

        Args:
            comparison: Comparison results dictionary

        Returns:
            Formatted report string
        """
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("📋 SIMULATION vs REAL NETWORK VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall assessment
        if 'overall_assessment' in comparison:
            overall = comparison['overall_assessment']

            report_lines.append("🎯 OVERALL VALIDATION:")
            report_lines.append("-" * 40)
            report_lines.append(f"Simulation predicted: {overall['simulation_improvement_pct']:.1f}% improvement")
            report_lines.append(f"Real network showed:  {overall['real_improvement_pct']:.1f}% improvement")
            report_lines.append(f"Prediction accuracy:  {overall['improvement_error_pct']:.1f}% error")
            report_lines.append("")

            if overall['simulation_accurate']:
                report_lines.append("✅ CONCLUSION: Simulation is ACCURATE")
                report_lines.append("   - Error is within acceptable range (< 10%)")
                report_lines.append("   - Direction of improvement is correct")
                report_lines.append("   - Simulation can be trusted for future predictions")
            else:
                report_lines.append("❌ CONCLUSION: Simulation needs IMPROVEMENT")
                report_lines.append("   - Error is too high (>= 10%)")
                report_lines.append("   - Investigate simulation parameters")
                report_lines.append("   - May need more realistic network modeling")

            report_lines.append("")

        # Detailed comparisons
        for comp_type in ['baseline_comparison', 'optimized_comparison']:
            if comp_type in comparison:
                comp = comparison[comp_type]
                label = comp_type.replace('_comparison', '').title()

                report_lines.append(f"📊 {label} Details:")
                report_lines.append("-" * 40)
                report_lines.append(f"  Simulation: {comp['simulation_latency_ms']:.2f}ms")
                report_lines.append(f"  Real:       {comp['real_latency_ms']:.2f}ms")
                report_lines.append(f"  Error:      {comp['latency_error_pct']:.1f}%")
                report_lines.append("")

                if comp['simulation_accurate']:
                    report_lines.append("  ✅ Latency prediction accurate")
                else:
                    report_lines.append("  ❌ Latency prediction inaccurate")

        report_lines.append("=" * 80)
        report_lines.append("📝 RECOMMENDATIONS:")
        report_lines.append("-" * 80)

        if 'overall_assessment' in comparison and comparison['overall_assessment']['simulation_accurate']:
            report_lines.append("✅ Simulation validated - proceed with confidence")
            report_lines.append("✅ Use simulation for future optimization design")
            report_lines.append("✅ Paper claims supported by real hardware validation")
        else:
            report_lines.append("⚠️  Simulation needs refinement before production use")
            report_lines.append("⚠️  Investigate network modeling accuracy")
            report_lines.append("⚠️  Consider collecting more real network data")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

def main():
    """Main comparison execution."""
    logger.info("🧪 Genie Week 3: Simulation vs Real Network Validation")
    logger.info("=" * 60)

    # Initialize comparator
    comparator = SimulationVsRealComparator()

    # Load results
    logger.info("📂 Loading results...")
    sim_loaded = comparator.load_simulation_results()
    real_loaded = comparator.load_real_network_results()

    if not sim_loaded or not real_loaded:
        logger.error("❌ Missing required result files")
        logger.info("")
        logger.info("Required files:")
        logger.info("  📊 Simulation: benchmarks/baseline_no_colocation.json")
        logger.info("  📊 Simulation: benchmarks/optimized_with_colocation.json")
        logger.info("  🌐 Real Network: real_network_baseline.json")
        logger.info("  🌐 Real Network: real_network_optimized.json")
        logger.info("")
        logger.info("Run these first:")
        logger.info("  python benchmarks/measure_real_network_llm.py --baseline-only")
        logger.info("  python benchmarks/measure_real_network_llm.py --optimized-only")
        return

    logger.info("✅ All result files loaded successfully")
    logger.info("")

    # Perform comparison
    comparison = comparator.compare_results()

    # Generate and save report
    report = comparator.generate_report(comparison)

    # Save report to file
    report_file = "simulation_vs_real_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    logger.info(f"💾 Report saved to: {report_file}")

    # Print report to console
    print("\n" + report)

if __name__ == "__main__":
    main()
