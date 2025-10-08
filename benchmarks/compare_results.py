"""
Compare baseline vs optimized results.
File: benchmarks/compare_results.py
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("📊 Performance Comparison: Baseline vs Optimized")
    logger.info("=" * 70)
    logger.info("")

    # Load results
    with open('baseline_no_colocation.json', 'r') as f:
        baseline = json.load(f)

    with open('optimized_with_colocation.json', 'r') as f:
        optimized = json.load(f)

    # Extract metrics
    baseline_avg = baseline['avg_latency_ms']
    optimized_avg = optimized['avg_latency_ms']

    baseline_total = baseline['total_ms']
    optimized_total = optimized['total_ms']

    # Calculate improvement
    latency_reduction = baseline_avg - optimized_avg
    latency_reduction_pct = (latency_reduction / baseline_avg) * 100

    total_reduction = baseline_total - optimized_total
    total_reduction_pct = (total_reduction / baseline_total) * 100

    # Display results
    logger.info("Results:")
    logger.info("")
    logger.info(f"{'Metric':<30} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
    logger.info("-" * 70)
    logger.info(
        f"{'Avg Latency (ms/step)':<30} "
        f"{baseline_avg:<15.2f} "
        f"{optimized_avg:<15.2f} "
        f"{latency_reduction:.2f} ms ({latency_reduction_pct:.1f}%)"
    )
    logger.info(
        f"{'Total Time (ms)':<30} "
        f"{baseline_total:<15.2f} "
        f"{optimized_total:<15.2f} "
        f"{total_reduction:.2f} ms ({total_reduction_pct:.1f}%)"
    )
    logger.info("")

    # Analysis
    logger.info("Analysis:")
    logger.info(f"  • Co-location reduces latency by {latency_reduction_pct:.1f}%")
    logger.info(f"  • Per-step savings: {latency_reduction:.2f}ms")
    logger.info(f"  • Total savings (10 steps): {total_reduction:.2f}ms")
    logger.info("")

    # Create visualization
    logger.info("Visual Comparison:")
    logger.info("")

    baseline_bar = "█" * int(baseline_avg)
    optimized_bar = "█" * int(optimized_avg)

    logger.info(f"Baseline:  {baseline_bar} {baseline_avg:.2f}ms")
    logger.info(f"Optimized: {optimized_bar} {optimized_avg:.2f}ms")
    logger.info("")

    # Conclusion
    if latency_reduction_pct >= 30:
        logger.info("✅ EXCELLENT: >30% improvement - semantic optimization works!")
    elif latency_reduction_pct >= 15:
        logger.info("✅ GOOD: >15% improvement - semantic optimization helps!")
    elif latency_reduction_pct >= 10:
        logger.info("⚠️  OK: >10% improvement - semantic optimization has some benefit")
    else:
        logger.info("❌ POOR: <10% improvement - need to investigate")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Comparison complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
