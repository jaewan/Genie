#!/usr/bin/env python3
"""
Result parser for Genie multi-node testing.
Parses JSON result files and generates analysis and visualizations.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

def load_json_file(filepath):
    """Load and parse JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_server_results(server_data):
    """Analyze server-side test results."""
    if not server_data:
        return {"error": "No server data available"}

    tests = server_data.get('tests', [])
    passed = len([t for t in tests if t.get('status') == 'PASS'])
    failed = len([t for t in tests if t.get('status') == 'FAIL'])

    # Extract key metrics
    gpu_test = next((t for t in tests if t.get('name') == 'gpu_detection'), None)
    server_ops_test = next((t for t in tests if t.get('name') == 'server_operations'), None)

    analysis = {
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(tests) * 100 if tests else 0,
        "gpu_detected": gpu_test and gpu_test.get('status') == 'PASS',
        "server_operations_working": server_ops_test and server_ops_test.get('status') == 'PASS'
    }

    if gpu_test and gpu_test.get('details'):
        details = gpu_test['details']
        if isinstance(details, str):
            import ast
            try:
                details = ast.literal_eval(details)
            except:
                details = {}
        analysis.update({
            "gpu_name": details.get('gpu_name', 'Unknown'),
            "cuda_available": details.get('cuda_available', False),
            "gpu_time_ms": details.get('gpu_time_ms', 0)
        })

    return analysis

def analyze_client_results(client_data):
    """Analyze client-side test results."""
    if not client_data:
        return {"error": "No client data available"}

    tests = client_data.get('tests', [])
    passed = len([t for t in tests if t.get('status') == 'PASS'])
    failed = len([t for t in tests if t.get('status') == 'FAIL'])

    # Extract key metrics
    network_test = next((t for t in tests if t.get('name') == 'network_connectivity'), None)
    remote_test = next((t for t in tests if t.get('name') == 'remote_execution'), None)
    perf_test = next((t for t in tests if t.get('name') == 'performance_comparison'), None)

    analysis = {
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(tests) * 100 if tests else 0,
        "network_working": network_test and network_test.get('status') == 'PASS',
        "remote_execution_working": remote_test and remote_test.get('status') == 'PASS'
    }

    if perf_test and perf_test.get('details'):
        details = perf_test['details']
        if isinstance(details, str):
            import ast
            try:
                details = ast.literal_eval(details)
            except:
                details = {}
        analysis.update({
            "cpu_time_ms": details.get('cpu_time_ms', 0),
            "remote_time_ms": details.get('remote_time_ms', 0),
            "network_overhead_ms": details.get('network_overhead_ms', 0),
            "total_overhead_ms": details.get('total_overhead_ms', 0)
        })

    return analysis

def generate_markdown_report(session_data, server_analysis, client_analysis, output_file):
    """Generate a markdown report from the analysis."""
    timestamp = session_data.get('timestamp', 'Unknown')
    server_ip = session_data.get('server_ip', 'Unknown')
    client_hostname = session_data.get('client_hostname', 'Unknown')

    report = f"""# Genie Multi-Node Testing Report

## Test Session Information
- **Timestamp**: {timestamp}
- **Server IP**: {server_ip}
- **Client Hostname**: {client_hostname}
- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Summary
- **Total Tests**: {session_data.get('summary', {}).get('overall', {}).get('total', 0)}
- **✅ Passed**: {session_data.get('summary', {}).get('overall', {}).get('passed', 0)}
- **❌ Failed**: {session_data.get('summary', {}).get('overall', {}).get('failed', 0)}

## Server-Side Results

### Test Summary
- **Total Tests**: {server_analysis.get('total_tests', 0)}
- **✅ Passed**: {server_analysis.get('passed', 0)}
- **❌ Failed**: {server_analysis.get('failed', 0)}
- **Success Rate**: {server_analysis.get('success_rate', 0):.1f}%

### Key Metrics
- **GPU Detected**: {'✅ Yes' if server_analysis.get('gpu_detected') else '❌ No'}
"""

    if server_analysis.get('gpu_name'):
        report += f"- **GPU Name**: {server_analysis['gpu_name']}\n"
    if server_analysis.get('gpu_time_ms'):
        report += f"- **GPU Performance**: {server_analysis['gpu_time_ms']:.1f}ms for 1000×1000 tensor\n"

    report += f"""
- **Server Operations**: {'✅ Working' if server_analysis.get('server_operations_working') else '❌ Failed'}

## Client-Side Results

### Test Summary
- **Total Tests**: {client_analysis.get('total_tests', 0)}
- **✅ Passed**: {client_analysis.get('passed', 0)}
- **❌ Failed**: {client_analysis.get('failed', 0)}
- **Success Rate**: {client_analysis.get('success_rate', 0):.1f}%

### Key Metrics
- **Network Connectivity**: {'✅ Working' if client_analysis.get('network_working') else '❌ Failed'}
- **Remote Execution**: {'✅ Working' if client_analysis.get('remote_execution_working') else '❌ Failed'}
"""

    if client_analysis.get('cpu_time_ms') and client_analysis.get('remote_time_ms'):
        report += f"""
### Performance Analysis
- **Local CPU Time**: {client_analysis['cpu_time_ms']:.1f}ms
- **Remote Execution Time**: {client_analysis['remote_time_ms']:.1f}ms
- **Network Overhead**: {client_analysis.get('network_overhead_ms', 0):.1f}ms
- **Total Overhead**: {client_analysis.get('total_overhead_ms', 0):.1f}ms
"""

    # Add analysis section if available
    analysis = session_data.get('analysis', {})
    if analysis.get('performance'):
        perf = analysis['performance']
        report += f"""
## Performance Analysis

### Network vs GPU Performance
- **Server GPU Time**: {perf.get('server_gpu_time_ms', 0):.1f}ms
- **Client Remote Time**: {perf.get('client_remote_time_ms', 0):.1f}ms
- **Network Overhead**: {perf.get('network_overhead_ms', 0):.1f}ms ({(perf.get('network_overhead_ms', 0) / perf.get('server_gpu_time_ms', 1) * 100):.1f}% of GPU time)

### End-to-End Performance
- **Client CPU Baseline**: {perf.get('client_cpu_time_ms', 0):.1f}ms
- **Remote GPU Execution**: {perf.get('client_remote_time_ms', 0):.1f}ms
- **Total Overhead**: {perf.get('total_overhead_ms', 0):.1f}ms ({(perf.get('total_overhead_ms', 0) / perf.get('client_cpu_time_ms', 1) * 100):.1f}% slower than local CPU)
"""

    report += """
## Recommendations

### For Week 2 Implementation
1. **LLM Decode Co-location**: Use the network overhead measurements to optimize placement
2. **Batch Processing**: Larger tensors will benefit more from GPU acceleration
3. **Connection Pooling**: Reuse HTTP connections to reduce overhead

### For Production Deployment
1. **Transport Optimization**: Consider WebSockets or gRPC for lower latency
2. **Connection Management**: Implement connection pooling and keep-alive
3. **Error Recovery**: Add retry logic and circuit breakers

## Files Referenced
- Server Results: Based on server-side test data
- Client Results: Based on client-side test data
- Analysis: Generated from combined metrics

---
*Report generated by Genie test result parser*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✅ Markdown report generated: {output_file}")
    return output_file

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_test_results.py <report_directory>")
        print("Example: python parse_test_results.py genie_test_results_20250109_143000")
        sys.exit(1)

    report_dir = sys.argv[1]

    if not os.path.exists(report_dir):
        print(f"Error: Directory '{report_dir}' not found")
        sys.exit(1)

    print("🔍 Parsing Genie test results...")
    print(f"📁 Report directory: {report_dir}")

    # Load comprehensive report
    comprehensive_file = os.path.join(report_dir, "comprehensive_report.json")
    if not os.path.exists(comprehensive_file):
        print(f"Error: {comprehensive_file} not found")
        sys.exit(1)

    with open(comprehensive_file, 'r') as f:
        session_data = json.load(f)

    # Analyze server and client results
    server_analysis = analyze_server_results(session_data.get('server_results'))
    client_analysis = analyze_client_results(session_data.get('client_results'))

    print("✅ Analysis complete")

    # Generate markdown report
    output_file = os.path.join(report_dir, "test_analysis_report.md")
    generate_markdown_report(session_data, server_analysis, client_analysis, output_file)

    # Display summary
    print("\n📊 Summary:")
    overall = session_data.get('summary', {}).get('overall', {})
    print(f"   Total Tests: {overall.get('total', 0)}")
    print(f"   ✅ Passed: {overall.get('passed', 0)}")
    print(f"   ❌ Failed: {overall.get('failed', 0)}")

    if overall.get('failed', 0) == 0:
        print("   🎉 All tests passed!")
    else:
        print("   ⚠️  Some tests failed. Check the markdown report for details.")

    print(f"\n📋 Files generated:")
    print(f"   📄 Markdown report: {output_file}")
    print(f"   📊 JSON data: {comprehensive_file}")

if __name__ == "__main__":
    main()
