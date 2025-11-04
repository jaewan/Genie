#!/usr/bin/env bash
#
# Genie Network Validation Setup Script
# Sets up two machines for real network performance validation
#
# Usage:
#   ./setup_network_validation.sh server   # Setup as server
#   ./setup_network_validation.sh client   # Setup as client
#   ./setup_network_validation.sh both     # Setup both (requires two terminals)
#

set -euo pipefail

# Colors and logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Configuration
SERVER_HOST=${SERVER_HOST:-"192.168.1.100"}
SERVER_PORT=${SERVER_PORT:-"8888"}
CLIENT_HOST=${CLIENT_HOST:-"192.168.1.101"}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Show usage
show_help() {
    cat << EOF
Genie Network Validation Setup

USAGE:
    $0 server         Setup this machine as server
    $0 client         Setup this machine as client
    $0 both           Setup both (requires two terminals)

ENVIRONMENT VARIABLES:
    SERVER_HOST       Server IP address (default: 192.168.1.100)
    SERVER_PORT       Server port (default: 8888)
    CLIENT_HOST       Client IP address (default: 192.168.1.101)

EXAMPLES:
    # Terminal 1 - Server
    SERVER_HOST=10.0.0.10 ./setup_network_validation.sh server

    # Terminal 2 - Client
    SERVER_HOST=10.0.0.10 CLIENT_HOST=10.0.0.11 ./setup_network_validation.sh client

    # Both in one script
    ./setup_network_validation.sh both

EOF
}

# Check if IP is reachable
check_connectivity() {
    local host=$1
    local port=${2:-22}

    log_step "Checking connectivity to $host:$port..."

    if nc -z -w5 "$host" "$port" 2>/dev/null; then
        log_info "✅ $host:$port is reachable"
        return 0
    else
        log_warn "⚠️  $host:$port is not reachable"
        return 1
    fi
}

# Setup server machine
setup_server() {
    log_info "Setting up server machine..."

    # Install basic dependencies
    log_step "Installing Python dependencies..."
    cd "$PROJECT_ROOT"

    # Install in development mode
    if ! pip install -e .; then
        log_error "Failed to install Genie in development mode"
        log_info "Trying alternative installation method..."
        python setup.py develop || {
            log_error "Installation failed. Please check Python environment."
            exit 1
        }
    fi

    # Create server startup script
    cat > start_server.sh << EOF
#!/bin/bash
# Genie Server Startup Script
export GENIE_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
cd "$(dirname "\$0")/../.."
python -m genie.runtime.simple_server --host 0.0.0.0 --port ${SERVER_PORT}
EOF

    chmod +x start_server.sh

    # Create server test script
    cat > test_server.sh << EOF
#!/bin/bash
# Test script for server
echo "🧪 Testing Genie Server"
echo "======================"

# Health check
echo -n "Health check... "
if curl -s "http://${SERVER_HOST}:${SERVER_PORT}/health" | grep -q '"status":"healthy"'; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    exit 1
fi

# Load test
echo -n "Load test... "
python -c "
import torch
import requests
import io

# Create test tensor
x = torch.randn(100, 100)
buffer = io.BytesIO()
torch.save(x, buffer)
buffer.seek(0)

# Send request
response = requests.post(
    'http://${SERVER_HOST}:${SERVER_PORT}/execute',
    files={'tensor_file': ('tensor.pt', buffer, 'application/octet-stream')},
    data={'operation': 'relu'},
    timeout=10
)

if response.status_code == 200:
    print('✅ PASS')
else:
    print(f'❌ FAIL ({response.status_code})')
"
EOF

    chmod +x test_server.sh

    log_info "✅ Server setup complete!"
    log_info ""
    log_info "To start server:"
    log_info "  ./start_server.sh"
    log_info ""
    log_info "To test server:"
    log_info "  ./test_server.sh"
}

# Setup client machine
setup_client() {
    log_info "Setting up client machine..."

    # Install basic dependencies
    log_step "Installing Python dependencies..."
    cd "$PROJECT_ROOT"

    # Install in development mode
    if ! pip install -e .; then
        log_error "Failed to install Genie in development mode"
        log_info "Trying alternative installation method..."
        python setup.py develop || {
            log_error "Installation failed. Please check Python environment."
            exit 1
        }
    fi

    # Create client configuration
    mkdir -p config
    cat > config/network_config.json << EOF
{
    "server_url": "http://${SERVER_HOST}:${SERVER_PORT}",
    "client_host": "${CLIENT_HOST}",
    "network_type": "ethernet",
    "expected_latency_ms": 1.0,
    "expected_bandwidth_gbps": 1.0
}
EOF

    # Create client test script
    cat > test_client.py << EOF
#!/usr/bin/env python3
"""
Test script for client connectivity and performance.
"""

import torch
import time
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_file = os.path.join(os.path.dirname(__file__), 'config', 'network_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}

def test_basic_connectivity():
    """Test basic connectivity to server."""
    logger.info("🔗 Testing basic connectivity...")

    config = load_config()
    server_url = config.get('server_url', 'http://${SERVER_HOST}:${SERVER_PORT}')

    try:
        import requests
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            logger.info(f"✅ Server healthy: {health.get('device', 'unknown')}")
            return True
        else:
            logger.error(f"❌ Server returned {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

def test_remote_execution():
    """Test remote tensor execution."""
    logger.info("🚀 Testing remote execution...")

    config = load_config()
    server_url = config.get('server_url', 'http://${SERVER_HOST}:${SERVER_PORT}')

    try:
        from genie.runtime.simple_client import RemoteExecutionClient

        # Create test tensor
        x = torch.randn(50, 50)

        # Execute remotely
        client = RemoteExecutionClient(server_url)
        start_time = time.time()
        result = client.execute('relu', x)
        elapsed = (time.time() - start_time) * 1000

        if result.shape == x.shape:
            logger.info(f"✅ Remote execution successful: {elapsed:.1f}ms")
            return elapsed
        else:
            logger.error(f"❌ Shape mismatch: {result.shape} vs {x.shape}")
            return None

    except Exception as e:
        logger.error(f"❌ Remote execution failed: {e}")
        return None

def test_llm_workload():
    """Test LLM workload on real network."""
    logger.info("🤖 Testing LLM workload...")

    config = load_config()
    server_url = config.get('server_url', 'http://${SERVER_HOST}:${SERVER_PORT}')

    try:
        from examples.simple_llm import SimpleLLM

        # Create model
        model = SimpleLLM(hidden_size=256, cache_seq_len=64, batch_size=1)  # Smaller for testing

        # Set server URL for remote execution
        os.environ['GENIE_SERVER_URL'] = server_url

        # Test generation
        start_time = time.time()
        generated = model.generate(num_steps=3, device="remote_accelerator:0")
        elapsed = (time.time() - start_time) * 1000

        logger.info(f"✅ LLM generation successful: {len(generated)} tokens in {elapsed:.1f}ms")
        return elapsed

    except Exception as e:
        logger.error(f"❌ LLM workload failed: {e}")
        return None

def main():
    logger.info("=" * 60)
    logger.info("🧪 Genie Client Test Suite")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    logger.info(f"Configuration: {config}")

    # Run tests
    tests = [
        ("Basic Connectivity", test_basic_connectivity),
        ("Remote Execution", test_remote_execution),
        ("LLM Workload", test_llm_workload),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        result = test_func()
        results[test_name] = result

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)

    for test_name, result in results.items():
        if result is True:
            logger.info(f"{test_name:<20} ✅ PASS")
        elif result is not None and result is not False:
            logger.info(f"{test_name:<20} ✅ PASS ({result:.1f}ms)")
        else:
            logger.info(f"{test_name:<20} ❌ FAIL")

    # Save results
    results_file = "client_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'config': config,
            'results': results
        }, f, indent=2)

    logger.info(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
EOF

    chmod +x test_client.py

    # Create benchmark script for real network
    cat > benchmark_real_network.py << EOF
#!/usr/bin/env python3
"""
Benchmark real network performance vs simulation.
"""

import torch
import time
import logging
import json
import os
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_file = os.path.join(os.path.dirname(__file__), 'config', 'network_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}

def measure_network_latency():
    """Measure actual network latency."""
    logger.info("📡 Measuring network latency...")

    config = load_config()
    server_url = config.get('server_url', 'http://${SERVER_HOST}:${SERVER_PORT}')

    try:
        import requests

        # Ping server multiple times
        latencies = []
        for i in range(10):
            start = time.time()
            response = requests.get(f"{server_url}/health", timeout=5)
            elapsed = (time.time() - start) * 1000
            if response.status_code == 200:
                latencies.append(elapsed)

        if latencies:
            avg_latency = statistics.mean(latencies)
            logger.info(f"✅ Network latency: {avg_latency:.2f}ms avg")
            return avg_latency
        else:
            logger.error("❌ No successful pings")
            return None

    except Exception as e:
        logger.error(f"❌ Network latency test failed: {e}")
        return None

def measure_remote_execution():
    """Measure remote execution performance."""
    logger.info("🚀 Measuring remote execution performance...")

    config = load_config()
    server_url = config.get('server_url', 'http://${SERVER_HOST}:${SERVER_PORT}')

    try:
        from genie.runtime.simple_client import RemoteExecutionClient

        # Test different tensor sizes
        sizes = [(10, 10), (100, 100), (500, 500)]
        results = {}

        for size in sizes:
            logger.info(f"  Testing {size} tensor...")

            # Create test tensor
            x = torch.randn(*size)

            # Measure multiple times
            latencies = []
            client = RemoteExecutionClient(server_url)

            for i in range(5):
                start = time.time()
                result = client.execute('relu', x)
                elapsed = (time.time() - start) * 1000
                latencies.append(elapsed)

            avg_latency = statistics.mean(latencies)
            results[str(size)] = {
                'avg_ms': avg_latency,
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'tensor_elements': size[0] * size[1]
            }

            logger.info(f"    Average: {avg_latency:.2f}ms")

        return results

    except Exception as e:
        logger.error(f"❌ Remote execution benchmark failed: {e}")
        return None

def measure_llm_real_network():
    """Measure LLM performance on real network."""
    logger.info("🤖 Measuring LLM on real network...")

    config = load_config()
    server_url = config.get('server_url', 'http://${SERVER_HOST}:${SERVER_PORT}')

    try:
        from examples.simple_llm import SimpleLLM

        # Create model (smaller for network testing)
        model = SimpleLLM(hidden_size=256, cache_seq_len=32, batch_size=1)

        # Set server URL
        os.environ['GENIE_SERVER_URL'] = server_url

        # Measure generation
        latencies = []
        for trial in range(3):
            start = time.time()

            # This should trigger remote execution via LazyTensor
            generated = model.generate(num_steps=5, device="remote_accelerator:0")

            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)

            logger.info(f"  Trial {trial + 1}: {elapsed:.2f}ms for 5 steps")

        avg_latency = statistics.mean(latencies)
        logger.info(f"✅ LLM real network: {avg_latency:.2f}ms avg per 5 steps")

        return {
            'avg_ms_per_5_steps': avg_latency,
            'avg_ms_per_step': avg_latency / 5,
            'trials': len(latencies)
        }

    except Exception as e:
        logger.error(f"❌ LLM real network test failed: {e}")
        return None

def main():
    logger.info("=" * 60)
    logger.info("📊 Real Network Benchmark Suite")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    logger.info(f"Testing against: {config.get('server_url', 'unknown')}")

    # Run benchmarks
    benchmarks = [
        ("Network Latency", measure_network_latency),
        ("Remote Execution", measure_remote_execution),
        ("LLM Real Network", measure_llm_real_network),
    ]

    results = {}
    for bench_name, bench_func in benchmarks:
        logger.info(f"\nRunning: {bench_name}")
        result = bench_func()
        results[bench_name] = result

    # Save results
    timestamp = int(time.time())
    results_file = f"real_network_benchmark_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': config,
            'results': results
        }, f, indent=2)

    logger.info(f"\n💾 Results saved to: {results_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 Summary")
    logger.info("=" * 60)

    for bench_name, result in results.items():
        if result and isinstance(result, dict):
            if 'avg_ms' in result:
                logger.info(f"{bench_name:<20} {result['avg_ms']:.2f}ms avg")
            elif 'avg_ms_per_5_steps' in result:
                logger.info(f"{bench_name:<20} {result['avg_ms_per_5_steps']:.2f}ms (5 steps)")
        elif result:
            logger.info(f"{bench_name:<20} ✅ Completed")

    logger.info("\nCompare these results with simulation results to validate accuracy!")

if __name__ == "__main__":
    main()
EOF

    chmod +x benchmark_real_network.py

    log_info "✅ Client setup complete!"
    log_info ""
    log_info "To test client:"
    log_info "  python test_client.py"
    log_info ""
    log_info "To benchmark real network:"
    log_info "  python benchmark_real_network.py"
}

# Setup both machines (requires coordination)
setup_both() {
    log_info "Setting up both machines for network validation..."
    log_warn "This requires two separate machines/terminals"
    log_info ""

    # Generate instructions
    cat << EOF
🎯 NETWORK VALIDATION SETUP INSTRUCTIONS
=======================================

You need TWO machines with network connectivity.

MACHINE 1 (Server):
------------------
1. Run: SERVER_HOST=<SERVER_IP> ./setup_network_validation.sh server
2. Start server: ./start_server.sh
3. Keep server running

MACHINE 2 (Client):
------------------
1. Run: SERVER_HOST=<SERVER_IP> CLIENT_HOST=<CLIENT_IP> ./setup_network_validation.sh client
2. Test connectivity: python test_client.py
3. Run benchmarks: python benchmark_real_network.py

CONFIGURATION:
- Server IP: ${SERVER_HOST}
- Client IP: ${CLIENT_HOST}
- Port: ${SERVER_PORT}

EXAMPLE:
# Machine 1 (Server)
SERVER_HOST=192.168.1.100 ./setup_network_validation.sh server

# Machine 2 (Client)
SERVER_HOST=192.168.1.100 CLIENT_HOST=192.168.1.101 ./setup_network_validation.sh client

EOF
}

# Main execution
main() {
    case "${1:-help}" in
        server)
            setup_server
            ;;
        client)
            setup_client
            ;;
        both)
            setup_both
            ;;
        help|-h|--help)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
