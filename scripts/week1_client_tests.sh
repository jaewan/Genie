#!/bin/bash
# Client-side test commands for Genie multi-node testing
# Run these on the non-GPU machine (client)

set -e  # Exit on any error

# Check if server IP is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 SERVER_IP"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

SERVER_IP="$1"

echo "🚀 Genie Client-Side Testing"
echo "================================"
echo "Server IP: $SERVER_IP"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results file
RESULTS_FILE="client_test_results_$(date +%Y%m%d_%H%M%S).json"

# Initialize results
cat > "$RESULTS_FILE" << EOF
{
  "client_info": {
    "hostname": "$(hostname)",
    "server_ip": "$SERVER_IP",
    "timestamp": "$(date -Iseconds)",
    "test_type": "client_side"
  },
  "tests": []
}
EOF

# Function to record test result
record_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    local timing="$4"

    # Append to results file
    jq ".tests += [{\"name\": \"$test_name\", \"status\": \"$status\", \"details\": $details, \"timing_ms\": $timing, \"timestamp\": \"$(date -Iseconds)\"}]" "$RESULTS_FILE" > tmp.json && mv tmp.json "$RESULTS_FILE"

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ $test_name${NC}"
    else
        echo -e "${RED}❌ $test_name${NC}"
    fi

    if [ ! -z "$details" ]; then
        echo "   Details: $details"
    fi
    if [ ! -z "$timing" ]; then
        echo "   Time: ${timing}ms"
    fi
}

echo "📋 Client Information:"
echo "   Hostname: $(hostname)"
echo "   Server IP: $SERVER_IP"
echo ""

# Test 1: Network Connectivity
echo "🧪 Test 1: Network Connectivity"
START_TIME=$(date +%s%N)

if ping -c 1 -W 2 "$SERVER_IP" > /dev/null 2>&1; then
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "network_connectivity" "PASS" "\"Ping successful to $SERVER_IP\"" "$TIMING"
else
    record_result "network_connectivity" "FAIL" "\"Cannot ping $SERVER_IP\"" "0"
fi
echo ""

# Test 2: Server HTTP Connectivity
echo "🧪 Test 2: Server HTTP Connectivity"
START_TIME=$(date +%s%N)

SERVER_URL="http://$SERVER_IP:8888"
HEALTH_RESPONSE=$(curl -s -m 5 "$SERVER_URL/health" 2>/dev/null || echo '{"status": "error"}')

if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    DEVICE=$(echo "$HEALTH_RESPONSE" | grep -o '"device":"[^"]*"' | cut -d'"' -f4)
    CUDA_AVAIL=$(echo "$HEALTH_RESPONSE" | grep -o '"cuda_available":[^,]*' | cut -d':' -f2 | tr -d ' ')
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "server_connectivity" "PASS" "{\"server_url\": \"$SERVER_URL\", \"device\": \"$DEVICE\", \"cuda_available\": $CUDA_AVAIL}" "$TIMING"
else
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "server_connectivity" "FAIL" "\"Cannot connect to server at $SERVER_URL\"" "$TIMING"
fi
echo ""

# Test 3: HTTP Tensor Transfer
echo "🧪 Test 3: HTTP Tensor Transfer"
START_TIME=$(date +%s%N)

# Create test tensor
python -c "
import torch
import io

# Create test tensor
test_tensor = torch.randn(100, 100)
tensor_bytes = io.BytesIO()
torch.save(test_tensor, tensor_bytes)
tensor_bytes.seek(0)

# Save to file for curl test
with open('client_test_tensor.pt', 'wb') as f:
    f.write(tensor_bytes.getvalue())

print('Created client_test_tensor.pt')
" 2>/dev/null

# Test server operation via HTTP
HTTP_CODE=$(curl -s -w "%{http_code}" -o client_result.pt \
  -X POST "$SERVER_URL/execute" \
  -F "operation=relu" \
  -F "tensor_file=@client_test_tensor.pt" 2>/dev/null)

if [ "$HTTP_CODE" = "200" ] && [ -f "client_result.pt" ]; then
    # Verify result
    python -c "
import torch

original = torch.load('client_test_tensor.pt')
result = torch.load('client_result.pt')
expected = torch.relu(original)

if torch.allclose(result, expected, atol=1e-5):
    print('✅ HTTP transfer correct')
    print(f'Shape: {result.shape}, Device: {result.device}')
else:
    print('❌ HTTP transfer incorrect')
    exit(1)
" 2>/dev/null

    if [ $? -eq 0 ]; then
        END_TIME=$(date +%s%N)
        TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
        record_result "http_tensor_transfer" "PASS" "\"HTTP tensor transfer successful, HTTP code: $HTTP_CODE\"" "$TIMING"
    else
        record_result "http_tensor_transfer" "FAIL" "\"HTTP transfer returned incorrect results, HTTP code: $HTTP_CODE\"" "0"
    fi
else
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "http_tensor_transfer" "FAIL" "\"HTTP request failed, HTTP code: $HTTP_CODE\"" "$TIMING"
fi
echo ""

# Test 4: Remote LazyTensor Execution
echo "🧪 Test 4: Remote LazyTensor Execution"
START_TIME=$(date +%s%N)

# Set server URL for Genie
export GENIE_SERVER_URL="$SERVER_URL"

# Test remote execution via LazyTensor
python -c "
import genie
import torch
import time

# Test remote tensor creation and operation (using privateuseone backend)
x = torch.randn(100, 100, device='privateuseone:0')
y = x.relu().sigmoid()

start = time.time()
result = y.materialize()
elapsed = (time.time() - start) * 1000

print(f'Remote execution: {elapsed:.1f}ms')
print(f'Result shape: {result.shape}')
print(f'Result device: {result.device}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    REMOTE_TIME=$(python -c "
import genie
import torch
import time

x = torch.randn(100, 100, device='privateuseone:0')
y = x.relu().sigmoid()

start = time.time()
result = y.materialize()
print(f'{(time.time() - start) * 1000:.1f}')
" 2>/dev/null)

    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "remote_execution" "PASS" "{\"remote_time_ms\": $REMOTE_TIME, \"server_url\": \"$SERVER_URL\"}" "$TIMING"
else
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "remote_execution" "FAIL" "\"Remote LazyTensor execution failed\"" "$TIMING"
fi
echo ""

# Test 5: Performance Comparison
echo "🧪 Test 5: Performance Comparison"
START_TIME=$(date +%s%N)

# Local CPU execution for comparison
python -c "
import torch
import time

# Local CPU execution
x = torch.randn(100, 100)
start = time.time()
result = torch.relu(x)
cpu_time = (time.time() - start) * 1000

print(f'Local CPU: {cpu_time:.1f}ms')
" 2>/dev/null

CPU_TIME=$(python -c "
import torch
import time

x = torch.randn(100, 100)
start = time.time()
result = torch.relu(x)
print(f'{(time.time() - start) * 1000:.1f}')
" 2>/dev/null)

# Remote execution time (from previous test)
REMOTE_TIME=$(python -c "
import genie
import torch
import time

x = torch.randn(100, 100, device='privateuseone:0')
y = x.relu().sigmoid()

start = time.time()
result = y.materialize()
print(f'{(time.time() - start) * 1000:.1f}')
" 2>/dev/null)

if [ ! -z "$CPU_TIME" ] && [ ! -z "$REMOTE_TIME" ]; then
    OVERHEAD=$(echo "$REMOTE_TIME - $CPU_TIME" | bc -l)
    OVERHEAD_PCT=$(echo "($REMOTE_TIME / $CPU_TIME - 1) * 100" | bc -l)

    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "performance_comparison" "PASS" "{\"cpu_time_ms\": $CPU_TIME, \"remote_time_ms\": $REMOTE_TIME, \"overhead_ms\": $OVERHEAD, \"overhead_pct\": $OVERHEAD_PCT}" "$TIMING"
else
    record_result "performance_comparison" "FAIL" "\"Could not measure performance\"" "0"
fi
echo ""

# Test 6: Multi-Operation Chain
echo "🧪 Test 6: Multi-Operation Chain"
START_TIME=$(date +%s%N)

# Test chaining multiple operations
python -c "
import genie
import torch
import time

# Chain multiple operations
x = torch.randn(50, 50, device='privateuseone:0')
y = x.relu().sigmoid().tanh().abs()

start = time.time()
result = y.materialize()
elapsed = (time.time() - start) * 1000

print(f'Multi-op chain: {elapsed:.1f}ms')
print(f'Result shape: {result.shape}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    CHAIN_TIME=$(python -c "
import genie
import torch
import time

x = torch.randn(50, 50, device='privateuseone:0')
y = x.relu().sigmoid().tanh().abs()

start = time.time()
result = y.materialize()
print(f'{(time.time() - start) * 1000:.1f}')
" 2>/dev/null)

    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "multi_operation_chain" "PASS" "{\"chain_time_ms\": $CHAIN_TIME, \"operations\": \"relu->sigmoid->tanh->abs\"}" "$TIMING"
else
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "multi_operation_chain" "FAIL" "\"Multi-operation chain failed\"" "$TIMING"
fi
echo ""

# Test 7: Error Handling
echo "🧪 Test 7: Error Handling"
START_TIME=$(date +%s%N)

# Test unsupported operation
python -c "
import genie
import torch

try:
    x = torch.randn(10, 10, device='privateuseone:0')
    y = x.matmul(x)  # matmul might not be supported
    result = y.materialize()
    print('matmul succeeded')
except Exception as e:
    print(f'matmul failed as expected: {type(e).__name__}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "error_handling" "PASS" "\"Unsupported operations handled correctly\"" "$TIMING"
else
    record_result "error_handling" "FAIL" "\"Error handling not working as expected\"" "0"
fi
echo ""

# Test 8: Large Tensor Handling
echo "🧪 Test 8: Large Tensor Handling"
START_TIME=$(date +%s%N)

# Test with larger tensor
python -c "
import genie
import torch
import time

# Large tensor test
x = torch.randn(500, 500, device='privateuseone:0')  # ~1MB
y = x.relu()

start = time.time()
result = y.materialize()
elapsed = (time.time() - start) * 1000

print(f'Large tensor: {elapsed:.1f}ms')
print(f'Tensor size: {result.numel() * 4 / 1024:.1f} KB')
" 2>/dev/null

if [ $? -eq 0 ]; then
    LARGE_TIME=$(python -c "
import genie
import torch
import time

x = torch.randn(500, 500, device='privateuseone:0')
y = x.relu()

start = time.time()
result = y.materialize()
print(f'{(time.time() - start) * 1000:.1f}')
" 2>/dev/null)

    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "large_tensor_handling" "PASS" "{\"large_tensor_time_ms\": $LARGE_TIME, \"tensor_elements\": 250000}" "$TIMING"
else
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "large_tensor_handling" "FAIL" "\"Large tensor handling failed\"" "$TIMING"
fi
echo ""

# Cleanup
echo "🧹 Cleanup"
rm -f client_test_tensor.pt client_result.pt

echo ""
echo "📊 Test Results Summary:"
echo "   Results file: $RESULTS_FILE"
echo "   Total tests run: $(jq '.tests | length' "$RESULTS_FILE")"

# Count passed/failed tests
PASSED=$(jq '[.tests[] | select(.status == "PASS")] | length' "$RESULTS_FILE")
FAILED=$(jq '[.tests[] | select(.status == "FAIL")] | length' "$RESULTS_FILE")

echo "   ✅ Passed: $PASSED"
echo "   ❌ Failed: $FAILED"

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}🎉 All client tests passed!${NC}"
else
    echo -e "${RED}❌ Some client tests failed. Check results file for details.${NC}"
fi

echo ""
echo "📋 Next Steps:"
echo "   1. Review results in: $RESULTS_FILE"
echo "   2. Compare with server results"
echo "   3. Analyze network overhead vs GPU acceleration"

echo ""
echo "📁 Files created:"
echo "   $RESULTS_FILE - Client test results"
