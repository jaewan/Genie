#!/bin/bash
# Server-side test commands for Genie multi-node testing
# Run these on the GPU machine (server)

set -e  # Exit on any error

echo "🚀 Genie Server-Side Testing"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results file
RESULTS_FILE="server_test_results_$(date +%Y%m%d_%H%M%S).json"

# Initialize results
cat > "$RESULTS_FILE" << EOF
{
  "server_info": {
    "hostname": "$(hostname)",
    "timestamp": "$(date -Iseconds)",
    "test_type": "server_side"
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

echo "📋 Server Information:"
echo "   Hostname: $(hostname)"
echo "   IP Addresses: $(hostname -I)"
echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch' 2>/dev/null && python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null; then
    echo "   GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"
fi
echo ""

# Test 1: Environment Setup
echo "🧪 Test 1: Environment Setup"
START_TIME=$(date +%s%N)
if python -c 'import torch; import fastapi; import uvicorn; print("✅ All dependencies available")' 2>/dev/null; then
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "environment_setup" "PASS" "\"Dependencies: torch, fastapi, uvicorn available\"" "$TIMING"
else
    record_result "environment_setup" "FAIL" "\"Missing dependencies\"" "0"
fi
echo ""

# Test 2: Server Startup
echo "🧪 Test 2: Server Startup"
START_TIME=$(date +%s%N)

# Start server in background
python -m genie.runtime.simple_server --host 0.0.0.0 --port 8888 > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "server_startup" "PASS" "\"Server PID: $SERVER_PID, listening on 0.0.0.0:8888\"" "$TIMING"

    # Test health endpoint
    echo "   Testing health endpoint..."
    HEALTH_RESPONSE=$(curl -s http://localhost:8888/health 2>/dev/null || echo '{"status": "error"}')

    if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
        DEVICE=$(echo "$HEALTH_RESPONSE" | grep -o '"device":"[^"]*"' | cut -d'"' -f4)
        CUDA_AVAIL=$(echo "$HEALTH_RESPONSE" | grep -o '"cuda_available":[^,]*' | cut -d':' -f2 | tr -d ' ')
        record_result "health_endpoint" "PASS" "{\"device\": \"$DEVICE\", \"cuda_available\": $CUDA_AVAIL}" "0"
    else
        record_result "health_endpoint" "FAIL" "\"Health endpoint not responding correctly\"" "0"
    fi
else
    END_TIME=$(date +%s%N)
    TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
    record_result "server_startup" "FAIL" "\"Server failed to start\"" "$TIMING"
fi
echo ""

# Test 3: GPU Detection and Performance
echo "🧪 Test 3: GPU Detection and Performance"
START_TIME=$(date +%s%N)

if python -c 'import torch; print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}")' 2>/dev/null; then
    CUDA_AVAIL=$(python -c 'import torch; print(torch.cuda.is_available())')
    GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
    GPU_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')

    if [ "$CUDA_AVAIL" = "True" ] && [ "$GPU_COUNT" -gt 0 ]; then
        # Test GPU computation
        python -c "
import torch
import time

# Test GPU computation
x = torch.randn(1000, 1000).cuda()
start = time.time()
result = torch.relu(x)
gpu_time = (time.time() - start) * 1000

print(f'GPU computation: {gpu_time:.1f}ms')
" 2>/dev/null

        GPU_TIME=$(python -c "
import torch
import time
x = torch.randn(1000, 1000).cuda()
start = time.time()
result = torch.relu(x)
print(f'{(time.time() - start) * 1000:.1f}')
" 2>/dev/null)

        END_TIME=$(date +%s%N)
        TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
        record_result "gpu_detection" "PASS" "{\"cuda_available\": true, \"gpu_count\": $GPU_COUNT, \"gpu_name\": \"$GPU_NAME\", \"gpu_time_ms\": $GPU_TIME}" "$TIMING"
    else
        record_result "gpu_detection" "FAIL" "{\"cuda_available\": $CUDA_AVAIL, \"gpu_count\": $GPU_COUNT}" "0"
    fi
else
    record_result "gpu_detection" "FAIL" "\"GPU detection failed\"" "0"
fi
echo ""

# Test 4: Server Operations
echo "🧪 Test 4: Server Operations"
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
with open('server_test_tensor.pt', 'wb') as f:
    f.write(tensor_bytes.getvalue())

print('Created server_test_tensor.pt')
" 2>/dev/null

# Test server operation via curl
RESPONSE=$(curl -s -X POST http://localhost:8888/execute \
  -F "operation=relu" \
  -F "tensor_file=@server_test_tensor.pt" \
  --output server_result.pt 2>&1)

if [ $? -eq 0 ] && [ -f "server_result.pt" ]; then
    # Verify result
    python -c "
import torch

original = torch.load('server_test_tensor.pt')
result = torch.load('server_result.pt')
expected = torch.relu(original)

if torch.allclose(result, expected, atol=1e-5):
    print('✅ Server operation correct')
    print(f'Shape: {result.shape}, Device: {result.device}')
else:
    print('❌ Server operation incorrect')
    exit(1)
" 2>/dev/null

    if [ $? -eq 0 ]; then
        END_TIME=$(date +%s%N)
        TIMING=$(( (END_TIME - START_TIME) / 1000000 ))
        record_result "server_operations" "PASS" "\"Tensor operation via HTTP successful\"" "$TIMING"
    else
        record_result "server_operations" "FAIL" "\"Server operation returned incorrect results\"" "0"
    fi
else
    record_result "server_operations" "FAIL" "\"HTTP request failed: $RESPONSE\"" "0"
fi
echo ""

# Test 5: Server Statistics
echo "🧪 Test 5: Server Statistics"
HEALTH_RESPONSE=$(curl -s http://localhost:8888/health 2>/dev/null)

if echo "$HEALTH_RESPONSE" | grep -q '"stats"'; then
    TOTAL_REQS=$(echo "$HEALTH_RESPONSE" | grep -o '"total_requests":[0-9]*' | cut -d':' -f2)
    SUCCESS_REQS=$(echo "$HEALTH_RESPONSE" | grep -o '"successful":[0-9]*' | cut -d':' -f2)
    FAILED_REQS=$(echo "$HEALTH_RESPONSE" | grep -o '"failed":[0-9]*' | cut -d':' -f2)

    record_result "server_stats" "PASS" "{\"total_requests\": $TOTAL_REQS, \"successful\": $SUCCESS_REQS, \"failed\": $FAILED_REQS}" "0"
else
    record_result "server_stats" "FAIL" "\"Could not retrieve server statistics\"" "0"
fi
echo ""

# Cleanup
echo "🧹 Cleanup"
kill $SERVER_PID 2>/dev/null || true
rm -f server_test_tensor.pt server_result.pt server.log

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
    echo -e "${GREEN}🎉 All server tests passed!${NC}"
else
    echo -e "${RED}❌ Some server tests failed. Check results file for details.${NC}"
fi

echo ""
echo "📋 Next Steps:"
echo "   1. Review results in: $RESULTS_FILE"
echo "   2. Run client-side tests on remote machine"
echo "   3. Compare server vs client performance"
