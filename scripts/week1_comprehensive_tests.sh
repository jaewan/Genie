#!/bin/bash
# Comprehensive multi-node testing script for Genie
# Runs both server and client tests and generates a combined report

set -e  # Exit on any error

echo "🚀 Genie Comprehensive Multi-Node Testing"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if server IP is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 SERVER_IP"
    echo "Example: $0 192.168.1.100"
    echo ""
    echo "This script will:"
    echo "1. Run server-side tests on the GPU machine"
    echo "2. Run client-side tests on this machine"
    echo "3. Generate a comprehensive report"
    exit 1
fi

SERVER_IP="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="genie_test_results_$TIMESTAMP"

echo "📋 Test Configuration:"
echo "   Server IP: $SERVER_IP"
echo "   Client Hostname: $(hostname)"
echo "   Report Directory: $REPORT_DIR"
echo ""

# Create report directory
mkdir -p "$REPORT_DIR"

# Step 1: Run server-side tests
echo -e "${BLUE}🖥️  Step 1: Running server-side tests...${NC}"

# This would normally be run on the server machine
# For now, we'll simulate it or run it if we're on the server

if [ "$(hostname)" = "server" ] || ping -c 1 "$SERVER_IP" > /dev/null 2>&1 && [ "$SERVER_IP" = "$(hostname -I | awk '{print $1}')" ]; then
    echo "   Running server tests on this machine..."
    bash scripts/week1_server_tests.sh > "$REPORT_DIR/server_test.log" 2>&1

    if [ -f "server_test_results_*.json" ]; then
        mv server_test_results_*.json "$REPORT_DIR/"
        echo "   ✅ Server tests completed"
    else
        echo "   ❌ Server tests failed or no results file found"
    fi
else
    echo "   ℹ️  Server tests should be run on the GPU machine ($SERVER_IP)"
    echo "   Please run: bash scripts/week1_server_tests.sh"
    echo "   on the server machine first"
fi

echo ""

# Step 2: Run client-side tests
echo -e "${BLUE}💻 Step 2: Running client-side tests...${NC}"
bash scripts/week1_client_tests.sh "$SERVER_IP" > "$REPORT_DIR/client_test.log" 2>&1

if [ -f "client_test_results_*.json" ]; then
    mv client_test_results_*.json "$REPORT_DIR/"
    echo "   ✅ Client tests completed"
else
    echo "   ❌ Client tests failed or no results file found"
fi

echo ""

# Step 3: Generate comprehensive report
echo -e "${BLUE}📊 Step 3: Generating comprehensive report...${NC}"

REPORT_FILE="$REPORT_DIR/comprehensive_report.json"

# Initialize comprehensive report
cat > "$REPORT_FILE" << EOF
{
  "test_session": {
    "timestamp": "$(date -Iseconds)",
    "server_ip": "$SERVER_IP",
    "client_hostname": "$(hostname)",
    "report_generated": "$(date -Iseconds)"
  },
  "summary": {},
  "server_results": {},
  "client_results": {},
  "analysis": {}
}
EOF

# Parse and combine results
echo "   📋 Parsing test results..."

# Find result files
SERVER_RESULTS=$(find "$REPORT_DIR" -name "server_test_results_*.json" | head -1)
CLIENT_RESULTS=$(find "$REPORT_DIR" -name "client_test_results_*.json" | head -1)

if [ -f "$SERVER_RESULTS" ]; then
    echo "   ✅ Found server results: $(basename "$SERVER_RESULTS")"

    # Extract server summary
    SERVER_PASSED=$(jq '[.tests[] | select(.status == "PASS")] | length' "$SERVER_RESULTS")
    SERVER_FAILED=$(jq '[.tests[] | select(.status == "FAIL")] | length' "$SERVER_RESULTS")
    SERVER_TOTAL=$((SERVER_PASSED + SERVER_FAILED))

    # Update comprehensive report
    jq ".summary.server_tests = {\"passed\": $SERVER_PASSED, \"failed\": $SERVER_FAILED, \"total\": $SERVER_TOTAL}" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"
    jq ".server_results = $(cat "$SERVER_RESULTS")" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"

else
    echo "   ⚠️  No server results found"
    jq ".summary.server_tests = {\"passed\": 0, \"failed\": 0, \"total\": 0}" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"
fi

if [ -f "$CLIENT_RESULTS" ]; then
    echo "   ✅ Found client results: $(basename "$CLIENT_RESULTS")"

    # Extract client summary
    CLIENT_PASSED=$(jq '[.tests[] | select(.status == "PASS")] | length' "$CLIENT_RESULTS")
    CLIENT_FAILED=$(jq '[.tests[] | select(.status == "FAIL")] | length' "$CLIENT_RESULTS")
    CLIENT_TOTAL=$((CLIENT_PASSED + CLIENT_FAILED))

    # Update comprehensive report
    jq ".summary.client_tests = {\"passed\": $CLIENT_PASSED, \"failed\": $CLIENT_FAILED, \"total\": $CLIENT_TOTAL}" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"
    jq ".client_results = $(cat "$CLIENT_RESULTS")" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"

    # Generate analysis if both results are available
    if [ -f "$SERVER_RESULTS" ]; then
        # Compare performance
        SERVER_GPU_TIME=$(jq '.tests[] | select(.name == "gpu_detection") | .details.gpu_time_ms' "$SERVER_RESULTS" 2>/dev/null || echo "null")
        CLIENT_REMOTE_TIME=$(jq '.tests[] | select(.name == "remote_execution") | .details.remote_time_ms' "$CLIENT_RESULTS" 2>/dev/null || echo "null")
        CLIENT_CPU_TIME=$(jq '.tests[] | select(.name == "performance_comparison") | .details.cpu_time_ms' "$CLIENT_RESULTS" 2>/dev/null || echo "null")

        if [ "$SERVER_GPU_TIME" != "null" ] && [ "$CLIENT_REMOTE_TIME" != "null" ] && [ "$CLIENT_CPU_TIME" != "null" ]; then
            NETWORK_OVERHEAD=$(echo "$CLIENT_REMOTE_TIME - $SERVER_GPU_TIME" | bc -l)
            TOTAL_OVERHEAD=$(echo "$CLIENT_REMOTE_TIME - $CLIENT_CPU_TIME" | bc -l)

            jq ".analysis.performance = {
                \"server_gpu_time_ms\": $SERVER_GPU_TIME,
                \"client_remote_time_ms\": $CLIENT_REMOTE_TIME,
                \"client_cpu_time_ms\": $CLIENT_CPU_TIME,
                \"network_overhead_ms\": $NETWORK_OVERHEAD,
                \"total_overhead_ms\": $TOTAL_OVERHEAD
            }" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"
        fi
    fi

else
    echo "   ❌ No client results found"
    jq ".summary.client_tests = {\"passed\": 0, \"failed\": 0, \"total\": 0}" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"
fi

# Calculate overall summary
TOTAL_PASSED=$(jq '.summary.server_tests.passed + .summary.client_tests.passed' "$REPORT_FILE")
TOTAL_FAILED=$(jq '.summary.server_tests.failed + .summary.client_tests.failed' "$REPORT_FILE")
TOTAL_TESTS=$(jq '.summary.server_tests.total + .summary.client_tests.total' "$REPORT_FILE")

jq ".summary.overall = {\"passed\": $TOTAL_PASSED, \"failed\": $TOTAL_FAILED, \"total\": $TOTAL_TESTS}" "$REPORT_FILE" > tmp.json && mv tmp.json "$REPORT_FILE"

echo "   ✅ Report generated: $(basename "$REPORT_FILE")"
echo ""

# Step 4: Display summary
echo -e "${BLUE}📋 Test Summary:${NC}"
echo "   Total Tests: $TOTAL_TESTS"
echo "   ✅ Passed: $TOTAL_PASSED"
echo "   ❌ Failed: $TOTAL_FAILED"

if [ "$TOTAL_FAILED" -eq 0 ]; then
    echo -e "   ${GREEN}🎉 All tests passed!${NC}"
else
    echo -e "   ${RED}❌ Some tests failed. Check report for details.${NC}"
fi

echo ""
echo "📁 Files Generated:"
echo "   Report: $REPORT_DIR/comprehensive_report.json"
echo "   Server Log: $REPORT_DIR/server_test.log"
echo "   Client Log: $REPORT_DIR/client_test.log"

if [ -f "$SERVER_RESULTS" ]; then
    echo "   Server Results: $REPORT_DIR/$(basename "$SERVER_RESULTS")"
fi

if [ -f "$CLIENT_RESULTS" ]; then
    echo "   Client Results: $REPORT_DIR/$(basename "$CLIENT_RESULTS")"
fi

echo ""
echo "📋 Next Steps:"
echo "   1. Review the comprehensive report in: $REPORT_DIR/comprehensive_report.json"
echo "   2. Check logs for detailed error information"
echo "   3. Use the results for Week 2 optimization analysis"

echo ""
echo -e "${GREEN}✅ Comprehensive multi-node testing completed!${NC}"
