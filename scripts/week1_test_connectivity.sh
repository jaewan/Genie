#!/bin/bash
# Quick connectivity test script

if [ $# -ne 1 ]; then
    echo "Usage: $0 SERVER_IP"
    echo "Example: $0 172.31.67.51"
    exit 1
fi

SERVER_IP="$1"

echo "🔍 Testing connectivity to $SERVER_IP"
echo "========================================"

# Test 1: Basic ping
echo -n "1. Ping test... "
if ping -c 1 -W 2 "$SERVER_IP" > /dev/null 2>&1; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# Test 2: Port connectivity
echo -n "2. Port 8888 connectivity... "
if command -v nc >/dev/null 2>&1; then
    if nc -zv "$SERVER_IP" 8888 2>/dev/null; then
        echo "✅ PASS"
    else
        echo "❌ FAIL"
    fi
else
    # Try alternative method using /dev/tcp if available
    if timeout 3 bash -c "echo >/dev/tcp/$SERVER_IP/8888" 2>/dev/null; then
        echo "✅ PASS (using /dev/tcp)"
    else
        echo "⚠️  SKIP (nc not available)"
    fi
fi

# Test 3: HTTP connectivity
echo -n "3. HTTP health check... "
if curl -s -m 3 "http://$SERVER_IP:8888/health" | grep -q '"status":"healthy"'; then
    echo "✅ PASS"
    echo "   Server is healthy!"
else
    echo "❌ FAIL"
    echo "   Cannot reach server"
fi

echo ""
echo "If all tests pass, run the full client test:"
echo "bash scripts/week1_client_tests.sh $SERVER_IP"
