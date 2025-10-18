# Genie Multi-Node Testing Guide

This guide explains how to run comprehensive tests for the Genie implementation across multiple nodes and how to record results for analysis.

## 🏗️ Architecture Overview

```
┌─────────────────┐    Network    ┌─────────────────┐
│  Client Node    │◄──────────────►│  Server Node    │
│  (No GPU)       │   HTTP/REST    │  (GPU: Tesla T4)│
│                 │                │                 │
│ • Test Scripts  │                │ • Genie Server  │
│ • Performance   │                │ • GPU Detection │
│ • Connectivity  │                │ • Tensor Ops    │
└─────────────────┘                └─────────────────┘
```

## 📋 Prerequisites

### Server Node (GPU Machine)
- ✅ GPU available (Tesla T4, RTX, etc.)
- ✅ CUDA installed and working
- ✅ Python environment with dependencies

### Client Node (Any Machine)
- ✅ Network connectivity to server
- ✅ Python environment with dependencies
- ✅ `jq` installed for JSON processing

## 🚀 Quick Start

### Step 1: Server Setup (GPU Machine)

```bash
# 1. Navigate to Genie directory
cd Genie

# 2. Start the server
python -m genie.runtime.simple_server --host 0.0.0.0 --port 8888

# 3. In another terminal, run server tests
bash scripts/week1_server_tests.sh

# 4. Note the server IP address for client testing
hostname -I  # e.g., 192.168.1.100
```

**Expected Output:**
```bash
🚀 Genie Server-Side Testing
===============================
📋 Server Information:
   Hostname: gpu-server
   IP Addresses: 192.168.1.100
   CUDA Available: True
   GPU: Tesla T4

🧪 Test 1: Environment Setup
✅ environment_setup

🧪 Test 2: Server Startup
✅ server_startup
   Details: Server PID: 1234, listening on 0.0.0.0:8888

🧪 Test 3: GPU Detection and Performance
✅ gpu_detection
   Details: {"cuda_available": true, "gpu_count": 1, "gpu_name": "Tesla T4", "gpu_time_ms": 19.2}

🎉 All server tests passed!
```

### Step 2: Client Setup (Any Machine)

```bash
# 1. Navigate to Genie directory
cd Genie

# 2. Run client tests (replace SERVER_IP with actual IP)
bash scripts/week1_client_tests.sh 192.168.1.100
```

**Expected Output:**
```bash
🚀 Genie Client-Side Testing
===============================
📋 Client Information:
   Hostname: client-machine
   Server IP: 192.168.1.100

🧪 Test 1: Network Connectivity
✅ network_connectivity

🧪 Test 2: Server HTTP Connectivity
✅ server_connectivity
   Details: {"server_url": "http://192.168.1.100:8888", "device": "cuda:0", "cuda_available": true}

🧪 Test 3: HTTP Tensor Transfer
✅ http_tensor_transfer
   Details: HTTP tensor transfer successful, HTTP code: 200

🎉 All client tests passed!
```

### Step 3: Comprehensive Testing (Optional)

```bash
# Run comprehensive tests (requires both server and client results)
bash scripts/week1_comprehensive_tests.sh 192.168.1.100

# Parse results for analysis
python scripts/parse_test_results.py genie_test_results_YYYYMMDD_HHMMSS/
```

## 📊 Test Categories

### Server-Side Tests (`scripts/server_test_commands.sh`)

| Test | Purpose | Measures |
|------|---------|----------|
| **Environment Setup** | Verify dependencies | Installation completeness |
| **Server Startup** | Test server initialization | Startup time, health endpoint |
| **GPU Detection** | Verify GPU availability | CUDA detection, GPU performance |
| **Server Operations** | Test HTTP tensor ops | End-to-end server functionality |
| **Server Statistics** | Monitor server health | Request counts, error rates |

### Client-Side Tests (`scripts/client_test_commands.sh`)

| Test | Purpose | Measures |
|------|---------|----------|
| **Network Connectivity** | Test basic networking | Ping latency, connectivity |
| **Server HTTP Connectivity** | Test HTTP communication | Response time, server health |
| **HTTP Tensor Transfer** | Test tensor serialization | Transfer time, correctness |
| **Remote LazyTensor Execution** | Test LazyTensor remote ops | End-to-end remote execution |
| **Performance Comparison** | Compare CPU vs remote GPU | Network overhead analysis |
| **Multi-Operation Chain** | Test operation chaining | Complex computation graphs |
| **Error Handling** | Test error scenarios | Graceful failure handling |
| **Large Tensor Handling** | Test memory limits | Scalability with large tensors |

## 📁 Result Recording Format

All scripts generate structured JSON results for easy parsing:

```json
{
  "test_info": {
    "hostname": "gpu-server",
    "timestamp": "2025-01-09T14:30:00",
    "test_type": "server_side"
  },
  "tests": [
    {
      "name": "gpu_detection",
      "status": "PASS",
      "details": "{\"cuda_available\": true, \"gpu_count\": 1, \"gpu_name\": \"Tesla T4\", \"gpu_time_ms\": 19.2}",
      "timing_ms": 150,
      "timestamp": "2025-01-09T14:30:01"
    },
    {
      "name": "server_operations",
      "status": "PASS",
      "details": "HTTP tensor transfer successful",
      "timing_ms": 85,
      "timestamp": "2025-01-09T14:30:02"
    }
  ]
}
```

### Key Metrics Captured

**Server Results:**
- GPU name and performance (ms for 1000×1000 tensor)
- Server startup time and health status
- HTTP operation latency
- Request statistics

**Client Results:**
- Network connectivity (ping time)
- HTTP response times
- Remote execution performance
- Network overhead vs local CPU
- Error handling effectiveness

## 🔍 Result Analysis

### Using the Result Parser

```bash
# Parse a specific test session
python scripts/parse_test_results.py genie_test_results_20250109_143000/

# Output: Detailed markdown report with analysis
```

**Generated Report Includes:**
- ✅ Overall test summary (passed/failed counts)
- 📊 Performance breakdowns (CPU vs GPU vs Remote)
- 🔍 Network overhead analysis
- 💡 Optimization recommendations
- 📈 Trend analysis across multiple runs

### Example Analysis Output

```markdown
## Performance Analysis

### Network vs GPU Performance
- **Server GPU Time**: 19.2ms
- **Client Remote Time**: 105.3ms
- **Network Overhead**: 86.1ms (448% of GPU time)

### End-to-End Performance
- **Client CPU Baseline**: 1.5ms
- **Remote GPU Execution**: 105.3ms
- **Total Overhead**: 103.8ms (6920% slower than local CPU)
```

## 🚨 Troubleshooting

### Common Issues

**Server Issues:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check server logs
tail -f server.log

# Test health endpoint manually
curl http://localhost:8888/health
```

**Client Issues:**
```bash
# Test network connectivity
ping SERVER_IP

# Test HTTP connectivity
curl http://SERVER_IP:8888/health

# Check environment
python -c "import torch; import requests; print('OK')"
```

**Result Parsing Issues:**
```bash
# Check JSON validity
cat results.json | jq .

# Install jq if missing
sudo apt-get install jq  # Ubuntu/Debian
brew install jq          # macOS
```

## 📈 Performance Expectations

### Typical Results (1000×1000 tensors)

| Metric | Local CPU | Local GPU | Remote GPU | Network Overhead |
|--------|-----------|-----------|------------|------------------|
| **Time** | 1.5ms | 19ms | 105ms | 86ms |
| **Rate** | Fast | 193 MB/s | 37 MB/s | 448% overhead |

### Interpretation
- **GPU Acceleration**: 13x faster than CPU for large tensors
- **Network Cost**: ~80-100ms per operation (HTTP + serialization)
- **Break-even Point**: Tensors >10MB benefit from remote GPU

## 🎯 Week 2 Integration

The test results provide baseline measurements for:

1. **LLM Decode Co-location**: Compare KV cache transfer vs co-location
2. **Transport Optimization**: HTTP vs WebSocket vs gRPC evaluation
3. **Batch Processing**: Optimal tensor sizes for GPU utilization
4. **Error Recovery**: Network failure and retry strategies

## 📋 File Structure

```
scripts/
├── server_test_commands.sh      # Server-side test runner
├── client_test_commands.sh      # Client-side test runner
├── run_comprehensive_tests.sh   # Combined test orchestrator
└── parse_test_results.py        # Result analysis and reporting

genie_test_results_YYYYMMDD_HHMMSS/
├── server_test_results_*.json   # Server test data
├── client_test_results_*.json   # Client test data
├── comprehensive_report.json     # Combined analysis
└── test_analysis_report.md       # Human-readable report
```

## 🚀 Advanced Usage

### Custom Test Parameters

```bash
# Modify tensor sizes in scripts
# Edit scripts/server_test_commands.sh and scripts/client_test_commands.sh

# Change test tensor sizes
# Line 165: test_tensor = torch.randn(100, 100)  # Change size here
```

### Batch Testing

```bash
# Run multiple iterations for statistical analysis
for i in {1..5}; do
    echo "=== Run $i ==="
    bash scripts/week1_client_tests.sh 192.168.1.100
done
```

### Integration with CI/CD

```bash
# Add to CI pipeline
#!/bin/bash
# Server setup (infrastructure)
bash scripts/week1_server_tests.sh

# Client testing (from CI runner)
bash scripts/week1_client_tests.sh $SERVER_IP

# Result analysis
python scripts/parse_test_results.py $(ls -td genie_test_results_* | head -1)
```

---

## ✅ Summary

This testing framework provides:

1. **✅ Comprehensive Coverage**: Server and client functionality
2. **✅ Structured Results**: JSON format for automated parsing
3. **✅ Performance Analysis**: Network overhead quantification
4. **✅ Error Detection**: Robust failure identification
5. **✅ Documentation**: Human-readable reports and recommendations

Use these scripts to validate your Week 1 implementation and gather baseline metrics for Week 2 optimizations!
