# Week 3 Quick Start: Real Network Validation

## Overview

Week 3 validates our Week 2 simulation results on real network hardware. We deploy Genie on two machines and measure actual performance to verify simulation accuracy.

## Prerequisites

### Hardware Requirements
- **2 Linux machines** with network connectivity
- **Python 3.8+** on both machines
- **Network connectivity** between machines (Ethernet preferred)

### Software Requirements
- **Genie codebase** installed on both machines
- **PyTorch** installed on both machines
- **Network tools** (ping, traceroute, iperf)

## Quick Setup (15 minutes)

### Step 1: Basic Setup on Both Machines

```bash
# On both machines:
cd /path/to/genie

# Install dependencies
pip install -e .

# Verify installation
python -c "import genie; print('✅ Genie installed')"
```

### Step 2: Network Configuration

**Choose IP addresses:**
```bash
# Machine 1 (Server)
export SERVER_HOST=192.168.1.100
export SERVER_PORT=8888

# Machine 2 (Client)
export SERVER_HOST=192.168.1.100  # Same as server
export CLIENT_HOST=192.168.1.101
```

**Test network connectivity:**
```bash
# On client machine:
ping $SERVER_HOST
# Should show low latency (<10ms)

# Test port connectivity
nc -z $SERVER_HOST $SERVER_PORT || echo "Port not open yet"
```

## Step 3: Server Setup (Machine 1)

```bash
# Terminal 1 - Server Machine
export SERVER_HOST=192.168.1.100
export SERVER_PORT=8888

# Start server
python -m genie.runtime.simple_server --host 0.0.0.0 --port $SERVER_PORT

# Server should show:
# 🚀 Server starting with GPU: [GPU name] or CPU
# ✅ Server ready on device: cuda:0 or cpu
```

## Step 4: Client Setup (Machine 2)

```bash
# Terminal 2 - Client Machine
export SERVER_HOST=192.168.1.100
export CLIENT_HOST=192.168.1.101

# Test connectivity
python benchmarks/measure_real_network_llm.py --server-url "http://$SERVER_HOST:$SERVER_PORT" --baseline-only

# Should show:
# 🧪 Genie Week 3: Real Network LLM Benchmark
# 📡 Measuring BASELINE (no co-location) on REAL NETWORK
# ✅ Baseline measurement complete
```

## Step 5: Run Full Validation

```bash
# On client machine:

# 1. Baseline measurement (no co-location)
python benchmarks/measure_real_network_llm.py \
    --server-url "http://$SERVER_HOST:$SERVER_PORT" \
    --baseline-only

# 2. Optimized measurement (with co-location)
python benchmarks/measure_real_network_llm.py \
    --server-url "http://$SERVER_HOST:$SERVER_PORT" \
    --optimized-only

# 3. Compare with simulation
python benchmarks/compare_simulation_vs_real.py
```

## Expected Results

### Week 2 Simulation Results (Reference)
```
Baseline:    12.45ms per step
Optimized:   6.23ms per step
Improvement: 50.0%
```

### Week 3 Real Network Results (Target)
```
Baseline:    ~12-15ms per step (simulation ±20%)
Optimized:   ~6-8ms per step (simulation ±20%)
Improvement: ~40-60% (simulation ±10%)
```

## Troubleshooting

### Common Issues

**1. Connection Refused**
```bash
# Check server is running
curl "http://$SERVER_HOST:$SERVER_PORT/health"

# Check firewall
sudo ufw status  # Ubuntu/Debian
# sudo ufw allow $SERVER_PORT

# Check network interface
ip addr show
```

**2. High Latency**
```bash
# Check network performance
ping $SERVER_HOST  # Should be <10ms
traceroute $SERVER_HOST  # Check route

# Test bandwidth
iperf3 -c $SERVER_HOST  # If iperf3 available
```

**3. Permission Errors**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Check file permissions
ls -la /path/to/genie/

# Try running as user
whoami  # Should not be root
```

**4. Import Errors**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check Genie installation
python -c "import genie; print(genie.__version__)"
```

## Success Criteria

✅ **Server starts successfully**
- Health check returns 200 OK
- Shows device (GPU or CPU)

✅ **Client connects successfully**
- Can reach server health endpoint
- Can execute remote operations

✅ **Baseline measurement completes**
- Measures 10 decode steps
- Saves results to JSON

✅ **Optimized measurement completes**
- Uses co-location optimization
- Shows performance improvement

✅ **Comparison validates simulation**
- Error <20% for latency predictions
- Improvement direction correct
- Error <10% for improvement magnitude

## Time Estimate

| Step | Time | Status |
|------|------|--------|
| Setup (both machines) | 15 min | ✅ Complete |
| Server startup | 5 min | ⏳ In progress |
| Client connectivity | 10 min | ⏳ Pending |
| Baseline measurement | 15 min | ⏳ Pending |
| Optimized measurement | 15 min | ⏳ Pending |
| Comparison & analysis | 20 min | ⏳ Pending |
| **Total** | **80 min** | **~50% complete** |

## Next Steps

After successful validation:

1. **Document findings** in `docs/EVALUATION_WEEK3.md`
2. **Update overall status** in `docs/OVERALL_STATUS.md`
3. **Plan Week 4** (writing evaluation section)
4. **Prepare for OSDI submission**

## Emergency Commands

```bash
# Quick server test
curl "http://localhost:8888/health"

# Quick client test
python -c "
from genie.runtime.simple_client import RemoteExecutionClient
client = RemoteExecutionClient('http://localhost:8888')
import torch
x = torch.randn(10, 10)
result = client.execute('relu', x)
print('✅ Client works:', result.shape)
"

# Quick network test
python benchmarks/measure_real_network_llm.py --server-url "http://localhost:8888" --num-steps 3
```

---

**🎯 Goal:** Validate simulation accuracy on real hardware
**⏱️ Timeline:** Complete by end of Week 3
**📊 Success:** Simulation error <20%, improvement >10%
