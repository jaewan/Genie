# Correct Multi-Node Test Execution Strategy

## 🚨 **Important: Script Execution Order**

**Do NOT run both scripts simultaneously on the same server!** Here's the correct approach:

## 🖥️ **Server Node (GPU Machine)**

### Step 1: Start Server (Terminal 1 - Keep Running)
```bash
# Start the server manually (persistent)
python -m genie.runtime.simple_server --host 0.0.0.0 --port 8888

# Server will run continuously and show:
# INFO:     Uvicorn running on http://0.0.0.0:8888
# Keep this terminal open!
```

### Step 2: Run Server Tests (Terminal 2)
```bash
# Wait 5 seconds for server to fully start, then run server tests
sleep 5
bash scripts/week1_server_tests.sh

# This will test the already-running server
# Server stays running after tests complete
```

**Expected Output:**
```bash
🧪 Test 2: Server Startup
✅ server_startup
   Details: Server PID: 1234, listening on 0.0.0.0:8888

🧪 Test 3: GPU Detection and Performance
✅ gpu_detection
   Details: {"cuda_available": true, "gpu_count": 1, "gpu_name": "Tesla T4", "gpu_time_ms": 19.2}
```

## 💻 **Client Node (Any Machine)**

### Step 3: Run Client Tests
```bash
# Run client tests against the running server
bash scripts/week1_client_tests.sh SERVER_IP

# Example:
bash scripts/week1_client_tests.sh 172.31.11.161
```

**Expected Output:**
```bash
🧪 Test 2: Server HTTP Connectivity
✅ server_connectivity
   Details: {"server_url": "http://172.31.11.161:8888", "device": "cuda:0", "cuda_available": true}

🧪 Test 4: Remote LazyTensor Execution
✅ remote_execution
   Details: {"remote_time_ms": 105.3, "server_url": "http://172.31.11.161:8888"}
```

## 📊 **Comprehensive Testing (Optional)**

### Step 4: Run Combined Analysis
```bash
# After both server and client tests are complete
bash scripts/week1_comprehensive_tests.sh SERVER_IP

# Example:
bash scripts/week1_comprehensive_tests.sh 172.31.11.161
```

## ⏱️ **Timing Sequence**

```
Time 0s:        Start server manually (Terminal 1)
Time 5s:        Server fully started
Time 5s:        Run server tests (Terminal 2)
Time 30s:       Server tests complete (server still running)
Time 30s:       Run client tests (from client machine)
Time 60s:       Client tests complete
Time 60s:       Run comprehensive analysis (optional)
```

## 🔧 **Alternative: Manual Server Control**

If you want more control over server lifecycle:

### Server Node Setup
```bash
# Terminal 1: Manual server control
python -m genie.runtime.simple_server --host 0.0.0.0 --port 8888 &
SERVER_PID=$!

# Terminal 2: Test against running server
sleep 3  # Wait for startup
curl http://localhost:8888/health  # Verify working
# Run your own tests here...

# When done:
kill $SERVER_PID
```

## 🚨 **Common Mistakes to Avoid**

❌ **Don't do this:**
```bash
# Wrong: Running server script while server is already running
python -m genie.runtime.simple_server --host 0.0.0.0 --port 8888 &
bash scripts/week1_server_tests.sh  # This starts ANOTHER server!
```

❌ **Don't do this:**
```bash
# Wrong: Running server and client scripts on same machine simultaneously
bash scripts/week1_server_tests.sh &  # Starts server
bash scripts/week1_client_tests.sh 127.0.0.1  # Tests against itself
```

## ✅ **Correct Execution Summary**

1. **Server Node:**
   ```bash
   # Terminal 1 (keep open):
   python -m genie.runtime.simple_server --host 0.0.0.0 --port 8888

   # Terminal 2:
   sleep 5
   bash scripts/week1_server_tests.sh
   ```

2. **Client Node:**
   ```bash
   bash scripts/week1_client_tests.sh SERVER_IP
   ```

3. **Analysis (any node):**
   ```bash
   bash scripts/week1_comprehensive_tests.sh SERVER_IP
   python scripts/parse_test_results.py genie_test_results_*/  # Most recent
   ```

## 📋 **Verification Commands**

### Check Server Status
```bash
# On server node:
curl http://localhost:8888/health

# Expected: {"status": "healthy", "device": "cuda:0", ...}
```

### Check Client Connectivity
```bash
# On client node:
curl http://SERVER_IP:8888/health
ping SERVER_IP
```

### Check Test Results
```bash
# List result directories (most recent first)
ls -td genie_test_results_* | head -3

# View latest results
python scripts/parse_test_results.py $(ls -td genie_test_results_* | head -1)
```

## 🎯 **Expected Timeline**

- **0-5s**: Server startup
- **5-30s**: Server tests (5 tests)
- **30-60s**: Client tests (8 tests)
- **60-90s**: Optional comprehensive analysis

**Total time: ~2-3 minutes for complete testing**

This approach ensures:
- ✅ Server runs persistently for all tests
- ✅ Tests run against the same server instance
- ✅ Realistic network conditions (not localhost)
- ✅ Proper error isolation and reporting
