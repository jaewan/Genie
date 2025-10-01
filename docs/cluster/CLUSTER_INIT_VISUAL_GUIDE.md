# Cluster Initialization - Visual Guide

**Quick visual reference** for understanding the cluster initialization feature.

---

## 🎯 What Problem Are We Solving?

### Before (Complex)

```
User Application
       │
       ├─ Must manually create TransportCoordinator
       ├─ Must configure DataPlaneConfig
       ├─ Must initialize control plane
       ├─ Must start monitoring
       ├─ Must handle errors
       └─ 50+ lines of boilerplate
```

### After (Simple)

```
User Application
       │
       └─ await genie.init(master_addr='gpu-server')
          ✓ Done! Use remote_accelerator device normally
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                         │
│                                                             │
│  import genie                                               │
│  await genie.init(master_addr='server')                     │
│                                                             │
│  x = torch.randn(1000, 1000, device='remote_accelerator:0')│
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                genie.cluster Package (NEW)                  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │    init.py   │  │ node_info.py │  │  monitoring.py   │ │
│  │              │  │              │  │                  │ │
│  │ • init()     │  │ • NodeInfo   │  │ • GPU monitor    │ │
│  │ • shutdown() │  │ • GPUInfo    │  │ • Health checks  │ │
│  │ • ClusterState│ │ • Status     │  │ • Events         │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           network_discovery.py (NEW)                 │  │
│  │  • Test TCP, DPDK, RDMA                             │  │
│  │  • Select optimal backend                           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           genie.runtime Package (EXISTING)                  │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────────────────┐    │
│  │ TCP Transport   │  │ DPDK Transport               │    │
│  │ (Fallback)      │  │ (Zero-copy)                  │    │
│  └─────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Initialization Flow

```
Step 1: User calls init()
   │
   ├─ Parse config (args + env vars)
   ├─ Validate parameters
   └─ Create ClusterState
   
Step 2: Create Local Node
   │
   ├─ Generate node_id
   ├─ Detect local GPUs
   └─ Set node role (client/server/worker)
   
Step 3: Discover Network
   │
   ├─ Test TCP connectivity ────────┐
   ├─ Test DPDK availability        │
   ├─ Test GPUDirect support        ├─> Select Best Backend
   └─ Test RDMA availability        │
                                    │
                                    ▼
                            Recommendation:
                            • dpdk_gpudev (best)
                            • rdma
                            • dpdk
                            • tcp (fallback)
   
Step 4: Initialize Backend
   │
   ├─ If DPDK: Init TransportCoordinator
   ├─ If TCP:  Init ControlDataIntegration
   └─ Connect to master node
   
Step 5: Start Monitoring
   │
   ├─ GPU monitoring (every 5s)
   ├─ Health checks (every 30s)
   ├─ Heartbeat (every 10s)
   └─ Event notifications
   
   ✅ READY! Return ClusterState
```

---

## 📊 Data Structures

### ClusterState (Singleton)

```
ClusterState
├─ initialized: bool
├─ config: ClusterConfig
├─ local_node: NodeInfo ────────┐
├─ nodes: Dict[str, NodeInfo]   │
├─ transport_coordinator        │
├─ control_integration          │
├─ monitor_tasks: List          │
└─ stats: Dict                  │
                                │
                                ▼
NodeInfo ──────────────────────────────┐
├─ node_id: str                        │
├─ hostname: str                       │
├─ role: NodeRole (client/server)      │
├─ host: str                           │
├─ control_port: int                   │
├─ network_backend: str (tcp/dpdk)     │
├─ gpu_count: int                      │
├─ gpus: List[GPUInfo] ────────┐      │
├─ status: NodeStatus          │      │
└─ last_heartbeat: float       │      │
                               │      │
                               ▼      │
                          GPUInfo     │
                          ├─ gpu_id   │
                          ├─ name     │
                          ├─ memory   │
                          ├─ util %   │
                          ├─ temp     │
                          └─ available
```

---

## 🌐 Network Discovery Process

```
Target: gpu-server.example.com:5555

Test 1: TCP Connectivity
   ├─ Connect to port 5555 ─────> ✓ Success (RTT: 1.2ms)
   └─ Result: tcp_reachable = True

Test 2: DPDK Availability (Local)
   ├─ Check libdpdk.so ─────────> ✓ Found
   ├─ Check EAL init ───────────> ✓ Can initialize
   └─ Result: dpdk_available = True

Test 3: GPUDirect RDMA
   ├─ Check CUDA GPUs ──────────> ✓ 2 GPUs found
   ├─ Check DPDK GPUDev ────────> ✓ rte_gpu_count() = 2
   └─ Result: gpu_direct_available = True

Test 4: RDMA Devices
   ├─ Check /sys/class/infiniband/
   ├─ Found: mlx5_0, mlx5_1
   └─ Result: rdma_available = True

Final Recommendation:
┌──────────────────────────────────────┐
│  Backend: dpdk_gpudev                │
│  Reason: Best performance available  │
│  Features: Zero-copy + GPU Direct    │
│  Est. Bandwidth: 95 Gbps             │
│  Est. Latency: 0.03 ms               │
└──────────────────────────────────────┘
```

---

## 📈 Monitoring Loop

```
Every 10 seconds: Heartbeat
   ├─ Send HEARTBEAT message to master
   ├─ Check peer nodes
   │  └─ If no response > 60s: Mark as disconnected
   └─ Update last_heartbeat timestamp

Every 5 seconds: GPU Monitor
   ├─ Query nvidia-smi
   ├─ Get utilization, memory, temp
   ├─ Update local_node.gpus
   └─ Emit events if changed:
      ├─ GPU_AVAILABLE (util < 90%)
      ├─ GPU_UNAVAILABLE (util > 90%)
      └─ GPU_OVERHEATED (temp > 85°C)

Every 30 seconds: Health Check
   ├─ Check GPU health
   ├─ Check network health
   ├─ Check peer connectivity
   ├─ Check memory pressure
   └─ Update node status:
      ├─ HEALTHY: All checks OK
      ├─ DEGRADED: Some issues
      ├─ UNHEALTHY: Major issues
      └─ CRITICAL: Severe failures
```

---

## 🎬 Example: Multi-Modal Model

```
User Code:
──────────
# Initialize once
await genie.init(master_addr='gpu-server')

# Create model
model = VQAModel()  # Vision + Text

# Run inference
image = load_image('cat.jpg')
text = "What animal is this?"
result = model(image, text)


Behind the Scenes:
──────────────────

1. genie.init() discovered:
   ✓ Server has 4x A100 GPUs
   ✓ DPDK GPUDirect available
   ✓ Network: 100 Gbps

2. Model operations on remote_accelerator:
   
   Vision Branch (GPU 0):
   ┌─────────────────────────┐
   │ image → ViT encoder     │
   │ → features [196, 768]   │
   └─────────────────────────┘
   
   Text Branch (GPU 1):
   ┌─────────────────────────┐
   │ text → BERT encoder     │
   │ → embeddings [128, 768] │
   └─────────────────────────┘
   
   Fusion (GPU 2):
   ┌─────────────────────────┐
   │ Combine vision + text   │
   │ → answer "A cat"        │
   └─────────────────────────┘

3. GPU Monitor detected:
   - GPU 0: 85% util (vision)
   - GPU 1: 75% util (text)
   - GPU 2: 60% util (fusion)
   
4. Health checks:
   ✓ All GPUs available
   ✓ Network OK (95 Gbps actual)
   ✓ All peers healthy
   
5. Result returned to user transparently!
```

---

## 🔀 Backend Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                    Backend Comparison                       │
├──────────┬────────────┬──────────┬──────────┬──────────────┤
│ Backend  │ Bandwidth  │ Latency  │ Zero-Copy│ Requirements │
├──────────┼────────────┼──────────┼──────────┼──────────────┤
│ TCP      │ 10 Gbps    │ 1.0 ms   │ No       │ None         │
│          │ ▓░░░░░░░░░ │          │          │ (Always OK)  │
├──────────┼────────────┼──────────┼──────────┼──────────────┤
│ DPDK     │ 90 Gbps    │ 0.05 ms  │ Yes      │ DPDK libs    │
│          │ ▓▓▓▓▓▓▓▓▓░ │          │          │              │
├──────────┼────────────┼──────────┼──────────┼──────────────┤
│ DPDK     │ 95 Gbps    │ 0.03 ms  │ Yes      │ DPDK +       │
│ GPUDev   │ ▓▓▓▓▓▓▓▓▓▓ │          │          │ CUDA GPUs    │
│ (BEST)   │            │          │          │              │
├──────────┼────────────┼──────────┼──────────┼──────────────┤
│ RDMA     │ 100 Gbps   │ 0.001 ms │ Yes      │ RDMA HW      │
│          │ ▓▓▓▓▓▓▓▓▓▓ │          │          │ (InfiniBand) │
└──────────┴────────────┴──────────┴──────────┴──────────────┘

Auto-Selection Priority:
1. DPDK GPUDev ─────────────────> If GPUs available
2. RDMA ────────────────────────> If RDMA hardware
3. DPDK ────────────────────────> If DPDK installed
4. TCP ─────────────────────────> Always available (fallback)
```

---

## 🎯 Configuration Options

```
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Matrix                      │
├─────────────────────┬──────────────┬────────────────────────┤
│ Option              │ Default      │ Purpose                │
├─────────────────────┼──────────────┼────────────────────────┤
│ master_addr         │ (required)   │ Server address         │
│ master_port         │ 5555         │ Control plane port     │
│ backend             │ auto         │ Network backend        │
│ node_role           │ client       │ client/server/worker   │
│ enable_heartbeat    │ True         │ Liveness detection     │
│ heartbeat_interval  │ 10.0s        │ How often to ping      │
│ heartbeat_timeout   │ 60.0s        │ When to give up        │
│ enable_gpu_monitor  │ True         │ Track GPU status       │
│ gpu_poll_interval   │ 5.0s         │ How often to poll      │
│ enable_health_checks│ True         │ Comprehensive checks   │
│ health_interval     │ 30.0s        │ Check frequency        │
│ timeout             │ 30.0s        │ Init timeout           │
└─────────────────────┴──────────────┴────────────────────────┘

Quick Configurations:
─────────────────────

Development (Fast init, verbose):
   await genie.init(
       master_addr='localhost',
       backend='tcp',
       heartbeat_interval=5.0
   )

Production (Reliable, monitored):
   await genie.init(
       master_addr='prod-server',
       backend='auto',  # Let it choose best
       enable_gpu_monitoring=True,
       enable_health_checks=True
   )

Server Mode (Provide GPUs):
   await genie.init(
       node_role='server',
       master_port=5555
   )
```

---

## 📊 Timeline Visualization

```
Week 1: Core Infrastructure
─────────────────────────────────
Mon  Tue  Wed  Thu  Fri
│    │    │    │    │
├─1.1: NodeInfo
│    │    │    │    │
│    ├─1.2: init()
│    │    │    │    │
│    │    ├─1.3: Env vars
│    │    │    │    │
│    │    │    ├─1.4: Tests
│    │    │    │    │
│    │    │    │    ✓ Phase 1 Complete


Week 2: Network Discovery
─────────────────────────────────
Mon  Tue  Wed  Thu  Fri
│    │    │    │    │
├─2.1: Discovery
│    │    │    │    │
│    ├─2.2: Backend selection
│    │    │    │    │
│    │    ├─2.3: Integration
│    │    │    │    │
│    │    │    ├─2.4: Tests
│    │    │    │    │
│    │    │    │    ✓ Phase 2 Complete


Week 3: Monitoring
─────────────────────────────────
Mon  Tue  Wed  Thu  Fri
│    │    │    │    │
├─3.2: GPU monitor
│    │    │    │    │
│    ├─3.3: Health checks
│    │    │    │    │
│    │    ├─3.4: Dashboard
│    │    │    │    │
│    │    │    ├─Tests
│    │    │    │    │
│    │    │    │    ✓ Phase 3 Complete


Week 4: Integration & Docs
─────────────────────────────────
Mon  Tue  Wed  Thu  Fri
│    │    │    │    │
├─4.1: Integration tests
│    │    │    │    │
│    ├─4.2: Documentation
│    │    │    │    │
│    │    ├─4.3: Examples
│    │    │    │    │
│    │    │    ├─4.4: Benchmarks
│    │    │    │    │
│    │    │    │    ✅ DONE!
```

---

## 🔍 Testing Visualization

```
Testing Pyramid
───────────────

                    ▲
                   ╱ ╲
                  ╱   ╲         Integration Tests
                 ╱  3  ╲        (test_end_to_end.py)
                ╱ Tests ╲       • Full workflow
               ╱─────────╲      • Multi-node
              ╱           ╲     • Real server
             ╱             ╲
            ╱               ╲
           ╱                 ╲
          ╱     Unit Tests    ╲  
         ╱      (15 files)     ╲ • Fast
        ╱   >90% coverage       ╲• Isolated
       ╱─────────────────────────╲• Mocked
      ────────────────────────────

Coverage Target: >90%
All tests must pass before merge!

Test Execution Time:
  Unit: ~30 seconds
  Integration: ~2 minutes
  Full suite: <3 minutes
```

---

## 🎓 Learning Path

```
Day 1: Understanding
├─ Read: Summary (15 min)
├─ Read: Quick Start (10 min)
└─ Setup environment (30 min)

Day 2-5: Week 1 Implementation
├─ Task 1.1: Node structures (1 day)
├─ Task 1.2: Init API (2 days)
├─ Task 1.3: Env vars (0.5 day)
└─ Task 1.4: Tests (0.5 day)

Day 6-10: Week 2 Implementation
├─ Task 2.1: Discovery (1.5 days)
├─ Task 2.2: Backend (1 day)
├─ Task 2.3: Integration (1 day)
└─ Task 2.4: Tests (0.5 day)

Day 11-15: Week 3 Implementation
├─ Task 3.2: GPU monitor (1.5 days)
├─ Task 3.3: Health (1.5 days)
└─ Task 3.4: Dashboard (1 day)

Day 16-20: Week 4 Integration
├─ Task 4.1: Integration (1 day)
├─ Task 4.2: Docs (2 days)
├─ Task 4.3: Examples (1 day)
└─ Task 4.4: Benchmarks (1 day)

✅ Done!
```

---

## 🚀 Quick Start Visual

```
Step 1: Read docs
   └─> CLUSTER_INIT_QUICK_START.md
   
Step 2: Setup
   └─> pip install -e .
   
Step 3: Pick task
   └─> Phase 1, Task 1.1
   
Step 4: Implement
   ├─> Copy code from plan
   ├─> Write tests
   └─> Make tests pass
   
Step 5: Commit
   └─> git commit -am "Task 1.1 complete"
   
Step 6: Next task
   └─> Repeat!
   
                    ┌─────────────────┐
After 4 weeks ───>  │  FEATURE DONE!  │
                    └─────────────────┘
```

---

**Questions?** → [See Documentation Index](CLUSTER_INIT_INDEX.md)

**Ready to start?** → [Go to Quick Start](CLUSTER_INIT_QUICK_START.md)

