
# Djinn: The Semantic Tensor Operating System

**Status**: Production Ready (v2.3.15, OSDI snapshot)  
**Last Updated**: November 25, 2025
**Target Audience**: Systems Researchers, ML Infrastructure Engineers, Platform Architects

---

## 1. The Return of the Mainframe: GPU Clusters as the Neo-Central Utility

In the period 1965–1985, the mainframe computer represented the ultimate concentration of computing power, accessible only to the largest organizations. Today’s large GPU clusters—specifically those built on NVIDIA H100 and GB200 architectures—have recreated the economic and technical conditions of the classic mainframe era with astonishing fidelity. Consequently, the industry is repeating the evolutionary trajectory that led from primitive batch monitors to genuine operating systems.

### 1.1 Economic Equivalence: The Return of the CapEx Barrier
The primary driver of centralization is the cost of scarcity.
*   **The Mainframe Era:** A fully configured IBM System/370 Model 195 (c. 1974) cost **$38–52 million** (2025 adj.).
*   **The AI Era:** A modern NVIDIA GB200 NVL72 rack (c. 2025) commands **$52–65 million**.

The "democratization of compute" characterized by the PC era has ended for frontier workloads. We have returned to a reality where the machinery of production is centralized, necessitating a shift from personal ownership models to **Time-Sharing**.

### 1.2 The Performance Chasm
The capability gap between local and frontier hardware necessitates a "Thin Client" architecture.
*   **1975:** A CDC 7600 offered a **5,000×** speed advantage over a departmental minicomputer.
*   **2025:** A GB200 rack delivers **10,000×** the FP8 throughput of a researcher's workstation.

The modern workstation has effectively reverted to the status of a "Dumb Terminal" (or 3270 console)—an interface to submit instructions to the actual computer: the GPU cluster.

### 1.3 The Software Lag: Schedulers are Batch Monitors
Modern cluster managers (Slurm, Kubernetes) are extraordinarily capable, yet they remain architecturally aligned with the **Batch Monitors** of the 1950s (GM-NAA I/O).
*   **No Hardware Abstraction:** They allocate physical boxes ("User A gets Node 4"), not virtual resources.
*   **No Unified Memory:** They lack a facility to treat cluster VRAM as a contiguous address space.
*   **Fragility:** Node failure kills the job, exactly as in early batch systems.

**The Solution:** A new class of GPU-native operating system is not merely desirable but architecturally necessary. This OS must provide topology abstraction, unified memory addressing, and preemptive multitasking. **Djinn is that Operating System.**

---

## 2. The Design Philosophy: Why a "Semantic" OS?

To build an OS for Deep Learning, we cannot simply clone Linux. Standard Operating Systems manage *bytes*; Deep Learning requires managing *tensors*.

### The Semantic Gap
A driver-level approach (e.g., modifying CUDA) fails because it lacks semantic context. A driver sees `malloc(1GB)` but cannot distinguish between:
*   **Weights:** Read-Only, Shared, Persistent.
*   **Activations:** Volatile, Discardable, Ephemeral.
*   **KV-Cache:** Private, Stateful, Persistent.

### The Framework as the Kernel
Djinn implements the **Library OS** architecture (similar to Exokernels). It embeds itself within the application layer—specifically **PyTorch**—to bridge the gap between high-level user intent and low-level hardware execution. By intercepting execution at the framework level, Djinn acts as a **Semantic Hypervisor**, translating Python intent into optimized OS primitives.

Applications express semantic intent via lightweight annotations (`djinn.session(phase='decode', priority='interactive', session_id=...)`), which the runtime translates into phase, QoS, and session metadata for the backend scheduler and memory kernel. This enables the OS to optimize placement, scheduling, and memory management based on workload semantics rather than treating all requests as generic compute.

### 2.x Theoretical Foundation

Djinn's design rests on principled separation of concerns, grounded in information theory and programming language semantics:

**Semantic Information Gap:** Framework-layer disaggregation is theoretically justified by the monotonicity of information: each layer below has strictly less semantic knowledge than the framework. A driver sees `malloc(1GB)` but a framework sees `allocate_kv_cache(1GB)`—all optimization gains come from this richer information.

**Selective Laziness:** The LazyTensor + materialization trigger model implements selective strictness in an otherwise lazy system, enabling dynamic control flow (e.g., MoE routing, conditional attention) while preserving static planning. This follows call-by-need evaluation principles, where materialization occurs only at control points, allowing planning to operate on resolved graphs.

---

## 3. System Architecture: The Big Picture

Djinn is architected like a modern Operating System, split into **User Space (Client)** and **Kernel Space (Server)**.

```
┌───────────────────────────────────────────┐      ┌───────────────────────────────────────────┐
│           USER SPACE (Client)             │      │           KERNEL SPACE (Server)           │
├───────────────────────────────────────────┤      ├───────────────────────────────────────────┤
│ 1. SYSCALL INTERFACE (LazyTensor)         │      │ 5. THE LINKER (Loader)                    │
│    • Intercepts Eager PyTorch             │      │    • Dynamic Linking (Ghost Models)       │
│    • Builds Semantically Rich Graph (SRG) │      │    • Shared Text Segment (Weights)        │
│                                           │      │                                           │
│ 2. THE COMPILER (Semantic Analysis)       │      │ 6. THE MEMORY KERNEL (Unified VMU)        │
│    • Phase Detection (Prefill vs Decode)  │      │    • Aligned Slab (Stack)                 │
│    • Optimization Hints                   │      │    • Private Heap (Data Segment)          │
│                                           │      │    • DMA Synchronization                  │
│ 3. THE I/O STACK (Serializer)             │      │                                           │
│    • Binary Protocol (No Pickle)          │      │ 7. THE SCHEDULER                          │
│    • Hybrid Transport (Coalesce/Scatter)  │      │    • Meta-Simulation (The Planner)        │
│                                           │      │    • Plan Caching (TLB)                   │
│ 4. SAFETY GUARD (Capability Engine)       │      │                                           │
│    • Local Resource Auditing              │      │ 8. GARBAGE COLLECTOR (Session Mgr)        │
└───────────────────────────────────────────┘      └───────────────────────────────────────────┘
```

### 3.1 End-to-End Execution Path: From Python Call to GPU Slab

Djinn's implementation mirrors a classic OS pipeline: **syscalls → planner → memory kernel → scheduler → device**—but for tensors.

1. **Syscall Interface (LazyTensor / Interceptor):**  
   The user writes standard PyTorch code: `model.to("remote_accelerator"); y = model(x)`. Djinn's `LazyTensor` and dispatcher intercept tensor operations and record them into a **Semantically Rich Graph (SRG)** instead of executing eagerly. This is analogous to an OS syscall layer: user code never sees the low-level protocol; it just calls "open file / run model."

2. **Compiler & Meta-Simulator (Planner):**  
   Before touching the GPU, Djinn runs the SRG on the `meta` device to compute exact tensor shapes, lifetimes (allocation → last use), phase labels (prefill vs decode) and lifecycle classes (weights, KV cache, activations). The result is a **Memory Plan**: a deterministic assignment of every activation to an offset in the **Stack Slab**, and every weight/state access to a location in the **Text** or **Data** segments. This is the analogue of an OS loader + virtual memory planner.

3. **Memory Kernel (VMU): Text / Data / Stack Segments:**  
   The server's **VMU** backs all model tensors: **Text Segment** (read-only weights, loaded once per GPU and shared across tenants), **Data Segment** (per-session state such as KV cache and persistent outputs), and **Stack Slab** (a large, aligned scratch region for intermediate activations, managed by a simple bump-pointer + reset). Execution never calls `cudaMalloc` for intermediates; it uses the precomputed slab plan, achieving OS-style zero external fragmentation.

4. **I/O Stack (Binary Protocol & Transport):**  
   The client serializes the request `(fingerprint, inputs, semantic hints)` with `ModelExecutionSerializer` into a **length-prefixed binary message**. Tensors are sent via a hybrid TCP transport that coalesces small messages and streams large ones, just as an OS network stack manages packetization and DMA.

5. **Scheduler (QoS + Meta-Simulation):**  
   On the server, each execution request carries semantic hints (extracted from `djinn.session()` context or explicitly provided): `execution_phase` (prefill/decode/vision), `priority`, `session_id`, `expected_tokens`, `kv_cache_size_mb`. These hints are normalized into `qos_class` (Realtime / Interactive / Batch) and used by the **BasicQosScheduler** to maintain per-class queues, enforce configurable concurrency shares, and record per-request `queue_latency_ms`. This is the analogue of a kernel runqueue with multi-class priorities and per-class concurrency limits.

6. **Execution on the GPU Slab (HybridExecutor):**  
   The `HybridExecutor` takes the plan and reads weights from the **Text Segment**, allocates activations in the **Stack Slab** according to the memory plan, and writes persistent state (e.g., updated KV cache) into the **Data Segment** for the next request. When the graph completes, it **copies out** only the requested outputs from the slab, then resets the slab pointer, reclaiming the full activation region in O(1) time.

7. **Result Return (Concrete or Skeletonized):**  
   The result is serialized back to the client either as **concrete tensors** (as in our current evaluation harness) or as **skeletonized structures** with remote references (Section 6.2) for lazy materialization. Client-side timing and GPU metrics (duration, `queue_latency_ms`, VRAM usage) are recorded to validate SLAs.

This end-to-end path is what makes Djinn behave like an operating system rather than a library wrapper: it has a clear syscall surface, a compiler and planner, a unified memory kernel (VMU), a QoS scheduler, and a well-defined I/O subsystem—all specialized for tensors and ML workloads.

---

## 4. The Frontend: From Eager Execution to Semantic Intent

Standard PyTorch is "Eager": it executes every operation immediately. This is excellent for debugging but inefficient for distributed systems.

### 4.1 The Interceptor: `LazyTensor`
Djinn introduces `LazyTensor`, a specialized tensor subclass that acts as a **Just-In-Time (JIT) Recorder**.
*   **Mechanism:** When a user runs `y = model(x)`, Djinn records the operation into a Directed Acyclic Graph (DAG) instead of executing it.
*   **Transparency:** This leverages `__torch_dispatch__`, requiring zero code changes from the user other than `model.to('remote')`.

### 4.2 The Artifact: Semantically Rich Graph (SRG)
Djinn does not just record a compute graph; it enriches the LazyTensor DAG with SRG metadata.
*   **Operation Semantics:** Distinguishes Compute-Bound ops (MatMul) from Memory-Bound ops (Attention).
*   **Phase Detection:** Automatically labels the graph as **"Prefill"** or **"Decode"**.
*   **Data Lifecycle:** Tags tensors as Ephemeral vs. Persistent, enabling the backend to optimize placement before allocation occurs.

> **Implementation note:** The SRG is a *view* over the LazyTensor DAG, not a second graph.  
> • Each node stores semantic fields lazily (phase, lifecycle, compute cost, memory hints).  
> • Subsystems (Meta-Simulator, Capability Interlock, VMU planner) query these fields directly.  
> • When we need portability or offline analysis, we materialize an SRG snapshot by walking the enriched DAG and emitting `(op, semantics, lifecycle)` tuples.  
> • This avoids duplicating graph storage while still enabling fast, query-specific caches (e.g., per-phase memory summaries) that are serialized alongside the Memory Plan.  
>
> Practically, this design keeps SRG construction overhead near-zero for common paths, yet preserves the option to export a normalized semantic graph for tooling, replay, or plan caching.

---

## 5. The Backend: The Unified Memory Kernel

To solve the fragmentation and latency issues inherent in multi-tenant GPU sharing, Djinn replaces standard allocators with an OS-inspired **Segmented Memory Model**.

### 5.1 The Unified VMU (Virtual Memory Unit)
Standard allocators (like `cudaMalloc`) fragment memory when handling concurrent dynamic workloads. Djinn's VMU partitions GPU memory into three segments, chosen to match the dominant (ownership, mutability, lifetime) classes observed in ML workloads:

| Memory Segment | OS Analogy | Lifecycle | Implementation |
| :--- | :--- | :--- | :--- |
| **Text Segment** | Shared Libs | **Read-Only** | **Model Weights.** Loaded once. Mapped into the virtual address space of every user running that model. Zero duplication. |
| **Data Segment** | Heap | **Private** | **KV-Cache & Outputs.** Owned by a specific Session ID (`session_id`). KV cache is keyed by `session_id` and persists across decode tokens within the same session, enabling efficient autoregressive generation. Outputs persist between requests to support stateful inference (e.g., Notebooks). Security is logically enforced by Session ID checks. |
| **Stack Slab** | Stack | **Volatile** | **Activations.** A massive, **256-byte aligned** scratchpad for intermediate compute. |

This segmentation minimizes duplication and maximizes cache efficiency: weights (shared, immutable) in Text; session state (private, persistent) in Data; activations (private, ephemeral) in the Stack with watermark reset, guaranteeing **zero external fragmentation** for intermediate computations.

### 5.2 The "Copy-Out" Execution Strategy
This strategy optimizes for **Zero External Fragmentation** (at the cost of minor internal fragmentation due to alignment):
1.  **Stack Allocation:** All intermediate activations use the **Stack Slab**. Allocation is a simple pointer increment (O(1)).
2.  **Execution:** The GPU computes the graph via the Slab.
3.  **Copy-Out:** Only the final requested outputs are cloned from the volatile Slab to the private **Data Segment**.
4.  **Reset:** The Slab pointer is reset to the watermark instantly. *Crucially, this reset occurs only after explicit `torch.cuda.synchronize()` ensures the GPU has finished computing.*

*Result:* A 70B model forward pass requiring 40GB of activation memory creates **zero** lasting external fragmentation. The memory is reclaimed the instant the pass completes.

### 5.3 Concurrency Model
Djinn employs a hybrid concurrency model:
*   **Space-Sharing (VRAM):** Users are isolated via private Data Segments (Heap). User A's KV Cache sits alongside User B's Weights.
*   **Time-Sharing (Compute):** The **Stack Slab** is a shared resource. Multiple users time-share the Slab for execution.
    *   **Text Segment:** Read-only, accessed concurrently by multiple CUDA streams.
    *   **Stack Slab:** Protected by stream synchronization. One user executes at a time per GPU stream, preventing data corruption.

### 5.4 VMU v2 Enhancements (Nov 2025)
*   **Dynamic safe capacity:** At startup the VMU now queries NVML (via `nvidia-ml-py`) to derive a `safe_capacity = min(total_free - os_reserve - safety_margin, total_memory - os_reserve)`. The new `ServerConfig` knobs (`os_reserve_gb`, `safety_margin_gb`, `min_viable_vmu_gb`, and per-segment ratios) guarantee we never enter Djinn with insufficient headroom.
*   **Per-session arenas:** The Data segment is now an arena allocator keyed by `session_id`. Each arena enforces quotas (`kv_cache_size_mb`, per-session bytes) and can be reclaimed instantly when a session ends (`gc_session`). When the most recent arena is freed, VRAM is physically returned to the segment tail, avoiding external fragmentation entirely.
*   **Admission control + preflight:** Evaluation entry points call `check_gpu_memory()` before launching Djinn so we abort fast if reservations cannot be honored. On the server we keep the same logic when sizing the VMU, so decode KV growth never starves the Text/Stack segments.
*   **Observability:** The new `vmu_metrics` hook exports Text/Data/Stack utilization, session counts, and fragmentation gaps after every request. These metrics feed both the OSDI paper and our alerting pipeline.

Together these upgrades explain why the CLIP "index is on cpu" bug disappeared (every buffer is now pinned into the Text segment) and why the Phase 4 load test sustained low tail latency even while LLM sessions expanded their KV caches.

---

## 6. The Virtualization Layer: Ghost Loading & Skeletonization

Djinn decouples the *definition* of data from the *location* of data, enabling true "Time-Sharing."

### 6.1 Ghost Interception & Shadow Sync
When a user loads a model, Djinn employs one of two strategies based on the source:

*   **Strategy A: Ghost Interception (HuggingFace):** Djinn hooks `from_pretrained`. The client creates a "Ghost Model" on the `meta` device (0 bytes RAM). The server downloads the weights directly to the **Text Segment**.
*   **Strategy B: Shadow Sync (Custom Models):** For user-defined architectures (e.g., `class MyNet(nn.Module)`), Djinn computes a structural hash, serializes the weights, and uploads them to the server's Text Segment in the background. This creates a "Ghost" replica for future runs.

Djinn streams those uploads asynchronously: each state dict is pinned on the CPU, copied over a dedicated CUDA stream, and the new `MetaSimulator` summaries (phase hint + lifecycle bytes + input bucket) are cached with the memory plan. This keeps registration efficient while still populating the segmented VMU slab with zero-copy transfers.

### 6.2 Output Skeletonization (Lazy I/O)

LazyTensor and the SRG decide **when** and **what** to execute on the remote GPU; they do not, by themselves, control **how much** of the result is brought back to the client. For that, Djinn introduces an orthogonal optimization: **output skeletonization**.

**The Problem:**  
Even with perfectly planned execution, naïvely returning full tensors across the network is wasteful. An LLM decode step produces a `[batch, seq, vocab]` logits tensor, but the client may only need `argmax(logits)` or the sampled token. Similarly, vision models produce dense feature maps, but applications typically use pooled embeddings or classification scores.

**The Solution:**  
When skeletonization is enabled, Djinn returns the **structure** of the result (dicts/tuples) with heavy tensors replaced by **remote references**. The client can operate on these references transparently; materialization occurs only when the user accesses specific data (e.g., `logits.argmax()` or `embedding.norm()`), triggering a remote fetch of only the needed slice or reduction.

**Current Status:**  
The **HybridExecutor** supports lazy output mode and can stream skeletons instead of dense tensors. For evaluation, we primarily return concrete tensors to keep measurements straightforward. Skeletonization is **implemented but not yet the default**; it serves as an optimization tier that can be selectively enabled.

In summary, LazyTensor and the SRG minimize *compute- and allocator-side overhead*, while skeletonization minimizes *network and client materialization overhead*. Both are necessary to make a disaggregated GPU system competitive with a local, monolithic GPU.

---

## 7. The Scheduler: Meta-Simulation & Planning

Traditional executors (like PyTorch Eager) calculate memory offsets at runtime, causing overhead. Djinn separates **Planning** from **Execution**.

### 7.1 The Meta-Simulator
Before execution, Djinn runs the SRG on the `meta` device (a zero-memory simulation) to calculate the exact size and lifespan of every intermediate tensor.
*   **The Output:** A deterministic **Memory Plan** mapping every operation to an exact offset in the **Stack Slab**.
*   **Benefit:** Eliminates runtime `malloc/free` calls entirely.
*   **Summary:** Every plan caches a lightweight SRG summary (`phase_hint`, `lifecycle` breakdowns, input bucket) so telemetry tools can ingest it without traversing the LazyTensor DAG again.

### 7.2 Plan Caching (The "TLB")
Simulating the graph takes time (~50ms). To achieve sub-millisecond latency, Djinn caches Plans.
*   **The Cache Key:** `(Model_Fingerprint, Input_Shape_Tuple)`.
*   **The TLB Effect:** For repeated requests (e.g., generating tokens 2, 3, 4...), the Scheduler skips simulation and loads the offsets from the cache (O(1) lookup).
*   **Result:** Reduces scheduling latency from 50ms to **<0.5ms**.

### 7.3 Basic QoS Classes (Realtime / Interactive / Batch)
Multi-tenant performance collapses when every request fights for the same slot. Djinn exposes **three QoS classes** that can be selected explicitly (`hints={'qos_class': 'realtime'}`) or inferred from deadlines:

*   **Realtime:** Strict priority, capped latency, intended for token-by-token decoding or streaming speech. Reserved concurrency slices ensure a realtime request is never starved by batch uploads.
*   **Interactive:** Default class for chatbots, notebook users, or dashboards. Shares the bulk of concurrency slots and is protected from background drains.
*   **Batch:** Background work (offline evals, artifact builds). Runs opportunistically and yields whenever higher classes arrive.

Under the hood a **Basic QoS Scheduler** keeps per-class queues, enforces configurable concurrency shares, and records per-request queue latency so we can chart SLA compliance. The scheduler is on by default (`GENIE_ENABLE_QOS=1`) but can be tuned via `GENIE_QOS_MAX_CONCURRENCY`, `GENIE_QOS_CLASS_SHARES` (e.g., `4,3,1` for realtime/interactive/batch), and `GENIE_QOS_ESCALATION_DELAY_MS` (configured at 300 ms for the latest load test) to keep P99 in check without sacrificing throughput.

---

## 8. The I/O Subsystem: High-Performance Plumbing

Connecting the Client and Server is a custom networking stack designed for Tensor workloads, addressing the "Serialization Bottleneck."

### 8.1 DjinnSerializer (Binary Protocol)
We replace Python's `pickle` (3-4ms overhead) with a **Length-Prefixed Binary Protocol**.
*   **Structure:** JSON header for metadata + Raw Byte Append for data.
*   **Zero-Copy:** Tensors are read directly from memory using `memoryview`, bypassing user-space copies.
*   **Performance:** Reduces serialization latency to **<0.5ms**.

### 8.2 Hybrid Transport & DMA
*   **Hybrid Coalescing:** Small requests (<1400B, e.g., prompts) are packed into single packets to minimize syscalls.
*   **Synchronized DMA:** On the server, data flows from Network → Pinned CPU Staging → GPU Slab via a **synchronized DMA pipeline**, ensuring the GPU never reads corrupted data during async transfers.

---

## 9. Reliability: The Kernel Guard

An OS must be robust. Djinn implements safeguards to prevent crashes in a distributed environment.

*   **Capability Interlock (Client-Side OOM Killer):** Before falling back to local execution (if the cluster is busy), Djinn audits local RAM. If the machine lacks resources (requires 1.5x headroom), it halts execution with a `ResourceError` instead of freezing the host OS.
*   **Session GC (Distributed Garbage Collection):** Distributed memory leaks are fatal. Djinn tracks **Session Leases** monitored by heartbeats. If a client disconnects, the Server immediately reclaims their private **Data Segment**, ensuring zero VRAM leaks.

---

## 10. Performance Summary

By moving from a "Network Wrapper" to a "Tensor Operating System," Djinn achieves performance metrics that validate the approach. The following table summarizes key improvements across different optimization dimensions:

| Metric | Baseline (Graph-Based) | Djinn v2.3 (Tensor OS) | Improvement |
| :--- | :--- | :--- | :--- |
| **Latency (Small)** | 31ms | **0.8ms** | **38x Faster** (via Plan Caching) |
| **Latency (Large)** | 868ms | **81ms** | **10.7x Faster** (via Zero-Copy) |
| **Bandwidth** | 100% (Full Return) | **0.3%** | **99.7% Reduction** (via Skeletonization) |
| **Fragmentation** | High (Standard Allocator) | **Zero (External)** | **Unified VMU (Slab)** |

**Production Load Test Results (Phase 4 Smoke, Nov 25, 2025):**  
With 14 concurrent users issuing LLM decode, CLIP classification, and multimodal embedding workloads through the QoS scheduler, Djinn delivered:
*   **P50 latency:** 78.6 ms
*   **P99 latency:** 703.8 ms (well below the 1 s realtime SLA and dramatically lower than the >4 s spikes we saw pre-fix)
*   **Throughput:** 4.6 req/s sustained with zero failed requests (568/568 successes)
*   **GPU utilization / memory:** 0.9 % SM, 14.6 GB VRAM in use thanks to shared Text segments
*   **Per-class P99:** 710 ms (LLM), 608 ms (vision), <10 ms (multimodal semantic lookups), showing the class shares and escalation delay working as intended
*   **Registration behavior:** Warmup pre-registers all 14 models in 20.7 s; zero registration stalls during steady state due to model fingerprinting + VMU pinning

**Semantic-aware evaluation highlights (Week 2 refresh):**
*   **Streaming audio (Exp 2.2, Whisper-Medium):** Fully semantic Djinn delivers ~12 % lower latency vs semantic-blind and ~99 % faster than native PyTorch thanks to encoder/decoder session reuse.
*   **Conversational AI (Exp 2.3, DialoGPT-Small):** Semantic Djinn is ~12× faster than semantic-blind and reduces data transfer by 96 %, demonstrating the power of KV-cache placement plus Lazy Reference Engine.

These results demonstrate that Djinn's OS-style architecture—with unified VMU memory management, QoS-aware scheduling, and semantic-aware planning—delivers predictable, low-latency performance even under concurrent multi-tenant load, validating the "Tensor Operating System" design philosophy.

---

## Contact & Citation

**Project Lead**: Jaewan Hong (jaewan@berkeley.edu)  

```bibtex
@inproceedings{hong2025lost,
  title={Lost in Translation: The Search for Meaning in Network-Attached AI Accelerator Disaggregation},
  author={Hong, Jaewan and Qiao, Yifan and Ponnapalli, Soujanya and Liu, Shu and Aguilera, Marcos K and Liu, Vincent and Rossbach, Christopher J and Stoica, Ion},
  booktitle={Proceedings of the 24th ACM Workshop on Hot Topics in Networks},
  pages={131--138},
  year={2025}
}
```