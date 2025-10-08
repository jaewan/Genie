# AI Agent Code Review Instructions

**Purpose:** Verify the implementation status and answer specific questions from the advisor review to create actionable next steps.

**Role:** You are a senior software engineer conducting a thorough code audit. Be precise, cite line numbers, and provide evidence for all claims.

---

## Part 1: End-to-End Execution Verification

### Objective
Answer: **"Can you actually send a tensor from GPU A to GPU B over the network right now?"**

### Files to Read

#### 1.1 Entry Point Analysis
**File:** `genie/core/executor.py`

**Tasks:**
- [ ] Find the `execute_subgraph()` function
- [ ] Trace what happens when `LazyTensor.materialize()` is called
- [ ] Identify where remote execution is triggered
- [ ] Check if there's a code path that actually sends data over network

**Report Format:**
```markdown
## 1.1 Executor Analysis

### execute_subgraph() implementation
- Location: `genie/core/executor.py:LINE_NUMBER`
- Function signature: [paste signature]
- What it does: [brief description]

### Materialize flow
```python
# Paste the actual code path from materialize() to network send
LazyTensor.materialize() 
  -> executor.execute_subgraph()
    -> ??? (fill in the chain)
      -> ??? 
        -> NETWORK SEND? (yes/no)
```

### Critical finding
- [ ] YES - There is a complete path to network send
- [ ] NO - The path is incomplete at: [describe where it breaks]
- Evidence: [paste relevant code]
```

#### 1.2 Transport Layer Analysis
**File:** `genie/runtime/transport_coordinator.py`

**Tasks:**
- [ ] Find `send_tensor()` method
- [ ] Trace what happens when it's called
- [ ] Check if `data_plane.send_tensor()` actually gets called
- [ ] Verify if it's calling C++ code or Python fallback

**Report Format:**
```markdown
## 1.2 Transport Coordinator Analysis

### send_tensor() implementation
[paste the full method with line numbers]

### Actual execution path
- [ ] Calls C++ data plane: YES/NO
- [ ] Calls Python TCP transport: YES/NO
- [ ] Falls back to mock: YES/NO
- Evidence: [paste relevant if/else branches]

### Data plane binding check
```python
# From transport_coordinator.py
self.data_plane = ??? # What gets initialized?
self.lib.genie_send_tensor = ??? # Does this exist?
```

### Critical finding
[Describe what actually happens when send_tensor is called]
```

#### 1.3 C++ Integration Check
**File:** `src/data_plane/genie_c_api.cpp`

**Tasks:**
- [ ] Find `genie_send_tensor()` function
- [ ] Check what it does with the tensor data
- [ ] Verify if DPDK actually sends packets
- [ ] Check if there's packet creation and network transmission code

**Report Format:**
```markdown
## 1.3 C++ API Analysis

### genie_send_tensor() implementation
```cpp
// Paste the function implementation with line numbers
```

### DPDK integration check
- [ ] Creates DPDK mbuf: YES/NO (line: X)
- [ ] Calls rte_eth_tx_burst(): YES/NO (line: X)
- [ ] Has packet creation: YES/NO (line: X)
- Evidence: [paste relevant DPDK calls]

### Critical finding
Does this function actually send network packets?
- [ ] YES - Complete implementation found
- [ ] NO - Only stub/placeholder
- [ ] PARTIAL - Has some parts but missing: [what's missing]
```

#### 1.4 Integration Test Check
**File:** `tests/test_integration.py` or similar

**Tasks:**
- [ ] Find end-to-end tests
- [ ] Check if any test actually runs remote execution
- [ ] Verify if tests start a server and client
- [ ] Check if tests verify actual tensor transfer

**Report Format:**
```markdown
## 1.4 Integration Test Analysis

### End-to-end tests found
1. Test name: [name]
   - File: [file:line]
   - What it tests: [description]
   - Does it test remote execution? YES/NO

### Missing tests
The following critical test is MISSING:
```python
def test_actual_remote_execution():
    """Test that we can actually send a tensor to remote GPU"""
    # This test does not exist
```

### Critical finding
[Describe the state of integration testing]
```

---

## Part 2: Transport Layer Clarity

### Objective
Answer: **"What actually happens when you call send_tensor()? Which backend is used by default?"**

### Files to Read

#### 2.1 Backend Selection
**File:** `genie/runtime/transport_coordinator.py`

**Tasks:**
- [ ] Find `__init__()` method
- [ ] Check how backend is selected
- [ ] Find all `if GENIE_DISABLE_CPP_DATAPLANE` checks
- [ ] Determine the default code path

**Report Format:**
```markdown
## 2.1 Backend Selection Analysis

### Initialization code
```python
# Paste __init__() method with line numbers
def __init__(self, ...):
    # What determines which backend?
```

### Environment variable checks
Found `GENIE_DISABLE_CPP_DATAPLANE` checks at:
1. Line X: [paste code]
2. Line Y: [paste code]

### Default behavior
With NO environment variables set:
- [ ] Uses C++ data plane
- [ ] Uses Python TCP
- [ ] Uses mock/simulation
- Evidence: [paste the default code path]

### Critical finding
The default execution path is: [describe]
```

#### 2.2 TCP Backend Implementation
**File:** `genie/runtime/transports.py`

**Tasks:**
- [ ] Find `PinnedTCPTransport` class
- [ ] Check if it has complete implementation
- [ ] Verify if it can actually send tensors
- [ ] Check if it's marked as "fallback" or "production"

**Report Format:**
```markdown
## 2.2 TCP Transport Analysis

### PinnedTCPTransport implementation
```python
# Paste the class with key methods
class PinnedTCPTransport:
    # Line numbers...
```

### Completeness check
- [ ] Has send_tensor(): YES/NO
- [ ] Has receive_tensor(): YES/NO
- [ ] Actually sends data: YES/NO
- [ ] Just a stub: YES/NO

### Usage in codebase
Search for `PinnedTCPTransport` usage:
- Used in: [list files]
- Called from: [list callers]
- Is it the active backend? YES/NO

### Critical finding
[Describe the state of TCP transport]
```

#### 2.3 Backend Factory Pattern
**File:** Search all files for backend initialization

**Tasks:**
- [ ] Find where backends are created
- [ ] Check if there's a factory pattern
- [ ] Determine selection priority
- [ ] Verify fallback chain

**Report Format:**
```markdown
## 2.3 Backend Factory Analysis

### Backend creation code
Found in: [file:line]
```python
# Paste the backend selection logic
```

### Selection priority
1. First choice: [backend name] - condition: [when]
2. Second choice: [backend name] - condition: [when]
3. Fallback: [backend name] - condition: [when]

### Critical finding
[Describe how backend selection actually works]
```

---

## Part 3: C++ Data Plane Status

### Objective
Answer: **"Is the C++ data plane code actually being compiled, linked, and called?"**

### Files to Read

#### 3.1 Build System Check
**Files:** 
- `setup.py`
- `CMakeLists.txt` (if exists)
- `Makefile` (if exists)

**Tasks:**
- [ ] Find how C++ code is compiled
- [ ] Check if `libgenie_data_plane.so` is built
- [ ] Verify build dependencies
- [ ] Check if build is optional or required

**Report Format:**
```markdown
## 3.1 Build System Analysis

### Build configuration found
- Build system: [cmake/make/setup.py]
- Location: [file path]

### C++ compilation
```python
# Paste relevant setup.py or CMakeLists.txt section
```

### Build outputs
Expected output: `libgenie_data_plane.so`
- Build command: [command]
- Output location: [path]
- Is build optional? YES/NO

### Dependencies
Required libraries:
- [ ] DPDK: version X
- [ ] CUDA: version Y
- [ ] Other: [list]

### Critical finding
[Describe if C++ code is actually built in standard installation]
```

#### 3.2 Library Loading Check
**File:** `genie/runtime/transport_coordinator.py` or `genie/runtime/dpdk_bindings.py`

**Tasks:**
- [ ] Find where `libgenie_data_plane.so` is loaded
- [ ] Check if loading failure is handled
- [ ] Verify if there's a graceful fallback
- [ ] Check error messages when library not found

**Report Format:**
```markdown
## 3.2 Library Loading Analysis

### Loading code
```python
# Paste the ctypes loading code with line numbers
```

### Error handling
```python
# Paste the try/except or fallback code
```

### What happens if .so is missing?
- [ ] Raises error and stops
- [ ] Falls back to Python
- [ ] Falls back to mock
- [ ] Silently does nothing
- Evidence: [paste code]

### Critical finding
[Describe what happens when C++ library is not available]
```

#### 3.3 Function Binding Check
**File:** `genie/runtime/transport_coordinator.py`

**Tasks:**
- [ ] Find where C function signatures are defined
- [ ] Check if `genie_send_tensor` is properly bound
- [ ] Verify argument types (ctypes definitions)
- [ ] Check if functions are actually called anywhere

**Report Format:**
```markdown
## 3.3 Function Binding Analysis

### C function bindings
```python
# Paste ctypes function signature definitions
self.lib.genie_send_tensor.argtypes = [...]
self.lib.genie_send_tensor.restype = ...
```

### Usage search
Search for `self.lib.genie_send_tensor(` in codebase:
- Found at: [file:line]
- Context: [paste usage]

### Are bindings used?
- [ ] YES - Called from: [list locations]
- [ ] NO - Only defined but never called
- Evidence: [paste code]

### Critical finding
[Describe if C++ functions are actually invoked]
```

---

## Part 4: Performance Measurements

### Objective
Answer: **"Do you have any actual measurements of throughput/latency?"**

### Files to Read

#### 4.1 Benchmark Code Check
**Files:** Search for:
- `tests/performance/`
- `benchmarks/`
- `examples/benchmark_*.py`

**Tasks:**
- [ ] Find any performance measurement code
- [ ] Check what metrics are measured
- [ ] Verify if measurements are for actual network transfer
- [ ] Check if results are documented

**Report Format:**
```markdown
## 4.1 Benchmark Analysis

### Benchmark files found
1. [file path]
   - Measures: [what]
   - Lines: [line numbers]
2. [file path]
   ...

### Measurements taken
- [ ] Throughput (Gbps): YES/NO
- [ ] Latency (ms): YES/NO
- [ ] CPU overhead: YES/NO
- [ ] Memory usage: YES/NO

### Example benchmark
```python
# Paste a representative benchmark
```

### Results
Are there any documented results?
- Location: [file/doc]
- Results: [paste if found]

### Critical finding
[Describe the state of performance measurement]
```

#### 4.2 Metrics Collection Check
**File:** `genie/runtime/metrics.py`

**Tasks:**
- [ ] Check if metrics are actually collected
- [ ] Verify if they're exposed (Prometheus, etc.)
- [ ] Check if latency profiler is used
- [ ] Verify measurement accuracy

**Report Format:**
```markdown
## 4.2 Metrics Collection Analysis

### Metrics exported
```python
# Paste metrics definitions
```

### Collection points
Metrics are collected at:
1. [location]: [what metric]
2. [location]: [what metric]

### Usage check
Is `metrics.observe_transfer_latency()` actually called?
- Search results: [list locations]
- Are values accurate? [analysis]

### Critical finding
[Describe if real measurements are being taken]
```

---

## Part 5: Static Analysis Timing

### Objective
Answer: **"When does FX tracing actually happen? Is it working?"**

### Files to Read

#### 5.1 FX Tracing Code
**File:** `genie/core/graph.py` or `genie/semantic/fx_analyzer.py`

**Tasks:**
- [ ] Find where `fx.symbolic_trace()` is called
- [ ] Check when tracing happens (init vs runtime)
- [ ] Verify if traced graph is cached
- [ ] Check if tracing can fail

**Report Format:**
```markdown
## 5.1 FX Tracing Analysis

### Tracing invocation
```python
# Paste the code that calls fx.symbolic_trace()
# Include line numbers
```

### When it happens
- [ ] At model registration time (once)
- [ ] At first forward pass
- [ ] Every forward pass
- [ ] Never (stub)
- Evidence: [paste call stack]

### Caching
```python
# Paste caching logic if exists
```

### Error handling
What happens if tracing fails?
```python
# Paste try/except or fallback code
```

### Critical finding
[Describe when and how FX tracing works]
```

#### 5.2 Analysis Pipeline
**File:** `genie/semantic/analyzer.py`

**Tasks:**
- [ ] Find `analyze_graph()` method
- [ ] Check what inputs it expects
- [ ] Verify if it uses FX graph
- [ ] Check if it uses dynamic dispatch graph
- [ ] Determine which graph is primary

**Report Format:**
```markdown
## 5.2 Analysis Pipeline

### analyze_graph() implementation
```python
# Paste the method with line numbers
```

### Input sources
- [ ] Uses FX graph: YES/NO (from where?)
- [ ] Uses LazyTensor graph: YES/NO (from where?)
- [ ] Uses both: YES/NO (how combined?)

### Analysis flow
```
analyze_graph()
  -> Gets FX graph from: [source]
  -> Gets operation data from: [source]
  -> Combines them via: [method]
  -> Returns: [what]
```

### Critical finding
[Describe the actual analysis pipeline]
```

---

## Part 6: Semantic Optimization Implementation

### Objective
Answer: **"Are semantic optimizations actually applied? Can you measure their benefit?"**

### Files to Read

#### 6.1 Optimizer Implementation
**File:** `genie/semantic/optimizer.py`

**Tasks:**
- [ ] Find `optimize()` method
- [ ] Check which optimizations are implemented
- [ ] Verify if optimizations modify the graph
- [ ] Check if optimizations are actually applied

**Report Format:**
```markdown
## 6.1 Optimizer Analysis

### optimize() implementation
```python
# Paste key sections with line numbers
```

### Implemented optimizations
- [ ] KV cache co-location: IMPLEMENTED/STUB
- [ ] Prefill parallelization: IMPLEMENTED/STUB
- [ ] CNN pipelining: IMPLEMENTED/STUB
- [ ] Multi-modal parallel: IMPLEMENTED/STUB

### Example: KV cache co-location
```python
# Paste the _apply_llm_optimizations code
```

### Graph modification
Does optimizer actually modify the graph?
- [ ] YES - modifies nodes: [how]
- [ ] NO - only adds metadata
- Evidence: [paste code]

### Critical finding
[Describe what optimizations actually do]
```

#### 6.2 Optimization Application Check
**File:** Search for optimization usage

**Tasks:**
- [ ] Find where `optimizer.optimize()` is called
- [ ] Check if optimization plan is used
- [ ] Verify if optimizations affect execution
- [ ] Check for any performance comparison

**Report Format:**
```markdown
## 6.2 Optimization Usage Analysis

### Optimizer invocation
Found calls to `optimizer.optimize()` at:
1. [file:line] - [context]
2. [file:line] - [context]

### Optimization plan usage
```python
# Paste code showing how optimization plan is used
```

### Effect on execution
Do optimizations change behavior?
- [ ] YES - execution path differs
- [ ] NO - optimizations ignored
- [ ] UNKNOWN - can't determine
- Evidence: [paste code]

### Comparison code
Is there any code comparing optimized vs unoptimized?
- Search for: "baseline", "comparison", "vs_"
- Found: [list findings]

### Critical finding
[Describe if optimizations have measurable effect]
```

---

## Part 7: Test Coverage Reality Check

### Objective
Answer: **"What do the 52 passing tests actually test?"**

### Files to Read

#### 7.1 Test Inventory
**Files:** All files in `tests/`

**Tasks:**
- [ ] List all test files
- [ ] Categorize tests (unit/integration/e2e)
- [ ] Count assertions about remote execution
- [ ] Identify mocked vs real tests

**Report Format:**
```markdown
## 7.1 Test Inventory

### Test files
1. `tests/test_lazy_tensor.py` - 15 tests
   - Category: Unit
   - Tests remote execution: NO (tests LazyTensor creation)
   - Uses mocks: YES/NO
   
2. `tests/test_...` 
   [list all]

### Test categories
- Unit tests (isolated components): X tests
- Integration tests (multiple components): Y tests
- End-to-end tests (full flow): Z tests

### Remote execution coverage
Tests that actually test remote execution:
- [ ] NONE found
- [ ] Found: [list tests that test network send/receive]

### Critical finding
[Summarize what tests actually cover]
```

#### 7.2 Mock Usage Analysis
**Files:** Look for Mock, MagicMock, patch in tests

**Tasks:**
- [ ] Find all mock usage
- [ ] Identify what's being mocked
- [ ] Check if transport layer is mocked
- [ ] Verify if any test uses real network

**Report Format:**
```markdown
## 7.2 Mock Analysis

### Mocked components
1. [component]: mocked in [test file]
   - Why: [reason]
   - Effect: [what this means for coverage]

### Transport layer mocking
```python
# Paste examples of transport mocking
```

### Real network tests
Tests that use actual network:
- [ ] NONE - all use mocks
- [ ] Found: [list tests]

### Critical finding
[Describe reliance on mocking vs real testing]
```

---

## Part 8: Scope and Complexity Analysis

### Objective
Answer: **"Is the implementation trying to do too much? What's actually needed?"**

### Files to Read

#### 8.1 Feature Matrix
**Files:** All implementation files

**Tasks:**
- [ ] Count lines of code by category
- [ ] Identify "Phase 1" vs "Phase 2+" code
- [ ] Find TODO/FIXME comments
- [ ] Identify unused code paths

**Report Format:**
```markdown
## 8.1 Scope Analysis

### Lines of code by category
- Semantic layer: X lines
  - Pattern recognition: Y lines
  - Optimizer: Z lines
- Transport layer: X lines
  - C++ DPDK: Y lines
  - Python TCP: Z lines
  - Python control: W lines
- Cluster management: X lines
- Tests: X lines
- Total: X lines

### Phase markers
Found "Phase 1" markers: [count]
Found "Phase 2+" markers: [count]
Example:
```python
# Line X: # Phase 1: no-op
# Line Y: # Phase 2+: Will implement
```

### TODO/FIXME analysis
Total TODOs: [count]
Critical TODOs:
1. [file:line] - [description]
2. [file:line] - [description]

### Unused code
Code that's defined but never called:
1. [function/class] - [file:line]
2. [function/class] - [file:line]

### Critical finding
[Describe scope creep and complexity]
```

#### 8.2 Dependency Analysis
**Files:** `setup.py`, `requirements.txt`, imports

**Tasks:**
- [ ] List all dependencies
- [ ] Identify optional vs required
- [ ] Check which are actually used
- [ ] Find dependencies of dependencies

**Report Format:**
```markdown
## 8.2 Dependency Analysis

### Direct dependencies
From setup.py/requirements.txt:
- torch: required, version X
- dpdk: optional/required? version Y
- [list all]

### Actually used
Dependencies with import found in code:
- torch: YES (used in X files)
- dpdk: YES/NO (used in X files)
- [list all]

### Optional dependencies
Dependencies that could be removed:
- [name]: used only in [file], could be replaced with [alternative]

### Heavyweight dependencies
Large/complex dependencies:
- DPDK: Size X, complexity Y, used in Z places
- [list if there are others]

### Critical finding
[Describe dependency complexity]
```

---

## Part 9: Working Demo Check

### Objective
Answer: **"Is there a working example that demonstrates the system?"**

### Files to Read

#### 9.1 Examples Search
**Files:** Search for `examples/`, `demos/`, `tutorials/`

**Tasks:**
- [ ] List all example files
- [ ] Check if examples are runnable
- [ ] Verify if examples show remote execution
- [ ] Test if examples are documented

**Report Format:**
```markdown
## 9.1 Examples Analysis

### Example files found
1. [file path]
   - Purpose: [what it demonstrates]
   - Runnable: YES/NO
   - Shows remote execution: YES/NO
   - Documentation: [where]

### Missing examples
Critical examples that DON'T exist:
- [ ] Simple remote execution demo
- [ ] LLM optimization demo
- [ ] Vision model demo
- [ ] Multi-modal demo

### Example: Simple demo
```python
# Paste simple_demo.py if it exists
# Otherwise note: FILE DOES NOT EXIST
```

### Critical finding
[Describe state of examples/demos]
```

#### 9.2 README/Docs Check
**Files:** `README.md`, `docs/` directory

**Tasks:**
- [ ] Check if README has "Quick Start"
- [ ] Verify if instructions are complete
- [ ] Check if installation instructions work
- [ ] Verify if examples are linked

**Report Format:**
```markdown
## 9.2 Documentation Analysis

### Quick Start
README has Quick Start section: YES/NO
If yes:
```markdown
# Paste Quick Start section
```

### Installation completeness
Steps to install:
1. [step]
2. [step]
...
Missing steps: [list if any]

### Running examples
Documentation shows how to run examples: YES/NO
If yes, paste instructions:
```

### Critical finding
[Describe if someone can actually get started]
```

---

## Part 10: Baseline Comparison Setup

### Objective
Answer: **"Is there a baseline implementation to compare against?"**

### Files to Read

#### 10.1 Baseline Search
**Files:** Search for "baseline", "naive", "vanilla"

**Tasks:**
- [ ] Find baseline implementation
- [ ] Check what baseline does
- [ ] Verify if baseline is comparable
- [ ] Check if comparison code exists

**Report Format:**
```markdown
## 10.1 Baseline Analysis

### Baseline implementation found
- File: [path] OR "NOT FOUND"
- What it does: [description]
- How it differs from Genie: [differences]

### Comparison code
Code that compares Genie to baseline:
- File: [path] OR "NOT FOUND"
- Metrics compared: [list]

### Example comparison
```python
# Paste comparison code if exists
```

### Critical finding
[Describe if baseline exists and is usable]
```

---

## Final Report Format

After completing all sections, provide a summary:

```markdown
# Code Review Summary

## Executive Findings

### 1. End-to-End Execution: [WORKING/BROKEN/INCOMPLETE]
**Evidence:** [summarize findings from Part 1]
**Impact:** [HIGH/MEDIUM/LOW]

### 2. Transport Layer: [PRODUCTION/DEVELOPMENT/MOCK]
**Evidence:** [summarize findings from Part 2]
**Impact:** [HIGH/MEDIUM/LOW]

### 3. C++ Integration: [ACTIVE/INACTIVE/BROKEN]
**Evidence:** [summarize findings from Part 3]
**Impact:** [HIGH/MEDIUM/LOW]

### 4. Performance Data: [EXISTS/MISSING]
**Evidence:** [summarize findings from Part 4]
**Impact:** [HIGH/MEDIUM/LOW]

### 5. Static Analysis: [WORKING/BROKEN/STUB]
**Evidence:** [summarize findings from Part 5]
**Impact:** [MEDIUM/LOW]

### 6. Optimizations: [IMPLEMENTED/STUB/MOCK]
**Evidence:** [summarize findings from Part 6]
**Impact:** [HIGH/MEDIUM/LOW]

### 7. Test Reality: [REAL/MOSTLY_MOCKED/FULLY_MOCKED]
**Evidence:** [summarize findings from Part 7]
**Impact:** [MEDIUM/LOW]

### 8. Scope: [FOCUSED/BLOATED/OVERWHELMING]
**Evidence:** [summarize findings from Part 8]
**Impact:** [HIGH/MEDIUM/LOW]

### 9. Demos: [WORKING/BROKEN/MISSING]
**Evidence:** [summarize findings from Part 9]
**Impact:** [HIGH]

### 10. Baseline: [EXISTS/MISSING]
**Evidence:** [summarize findings from Part 10]
**Impact:** [HIGH]

## Critical Blockers

Issues that prevent system from working:
1. [Issue] - Severity: [HIGH/MEDIUM/LOW]
2. [Issue] - Severity: [HIGH/MEDIUM/LOW]

## Action Items Priority Matrix

### P0 - Blocking (Must fix to have working system)
1. [Action item based on findings]
2. [Action item based on findings]

### P1 - Critical (Needed for OSDI submission)
1. [Action item based on findings]
2. [Action item based on findings]

### P2 - Important (Should have)
1. [Action item based on findings]

### P3 - Nice to have (Future work)
1. [Action item based on findings]

## Recommended Immediate Actions (This Week)

Based on all findings, here's what to do next:

**Day 1-2:**
- [ ] [Specific action with file:line references]
- [ ] [Specific action with file:line references]

**Day 3-4:**
- [ ] [Specific action with file:line references]
- [ ] [Specific action with file:line references]

**Day 5:**
- [ ] [Specific action with file:line references]

## Answers to Advisor's Questions

### 1. Can you send a tensor from GPU A to GPU B?
**Answer:** [YES/NO/PARTIALLY]
**Evidence:** [cite code locations]

### 2. What's the actual throughput?
**Answer:** [Number or "NOT MEASURED"]
**Evidence:** [cite benchmark location or "NO BENCHMARKS"]

### 3. Is C++ data plane being called?
**Answer:** [YES/NO/CONDITIONALLY]
**Evidence:** [cite code locations]

### 4. When does FX tracing happen?
**Answer:** [Describe with evidence]

### 5. Are optimizations measurably beneficial?
**Answer:** [YES/NO/UNKNOWN]
**Evidence:** [cite code locations]

## Risk Assessment for OSDI Deadline

**Likelihood of having working system:** [%]
**Likelihood of having evaluation:** [%]
**Likelihood of making deadline:** [%]

**Biggest risks:**
1. [Risk with mitigation]
2. [Risk with mitigation]

## Recommended Scope Changes

Based on findings:
- **Remove:** [Features to remove]
- **Defer:** [Features to defer]
- **Focus on:** [Core features to focus on]
```

---

## Instructions for AI Agent

1. **Read files systematically** following the order above
2. **Be precise** - cite line numbers and paste code
3. **Be honest** - if something doesn't work, say so clearly
4. **Provide evidence** - every claim needs code citation
5. **Think critically** - if code seems wrong, explain why
6. **Be actionable** - your findings should lead to clear next steps

### Output Format

Generate a single markdown document with:
1. All section reports filled in
2. Code citations with line numbers
3. Final summary report
4. Prioritized action items

### Time Estimate

This review should take approximately:
- Reading code: 2-3 hours
- Analysis: 1-2 hours
- Report writing: 1 hour
- **Total: 4-6 hours**

But the output will be invaluable for planning next steps.

---

## Example of Good vs Bad Reporting

### ❌ Bad Report
```
## Transport Analysis
The transport layer looks incomplete.
```

### ✅ Good Report
```
## Transport Analysis

### send_tensor() implementation (transport_coordinator.py:245-267)
```python
245: async def send_tensor(self, tensor, target_node):
246:     # Create context
247:     context = TransferContext(...)
248:     
249:     # Check if C++ available
250:     if self.data_plane is not None:
251:         return await self._start_data_plane_send(context)
252:     else:
253:         # Fallback... but fallback is not implemented!
254:         raise NotImplementedError("Fallback transport not implemented")
```

### Critical Finding
**The transport layer has no working fallback.** If C++ data plane fails to initialize (line 250), the system raises NotImplementedError (line 254). This means:
1. System cannot work without C++ library
2. No graceful degradation
3. Testing requires C++ compilation

**Recommendation:** Implement TCP fallback at line 253-254:
```python
return await self._tcp_fallback_send(context)
```
```

---

**Start with Part 1** and work through systematically. Each part builds on previous findings.

Good luck! 🔍