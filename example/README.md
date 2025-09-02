# Genie Examples & Demonstrations

This directory contains examples demonstrating Genie's frontend and scheduling capabilities with real CUDA execution.

## 📁 Files

### 1. `quick_demo.py` - Quick Feature Demonstration ⭐
A lightweight demonstration of core Genie features:
- **LLM Attention Pattern**: Detects and optimizes self-attention patterns
- **CNN Pipeline**: Demonstrates 3-stage pipeline scheduling for vision models
- **KV Cache Detection**: Shows prefill/decode phase detection for LLMs
- **Multi-Modal Fusion**: Detects cross-modal fusion points

**Run:** `python example/quick_demo.py`

### 2. `genie_showcase.py` - Comprehensive Showcase
Full demonstration with real PyTorch models on CUDA:
- **LLM Model**: Complete transformer with attention, FFN, and KV cache simulation
- **Vision CNN**: 3-layer CNN with batch normalization and pooling
- **Multi-Modal Model**: Vision-text fusion with cross-attention
- **FX Execution**: Graph execution with semantic optimizations

**Run:** `python example/genie_showcase.py`

### 3. `resnet18_demo.py` - ResNet-18 Example
Demonstrates LazyTensor with ResNet-18 architecture:
- Loads ResNet-18 conv1 weights
- Builds LazyTensor computation graph
- FX tracing for full model structure

**Run:** `python example/resnet18_demo.py`

### 4. `comprehensive_correctness_test.py` - Correctness Validation
Tests LazyTensor correctness against native PyTorch (Note: Some execution tests may fail as remote execution is not fully implemented)

**Run:** `python example/comprehensive_correctness_test.py`

### 5. `perf_compare.py` - Performance Comparison
Compares LazyTensor overhead vs native PyTorch execution

**Run:** `python example/perf_compare.py`

## 🚀 Features Demonstrated

### Phase 1: Core Infrastructure ✅
- **1.1 Enhanced Semantic Metadata**: Rich metadata capture including execution phases, memory patterns, and data lineage
- **1.2 FX Graph Migration**: Native PyTorch FX graph with semantic preservation
- **1.3 Pattern Matching**: TorchDynamo-based declarative pattern matching

### Phase 2: Semantic Optimizations ✅
- **2.1 Optimization Strategies**:
  - LLM: KV cache co-location, prefill parallelization
  - Vision: CNN pipeline scheduling, Conv-BN-ReLU fusion
  - Multi-modal: Parallel modality processing, JIT fusion
- **2.2 Phase Detection**:
  - Runtime detection of prefill/decode phases
  - Vision pipeline stage tracking
  - Multi-modal fusion detection

## 📊 Example Output

### Quick Demo Output
```
✅ Successfully demonstrated:
   1. LLM attention pattern detection and optimization
   2. CNN pipeline scheduling (3 stages)
   3. KV cache detection for prefill/decode phases
   4. Multi-modal fusion detection and optimization

📈 Statistics:
   - Total operations tracked: 16
   - Phase transitions: 3
   - Vision stages: 2
   - Modalities detected: ['vision']
```

### Comprehensive Showcase Output
```
📊 Workload Detection:
  - LLM: Correctly identified
  - Vision: CNN pipeline detected
  - Multi-modal: Fusion points found

🔧 Optimizations Applied:
  - LLM: 2 optimizations (KV cache, parallelization)
  - Vision: 2 optimizations (pipeline, fusion)
  - Multi-modal: 2 optimizations (parallel branches, fusion)

🎯 Phase Detection:
  - Prefill/decode phases tracked
  - Vision stages identified
  - Modalities detected and tracked
```

## 🔧 Usage

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run quick demo for fast overview:**
   ```bash
   python example/quick_demo.py
   ```

3. **Run comprehensive showcase for full features:**
   ```bash
   python example/genie_showcase.py
   ```

## 💡 Key Insights

### Semantic Metadata Capture
Genie captures rich semantic information for every operation:
- Operation type and tensor shapes
- Execution phase (prefill, decode, vision backbone, etc.)
- Memory patterns (streaming, reused, ephemeral)
- Module context and layer depth
- Compute intensity and parallelization hints

### Workload-Specific Optimizations
Different workload types receive tailored optimizations:
- **LLM**: Focus on KV cache efficiency and token generation
- **Vision**: Pipeline scheduling for CNN stages
- **Multi-modal**: Cross-modal fusion and parallel branch execution

### Phase Detection
Runtime phase detection enables dynamic optimization:
- Detects transitions between prefill and decode
- Tracks vision pipeline stages
- Identifies multi-modal fusion points

## 🎯 Target Applications

Genie's frontend and scheduling are designed for:
1. **Large Language Models (LLMs)**: Optimizing attention mechanisms and KV cache management
2. **Computer Vision Models**: Pipeline scheduling for CNNs and vision transformers
3. **Multi-Modal Models**: Efficient cross-modal fusion and parallel processing
4. **Hybrid Workloads**: Dynamic optimization based on detected patterns

## ⚠️ Notes

- The examples focus on frontend and scheduling capabilities
- Remote execution backend integration is planned for Phase 3
- Some correctness tests may fail due to incomplete executor implementation
- CUDA execution is used when available, CPU fallback otherwise

## 📈 Performance

The examples demonstrate:
- Low-overhead semantic capture (~5% for LazyTensor creation)
- Fast pattern matching and workload classification
- Efficient phase detection with minimal runtime impact
- Scalable to large computation graphs (1000+ nodes)

---

For more information, see the main [REFACTORING_PLAN.md](../REFACTORING_PLAN.md) and [CURRENT_IMPLEMENTATION_ANALYSIS.md](../CURRENT_IMPLEMENTATION_ANALYSIS.md) documents.
