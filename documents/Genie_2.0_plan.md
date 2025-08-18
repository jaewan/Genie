# Genie 2.0: From Research to Production

**Document Purpose:** This document outlines the strategic vision, architectural evolution, and phased transition plan for advancing Genie from a version 1.0 research prototype to a version 2.0 production-ready, open-source disaggregation framework.

**Target Audience:** Genie Development Team, Project Stakeholders, and future Open-Source Contributors.

---

## 1. Executive Summary

Genie 1.0 is designed to prove a critical scientific point: **framework-level disaggregation with semantic awareness is not only feasible but also highly performant.** Its architecture is focused on demonstrating minimal overhead and validating the core concepts for an academic audience.

Genie 2.0 will build on this success, transitioning the project from a proof-of-concept to a **robust, developer-centric, and community-driven platform.** The goal is to create a system that is not just performant but also **usable, debuggable, extensible, and trustworthy.** This evolution is essential for attracting a diverse user base and establishing Genie as a foundational technology in modern AI infrastructure.

---

## 2. Guiding Principles for Genie 2.0

* **Developer Experience is Paramount:** The system must be easy to use, debug, and reason about. We will trade a small amount of "magic" for a large gain in transparency and predictability.
* **Modularity over Monolith:** The architecture must be decoupled to encourage community contributions and support for multiple ML frameworks and hardware backends.
* **Trust through Verifiability:** Developers must be able to inspect, understand, and even override the system's decisions. Performance must be predictable and reproducible.
* **Community First:** The architecture and development process will be designed to be open and accessible, fostering a vibrant ecosystem of plugins, backends, and contributors.

---

## 3. Genie 2.0 High-Level Architecture and Comparison

### Genie 2.0 Architecture Diagram

The Genie 2.0 architecture is a modular, decoupled system where components interact through a standardized `ExecutionPlan` artifact. This design promotes extensibility and clear separation of concerns.


```
              ┌──────────────────────────────────────────┐
              │           User Application               │
              │ (PyTorch, JAX, ONNX Runtime, etc.)       │
              └────────────────────┬─────────────────────┘
                                   │ Framework-Specific Ops
                                   ▼

╔════════════════════════╤════════════════════════╤═══════════════════════╗
║         FRONTENDS      │      CORE SCHEDULER      │       BACKENDS        ║
╟────────────────────────┼────────────────────────┼───────────────────────╢
║ ┌────────────────────┐ │                        │ ┌───────────────────┐ ║
║ │  PyTorch Frontend  │ │                        │ │ Local Emulation   │ ║
║ │ (LazyTensor Engine)├─┐                      ┌─▶│ Backend (Debug)   │ ║
║ └────────────────────┘ │                        │ └───────────────────┘ ║
║                        │                        │                       ║
║ ┌────────────────────┐ │   ┌────────────────┐   │ ┌───────────────────┐ ║
║ │   JAX/ONNX         ├─┼──▶│ Profile-Guided │───┼─▶│  Zero-Copy RDMA   │ ║
║ │   Frontend (Future)│ │   │  Optimization  │   │ │  Backend          │ ║
║ └────────────────────┘ │   │      Engine    │   │ └───────────────────┘ ║
║                        │   └────────────────┘   │                       ║
║                        │                        │ ┌───────────────────┐ ║
║                        │                        │ │ Kubernetes/SLURM  │ ║
║                        └────────────────────────┼─▶│ Backend (Future)  │ ║
║                        ┌──────────────────┐     │ └───────────────────┘ ║
║                        │ Tunable Policies │     │                       ║
║                        └──────────────────┘     │                       ║
╚════════════════════════╧════════════════════════╧═══════════════════════╝
│                      │                      │
│ Produces             │ Annotates/Optimizes  │ Consumes
▼                      ▼                      ▼
┌───────────────────────────────────────────────────────────────────┐
│          Standardized Artifact: The `ExecutionPlan` (SRG)         │
└───────────────────────────────────────────────────────────────────┘

````

### Key Differences from Genie 1.0

Genie 2.0 refactors the monolithic research prototype into a production-ready, modular system.

* **What's New or Integrated:** `ExecutionPlan` as a standardized artifact, pluggable Frontends & Backends, a Local Emulation Backend for debugging, a Profile-Guided Optimization (PGO) Engine, and a high-level, Tunable Policy API.
* **What's Refactored:** The monolithic logic of Genie 1.0 is cleanly separated into the new `Core Scheduler` and `Backends`. Implicit scheduling is replaced by the explicit production and consumption of the `ExecutionPlan`.

---

## 4. A Universal Platform for Disaggregation

To fulfill the vision of Genie as a universal system, we will evolve the low-level "Hook" mechanism of Genie 1.0 into a high-level, developer-centric **Policy API**. This allows developers to express complex disaggregation strategies declaratively, focusing on *what* they want to achieve, while the Genie platform handles *how* to execute it.

### High-Level Policy API: From Hooks to Declarative Intent

Instead of writing imperative code to intercept and manipulate the computation graph, developers will use a simple, idiomatic Python decorator-based API to apply disaggregation policies to their models.

**Example: Implementing the LLM Prefill/Decode Pattern with a Custom Policy**

```python
from genie import policies
import torch.nn as nn

# 1. Developer provides the business logic to identify execution phases.
#    This logic is simple, self-contained, and easy to test.
def is_prefill_phase(module, inputs):
    """Selector function: returns True if this is a prefill step."""
    # Prefill is characterized by processing a prompt with sequence length > 1
    return inputs[0].shape[1] > 1

def is_decode_phase(module, inputs):
    """Selector function: returns True if this is a decode step."""
    # Decode is auto-regressive with sequence length == 1
    return inputs[0].shape[1] == 1

# 2. Developer declaratively applies a disaggregation strategy to their model.
#    This policy is readable and clearly expresses intent.
@policies.define_disaggregation(
    # Phase 1: Prefill
    policies.Phase(
        name="prefill",
        selector=is_prefill_phase,
        # Request a pool of 8 H100s for this parallelizable phase
        resource_policy=policies.ScaleOut(gpus=8, hardware_type="H100")
    ),
    # Phase 2: Decode
    policies.Phase(
        name="decode",
        selector=is_decode_phase,
        # Collapse to a single A100 for this memory-bound phase
        resource_policy=policies.ScaleTo(gpus=1, hardware_type="A100"),
        # Critical Hint: Ensure the KV Cache is physically co-located
        # with the GPU executing this phase to avoid network transfers.
        data_dependency=policies.ColocateWith("kv_cache")
    )
)
class MyAwesomeLLM(nn.Module):
    # ... standard PyTorch model implementation ...
    pass
````

This declarative approach is vastly superior to low-level hooks. It's more readable, less error-prone, and cleanly separates the disaggregation strategy from the core model logic.

### A Library of Disaggregation Patterns

Most users will not need to write custom policies. Genie 2.0 will ship with a rich library of pre-built, optimized disaggregation patterns that cover common use cases. Developers can apply these with a single line of code.

**Example Usage:**

```python
from genie.patterns import llm, vision

# Apply the standard, optimized Prefill/Decode pattern
@llm.prefill_decode_disaggregation
class MyLanguageModel(nn.Module):
    ...

# Automatically apply pipeline parallelism to a large vision model
@vision.pipeline_parallel(num_stages=4)
class MyVisionTransformer(nn.Module):
    ...
```

**Initial Library of Patterns will include:**

  * `llm.prefill_decode_disaggregation`: The canonical pattern for high-throughput LLM serving.
  * `vision.pipeline_parallel`: Automatically partitions a sequential model (like a ViT or ConvNeXt) across multiple accelerators to hide latency.
  * `moe.expert_parallel`: A specialized policy for Mixture-of-Experts models that intelligently routes requests to remote "expert" GPUs.
  * `multimodal.parallel_fusion`: A policy for models with distinct branches (e.g., vision and language) that executes the branches in parallel on heterogeneous hardware before fusing the results.

This library makes powerful disaggregation techniques accessible to all developers, while the underlying Policy API provides the flexibility for researchers to experiment with novel schemes.

-----

## 5\. Phased Transition and Open-Source Plan

The transition will occur in four distinct phases, moving from the academic prototype to a community-driven platform.

### Phase 1: Success and Open-Sourcing of Genie 1.0 (Current Goal)

  * **Action:** Complete the implementation and evaluation of Genie 1.0.
  * **Goal:** Achieve publication in a top-tier systems conference (e.g., OSDI, SOSP, NSDI).
  * **Action:** Release Genie 1.0 to the public under a permissive license (e.g., Apache 2.0).

### Phase 2: Architecting Genie 2.0 (Post-Publication)

  * **Action:** Formalize the `ExecutionPlan` specification and refactor the Genie 1.0 codebase into the three distinct components (Frontend, Core, Backend).
  * **Goal:** Create a working internal version of Genie 2.0 with feature parity to 1.0 but following the new, modular architecture.

### Phase 3: Building the Developer Experience (Community Focus)

  * **Action:** Develop the **Local Emulation Backend**, the **Policy API**, and the initial **Library of Patterns**.
  * **Action:** Implement the **PGO Engine** and visualization tools for inspecting execution plans.
  * **Goal:** Attract the first external contributors by focusing on usability and debuggability.

### Phase 4: Ecosystem Expansion and Production Readiness

  * **Action:** Develop a second frontend (e.g., `genie-jax`) and backend (e.g., Kubernetes) to prove the architecture's viability.
  * **Action:** Harden the system for multi-tenancy, security, and fault tolerance.
  * **Goal:** Onboard the first stable, production-oriented users.

-----

## 6\. Conclusion

Genie 1.0 is poised to make a significant academic impact. The Genie 2.0 plan provides a clear path to translate that possibility into a practical, powerful, and community-driven reality. By building a universal platform centered on a high-level Policy API, we can empower the entire ML community to harness the power of disaggregation.

```
