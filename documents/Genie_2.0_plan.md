# Genie 2.0: From Research to Production

**Document Purpose:** This document outlines the strategic vision, architectural evolution, and phased transition plan for advancing Genie from a version 1.0 research prototype to a version 2.0 production-ready, open-source disaggregation framework.

**Target Audience:** Genie Development Team, Project Stakeholders, and future Open-Source Contributors.

-----

## 1\. Executive Summary

Genie 1.0 is designed to prove a critical scientific point: **framework-level disaggregation with semantic awareness is not only feasible but also highly performant.** Its architecture is focused on demonstrating minimal overhead and validating the core concepts for an academic audience.

Genie 2.0 will build on this success, transitioning the project from a proof-of-concept to a **robust, developer-centric, and community-driven platform.** The goal is to create a system that is not just performant but also **usable, debuggable, extensible, and trustworthy.** This evolution is essential for attracting a diverse user base and establishing Genie as a foundational technology in modern AI infrastructure.

-----

## 2\. Guiding Principles for Genie 2.0

The transition from 1.0 to 2.0 will be guided by the following core principles:

  * **Developer Experience is Paramount:** The system must be easy to use, debug, and reason about. We will trade a small amount of "magic" for a large gain in transparency and predictability.
  * **Modularity over Monolith:** The architecture must be decoupled to encourage community contributions and support for multiple ML frameworks and hardware backends.
  * **Trust through Verifiability:** Developers must be able to inspect, understand, and even override the system's decisions. Performance must be predictable and reproducible.
  * **Community First:** The architecture and development process will be designed to be open and accessible, fostering a vibrant ecosystem of plugins, backends, and contributors.

-----

## 3\. Genie 2.0 High-Level Architecture and Comparison

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
```

### Key Differences from Genie 1.0

Genie 2.0 refactors the monolithic research prototype into a production-ready, modular system. The evolution is not just about adding features, but fundamentally changing how the components interact.

**What's New or Integrated into a Formal Component:**

  * **`ExecutionPlan` as a Standardized Artifact:** In Genie 1.0, the computation graph is an internal, transient data structure. In 2.0, it becomes a **formal, serializable artifact**. This is the most crucial change, enabling the entire decoupled architecture.
  * **Pluggable Frontends & Backends:** Genie 1.0 has a single, hard-wired PyTorch integration and RDMA execution path. Genie 2.0 splits this into a formal **Frontend** and **Backend** system, allowing for community contributions like JAX support or a Kubernetes backend.
  * **Local Emulation Backend:** This is a brand new component, essential for developer experience. It allows users to debug complex disaggregation logic on a single machine without specialized hardware, a capability completely absent in the research-focused 1.0.
  * **Profile-Guided Optimization (PGO) Engine:** The `Core Scheduler` in 2.0 formalizes the optimization engine of 1.0 and enhances it with a PGO component. This grounds scheduling decisions in empirical data rather than relying solely on the static heuristics of 1.0.
  * **Tunable Policies:** Genie 1.0's scheduler is a "black box." Genie 2.0 introduces an explicit API for developers to inject hints and override decisions, making the system transparent and controllable.

**What's Removed or Refactored from Genie 1.0:**

  * **Monolithic Structure:** The biggest removal is the single, tightly-coupled codebase. The logic from Genie 1.0's `Semantic Analysis`, `Optimization & Execution`, and `Zero-Copy Runtime` layers is **refactored and cleanly separated** into the new `Core Scheduler` and `Zero-Copy RDMA Backend`.
  * **Implicit Scheduling:** The direct, internal calls between graph building and execution in 1.0 are replaced by the explicit production and consumption of the `ExecutionPlan`. The scheduler no longer directly drives execution; it produces a plan that a backend can consume.

In essence, the transition moves Genie from a system where all components are aware of each other to one where they only need to understand a single, common language: the `ExecutionPlan`.
-----

## 4\. Targeted Improvements


###  Addressing the "Oracle" and "Monolithic" Problems

The Genie 2.0 architecture solves these two issues with distinct, targeted design changes:

1.  **From Oracle to Co-Pilot:** The problem of the "oracle" scheduler, which was expected to magically know the optimal execution plan, is solved by making the scheduler interactive and data-driven.
    * **Profile-Guided Optimization (PGO) Engine:** This new component in the **Core Scheduler** moves away from relying solely on static heuristics. It grounds scheduling decisions in **empirical performance data** gathered by actually running parts of the model on the target hardware.
    * **Tunable Policies:** The architecture explicitly includes an interface for developers to provide hints or override the scheduler's decisions. This gives the user an "escape hatch," transforming the scheduler from an opaque, brittle authority into a powerful assistant that the developer ultimately controls.

2.  **From Monolithic Framework to Decoupled Architecture:** The problem of the monolithic, tightly-coupled system is solved by refactoring the entire architecture around a central, standardized artifact: the **`ExecutionPlan`**.
    * Genie 1.0 is a single, integrated system. Genie 2.0 is broken into three independent component types: **Frontends**, the **Core Scheduler**, and **Backends**.
    * This modularity means the `genie-pytorch` frontend can evolve independently of the `Zero-Copy RDMA` backend. This design is crucial for encouraging community contributions, as a developer can build a new backend without needing to understand the complexities of the PyTorch frontend, and vice-versa.

---

### How the Standardized Graph Mitigates Brittleness

The standardized `ExecutionPlan` (or Semantically-Rich Graph, SRG) is the key to mitigating the brittleness of the semantic analysis layer (FX static analysis, lazy tensor capture, hooks). It achieves this by introducing a crucial layer of **abstraction and validation**.

Think of the `ExecutionPlan` as being analogous to a compiler's Intermediate Representation (IR) like LLVM IR. The process works like this:

1.  **Encapsulation of Brittle Logic:** All the potentially brittle capture logic is now confined to the **Frontend**. The `genie-pytorch` frontend is solely responsible for translating the dynamic world of PyTorch—with all its FX graph quirks and changing APIs—into the stable, well-defined `ExecutionPlan` format. If a new PyTorch version breaks the FX tracing, only the frontend needs to be updated. The Core Scheduler and all the Backends remain completely unaffected because they only consume the standardized plan. This contains the "blast radius" of any single point of failure.

2.  **Introduction of a Validation Layer:** Because the `ExecutionPlan` is a formal, standardized artifact, it can be validated. After a frontend produces a plan, it can be passed through a validation step to check for correctness (e.g., graph is a valid DAG, tensor shapes are consistent, metadata is well-formed) *before* it is sent to the complex scheduler or expensive hardware. This catches errors early, preventing a faulty graph from causing difficult-to-debug failures during remote execution.

3.  **Simplifies the Capture Goal:** The goal of the semantic capture layer is no longer to produce a perfect, directly executable plan. Its goal is now simpler: to produce a **correct and valid `ExecutionPlan`**. The much more complex task of *optimizing* and *executing* that plan is handled by different components (the Core Scheduler and Backends), which can now rely on receiving a well-structured, predictable input.

In short, the standardized graph decouples the **"messy act of semantic capture"** from the **"structured act of scheduling and execution."** This separation makes the system more robust, easier to debug, and far more extensible than the original monolithic design.

-----

## 5\. Phased Transition and Open-Source Plan

The transition will occur in four distinct phases.

### Phase 1: Success and Open-Sourcing of Genie 1.0 (Current Goal)

  * **Action:** Complete the implementation and evaluation of Genie 1.0.
  * **Goal:** Achieve publication in a top-tier systems conference (e.g., OSDI, SOSP, NSDI).
  * **Action:** Upon acceptance, clean up the codebase, add documentation, and release Genie 1.0 to the public under a permissive license (e.g., Apache 2.0). This will build initial interest and validate the core concepts with a wider audience.

### Phase 2: Architecting Genie 2.0 (Post-Publication)

  * **Action:** Formalize the `ExecutionPlan` specification. This is the most critical design step.
  * **Action:** Refactor the Genie 1.0 codebase into the three distinct components (Frontend, Core, Backend). Initially, these will still be tightly integrated.
  * **Goal:** Create a working internal version of Genie 2.0 that has feature parity with 1.0 but follows the new, modular architecture.

### Phase 3: Building the Developer Experience (Community Focus)

  * **Action:** Develop the **Local Emulation Backend**. This is the top priority for attracting contributors, as it makes development accessible without specialized hardware.
  * **Action:** Implement the **Profile-Guided Optimization (PGO) Engine** and the visualization tools for inspecting execution plans.
  * **Action:** Create the **Semantic Debugging** layer to provide actionable error messages.
  * **Goal:** Attract the first external contributors. The project's success will be measured by its usability and debuggability.

### Phase 4: Ecosystem Expansion and Production Readiness

  * **Action:** Develop a second frontend (e.g., `genie-jax` or `genie-onnx`) to prove the viability of the decoupled architecture.
  * **Action:** Develop a second backend (e.g., a Kubernetes backend that requests transient pods) to demonstrate platform flexibility.
  * **Action:** Harden the system for multi-tenancy, security, and fault tolerance.
  * **Goal:** Onboard the first stable, production-oriented users who rely on Genie for real workloads.

-----

## 6\. Conclusion

Genie 1.0 is poised to make a significant academic impact by demonstrating what is possible. The Genie 2.0 plan provides a clear and strategic path to translate that possibility into a practical, powerful, and community-driven reality. By prioritizing developer experience, modularity, and verifiability, we can build a system that not only optimizes resource utilization but also empowers the next generation of AI development.
