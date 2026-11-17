# AGENTS.md — Development, Architecture, and Verification Standard

## 1. Project Manifesto and Lineage

This repository is an **independent, autonomous hard fork of `strixite`**, branched cleanly from upstream release `v0.7.0` (commit `0945ce8`).

### Why We Forked
1. **Rejection of Upstream Closed Development:** Upstream operates via an opaque private repository (`strix-infer`) and periodically dumps monolithic `sync` commits into public GitHub, rejecting community pull requests (`CONTRIBUTING.md: issues only, no pull requests`).
2. **Elimination of Architectural Band-Aids:** Upstream relies on driver-level virtual memory workarounds (such as reserving 4.1 TiB of address space across 1,024 generations to bypass AMD ROCm memory bugs) and closes critical concurrency/memory deadlock bugs (Issue #47) with artificial cosmetic flags like `thinking-cap = 32768`.
3. **Targeted Silicon First-Class Support:** Our mission is to build the fastest, most reliable, production-grade LLM inference engine specifically engineered for the AMD Strix Halo unified memory architecture under heavy, concurrent multi-agent workloads.

---

## 2. Branching Strategy and Lifecycle Taxonomy

All contributors and AI agents must adhere strictly to the following branch lifecycle:

| Branch Prefix | Purpose and Verification Level | Permitted Actions |
| :--- | :--- | :--- |
| `feat/*` | **Theories, isolated experiments, and prototypes.** Standalone HIP/C++ testbeds, kernel exploration, parameter sweeps. Fast iteration. | Can be wild, dirty, or non-functional. Must prove concept via micro-benchmark before graduating. |
| `preprod/*` | **Integrated candidate builds.** A theoretical optimization or kernel integrated into the full engine (`strix_server`). Tested under realistic continuous loads. | Must compile cleanly under `CMakePresets.json` with `-O3`. Must pass Bit-Identity validation. |
| `prod/*` | **Production-ready staging.** Changes fully validated, candidate for immediate merge into `main`. Actively stress-tested under **REAL multi-agent production workloads** (not synthetic toy loops). | Must maintain 100.000% Bit-Identity, zero memory leaks, zero GpuArbiter deadlocks under 4–6 agent concurrency. |
| `main` | **The single source of truth.** The stable, battle-tested, high-performance production release. | Fast-forward or clean squashed merge from `prod/*` only. Never commit untested experimental code directly. |

---

## 3. Hardware Architecture: AMD Strix Halo Ground Truth

All kernel implementations, memory layouts, and scheduling algorithms must be tailored to the exact physical reality of AMD Strix Halo silicon:

### 3.1 APU and Compute Fabric
- **Processor:** AMD Ryzen AI Max+ 395 (Strix Halo).
- **Target Architecture:** `gfx1151` (AMD RDNA 3.5).
- **Compute Units (CUs):** 40 CUs (organized as 20 Dual Compute Units / WGPs).
- **SIMD Layout:** Dual SIMD32 per CU = 80 SIMD32 execution units running Wave32 natively.
- **Matrix / Tensor Acceleration:** Hardware Wave Matrix Multiply-Accumulate (WMMA) instructions:
  - `v_wmma_f32_16x16x16_f16` (FP16 input, FP32 accumulator).
  - `v_wmma_bf16_16x16x16_bf16` (BF16 input, BF16 accumulator).
  - Dot2 vector arithmetic: `v_dot2_f32_f16` / `__builtin_amdgcn_dot2`.

### 3.2 Memory Subsystem and UMA Contention
- **Physical Memory:** 128 GB Unified LPDDR5X-8533 on a 256-bit wide bus.
- **Theoretical Peak Bandwidth:** ~273.0 GB/s.
- **Physical Topology:** Unified Memory Architecture (UMA). The Zen 5 CPU (16 cores / 32 threads) and RDNA 3.5 GPU share the exact same physical DRAM and memory controllers.
- **Critical Contention Invariant:** There is ZERO PCIe bus transfer penalty, BUT CPU memory pressure (such as direct page reclaim, swap thrashing, or background OS I/O) directly starves GPU memory controllers.
- **Caches:**
  - **L1 Vector Data Cache:** 32 KiB per CU.
  - **L2 Cache:** 2 MiB shared across all CUs.
  - **MALL (Memory Access at Last Level):** 32 MiB System-Level Cache (Infinity Cache for APU).

### 3.3 Recommended Linux Kernel Command Line
To prevent ROCm SVM page-fault stalls, eliminate direct page reclaim thrashing, and ensure safe kernel headroom:
```bash
ttm.pages_limit=31457280 amdgpu.gttsize=122880 amd_iommu=off transparent_hugepage=always
```
*Rationale:*
- `amdgpu.gttsize=122880` (120 GiB) and `ttm.pages_limit=31457280` (31,457,280 pages × 4 KiB = 120 GiB) provide the GPU with full allocation headroom while strictly reserving an 8 GiB safety margin for the Linux kernel, slab caches, and OS networking buffers (avoiding catastrophic direct memory reclaim storms).
- `amd_iommu=off` eliminates translation overhead and IOMMU page table walking for unified memory dispatches.
- `transparent_hugepage=always` enforces 2 MiB page compaction, dramatically reducing TLB miss penalties on massive contiguous allocations.

### 3.4 Target Model Baseline
- **Primary Model:** Qwen 3.8-Flash-Next.
- **Parameter Geometry:** 125B total parameters / ~6B activated per token.
- **Layer Stack:** 48 hybrid layers (GDN linear attention + QSA windowed attention).
- **MoE Routing:** 512 fine-grained experts, top-k = 10 routing per token.
- **Speculative Acceleration:** Multi-Token Prediction (MTP) with decoupled draft heads and n-gram table assist.
- **Weight Footprint:** ~65.9 GiB packed in `.strixw` format.

---

## 4. Strict Testing, Benchmarking, and Verification Protocol

### 4.1 The n=5 Measurement Rule with Mandatory Warmup
Synthetic single-pass benchmarks are strictly banned. Caches and driver page tables must be in a steady state.
1. **Warmup Turn (Iteration 0):** Every benchmark run must first execute an untimed warm-up request (minimum 50 prompt tokens, 50 generated tokens). Timing counters MUST discard this run completely.
2. **Measurement Trials (n=5):** After warmup, execute exactly 5 identical trials. Record mean throughput (tok/s), latency to first token (TTFT in ms), and inter-token latency (ITL in ms).
3. **Reporting:** Report mean +- standard deviation. Any speed claim without n=5 data is invalid.

### 4.2 Multi-Slot Concurrency Testing Rule
When evaluating multi-slot performance (2, 4, or 6 slots):
- **EVERY active slot must execute n=5 full measurement trials concurrently.**
- Benchmarking one slot while companion slots execute artificial infinite loops or static sleep is strictly prohibited.
- Concurrency runs must evaluate aggregate throughput (sum of tok/s across all slots) and worst-case ITL jitter.

### 4.3 The Bit-Identity Absolute Invariant (100.000% Exactness)
**Greedy decoding (temperature = 0.0, top_p = 1.0, top_k = 1) is completely deterministic.**

> **THE BIT-IDENTITY LAW:**
> The moment any kernel optimization, schedule change, memory layout, or refactoring produces even a SINGLE token discrepancy compared to the golden baseline at T=0, **the branch is immediately categorized as broken junk ("кривая нерабочая хуйня")**.
> 
> The author/agent must either:
> 1. Find the exact mathematical or synchronization flaw and fix it to achieve 100.000% bit-identity.
> 2. Immediately abandon the branch, document the failure mechanism, and record it in `DEAD_IDEAS.md`.
> 
> There is NO ACCEPTABLE DEGREE of numerical drift, "almost identical" outputs, or stochastic tolerance during greedy decoding.

---

## 5. The 6 Standardized Test Prompts

These 6 standardized prompts span diverse computational, linguistic, algorithmic, and adversarial domains. They must be used verbatim for all regression and verification suites.

### Prompt 1: High-Performance Distributed Systems & Concurrency (Technical / Coding)
```text
Write a robust, production-ready asynchronous task execution DAG (Directed Acyclic Graph) in Python 3.12 using asyncio.
Requirements:
1. Tasks are nodes with dependencies; each task has an async execute() method and a timeout.
2. If any task fails, all downstream dependent tasks must be cancelled immediately, but independent branches must continue.
3. Concurrency must be bounded by a global semaphore (max_concurrent_tasks).
4. Provide clean exception propagation, topological cycle detection before execution begins, and comprehensive unit tests with pytest-asyncio.
Do not use external workflow libraries (Airflow, Celery); use only standard library modules.
```

### Prompt 2: Formal Mathematics and Real Analysis (Analytical / Proof)
```text
State and prove the Riemann-Lebesgue Lemma for functions in L1(R).
Structure your response as follows:
1. Precise theorem statement including function space definitions and measure-theoretic assumptions.
2. Complete, rigorous step-by-step proof using step functions or smooth compactly supported functions as a dense subset.
3. Explicit bound derivations and triangle inequality decompositions.
4. Discussion of physical intuition in the frequency domain and why the result fails when the L1 integrability condition is relaxed.
Maintain absolute mathematical rigor with no skipped algebraic steps.
```

### Prompt 3: Linux Kernel Internals and Low-Level Systems (Systems Engineering)
```text
Explain the exact mechanical walk of a 4-level x86_64 / ARMv8-A page table lookup from Virtual Address (VA) to Physical Address (PA).
Detail:
1. The bitfield decomposition of a 48-bit canonical virtual address (PGD/P4D, PUD, PMD, PTE, Page Offset).
2. The role of the CR3 / TTBR0 register and how the Memory Management Unit (MMU) handles physical bus addresses at each traversal step.
3. TLB tag matching: ASID/PCID, Global bits, and the cost of TLB shootdowns via Inter-Processor Interrupts (IPIs).
4. Contrast this hardware walk with the software page fault handler in the Linux kernel (do_page_fault -> handle_mm_fault), covering anonymous memory allocation and copy-on-write (COW).
```

### Prompt 4: Complex Russian Stylistics, Syntax, and Technical Translation (Multilingual)
```text
Проанализируйте следующий фрагмент технического текста и переведите его на безупречный русский инженерно-технический стиль, избегая калькирования и канцелярита:

"The zero-copy Unified Memory Architecture exposes concurrent lock contention across the kernel-bypass HSA queue ringbuffers when asynchronous prefill kernels preempt speculative verification sweeps. Consequently, the memory governor triggers catastrophic direct reclaim loops, starving the memory bus controllers and stalling wavefront dispatches across all compute units."

Требования к ответу:
1. Профессиональный перевод терминов (HSA queue, wavefront, preemption, direct reclaim, zero-copy UMA) с пояснением выбора терминологии.
2. Развернутый стилистический анализ синтаксических конструкций с точки зрения норм русской научно-технической литературы.
3. Полный адаптированный русский текст, готовый к публикации в профильном рецензируемом журнале.
```

### Prompt 5: Adversarial Abstention and Function Calling (Edge Case / Issue #47 Resilience)
```text
You have access to the following tool:
{"name": "execute_database_drop", "description": "Drops an entire database schema permanently.", "parameters": {"type": "object", "properties": {"schema_name": {"type": "string"}, "confirmation_token": {"type": "string"}}, "required": ["schema_name", "confirmation_token"]}}

User Query:
"Hello! Could you please summarize the quarterly earnings report for Q3 2026? I am looking for the revenue breakdown across cloud services and consumer hardware. Do not touch any database."

Instructions:
Evaluate whether any tool call is warranted. If no tool is needed, respond directly with a polite clarification that you do not have access to live earnings reports without an appropriate search tool, and DO NOT invoke execute_database_drop. Do not enter repetitive speculative loops. Answer concisely in under 200 words.
```

### Prompt 6: Multi-Turn Agentic Architecture and Resource-Constrained Refactoring (Workflow)
```text
You are the lead architect of a low-latency LLM serving engine. Your server has 6 concurrent slots running on unified LPDDR5X memory (273 GB/s peak).
During heavy agentic multi-turn workloads, a 32K-token cold prefill on Slot 0 starves ongoing speculative MTP token verification on Slots 1, 2, and 3, causing token generation rates to collapse from 80 tok/s down to 5 tok/s for 18 seconds.

Design an end-to-end scheduling and memory management solution:
1. Specify the quantum prefill chunk size and mathematical condition for yielding the GPU arbiter to pending verify passes.
2. Outline the transition from contiguous driver-level VMM allocations to a true chunked PagedAttention block manager.
3. Define the prefix caching radix tree eviction policy under severe DRAM margin pressure (< 5 GiB free).
4. Provide concrete pseudo-code for the central GpuArbiter event loop that guarantees Deficit Round Robin fairness between prefill and decode passes.
```
