# Strixite: Dead Ideas & Architectural Graveyard

This document permanently records speculative decoding, kernel, and architectural ideas that were hypothesized, implemented, empirically benchmarked on AMD Strix Halo (Ryzen AI Max+ 395, GFX1151, 128 GB LPDDR5X-8533 UMA), and proven to be inferior, dead ends, or harmful to throughput.

Do NOT reimplement, resurrect, or spend further engineering time on these concepts unless underlying hardware architecture or foundational model paradigms fundamentally change.

---

## 1. AIMD Dynamic Draft Depth Controller (Adaptive Additive Increase / Multiplicative Decrease)

### Concept
Borrow TCP congestion control principles for speculative MTP draft depth:
- Dynamically track acceptance length `j` per verification round.
- On full acceptance (`j == draft_depth`): increase draft depth (`draft_depth += 1`, or `+2`).
- On early rejection (`j < threshold`): back off multiplicatively (`draft_depth = max(min_depth, draft_depth / 2)`).
- Search bounds explored: [2, 15] and [4, 12].

### Empirical Results (n=3 on 5 Engineering Domains)
- **Baseline (Static MTP 7):** 70.07 +- 0.57 tok/s
- **Pure AIMD (Dynamic depth 4..12):** 70.32 +- 0.01 tok/s (+0.25 tok/s, ~0.35% delta, statistically indistinguishable from run-to-run noise).
- **Combined (AIMD + Escalating Margin):** 71.18 tok/s vs **71.52 tok/s for Escalating Margin alone** (-0.34 tok/s regression).

### Why It Failed (Root Causes)
1. **MTP Draft Compute is Not Free on Strix Halo:**
   Unlike lightweight small draft models on discrete GPUs with high VRAM bandwidth, MTP draft steps on Strix Halo require sequential small GEMMs (FC projections) through the draft head. At depth 10..12, sequential draft latency grows from ~2.8 ms to ~5.5 ms per round.
2. **MoE Verification Batch Explosion:**
   When draft depth expands to 11..12, the verify batch size `T = draft_depth + 1` reaches 12..13. In Strixite's MoE implementation, batch sizes `T >= 6` force the verify forward pass to switch from single-token direct decode to grouped routing (`kGroupedMinTokens = 6`). At `T = 12..13`, tokens hit distinct experts across all 128 experts, saturating the 256-bit LPDDR5X memory bus (273 GB/s peak).
3. **Controller Lag & Autoregressive Phase Shifts:**
   Token predictability shifts abruptly between tokens (syntax boilerplate -> variable names -> complex algorithmic reasoning). By the time AIMD notices a 7-token acceptance streak and scales depth to 12, the model has entered an unpredictable reasoning span. The deep draft chain is rejected, wasting both the draft GEMMs and the heavy verify pass.
4. **Redundancy with Intra-Round Confidence Gating:**
   Escalating margin already terminates drafting at the exact token where top-1 confidence drops below threshold. Adding an external controller on top adds dead compute and oscillations without improving accuracy.

**Status: PERMANENTLY DEAD. DO NOT RESURRECT.**

---

## 2. Inter-Round Adaptive Margin Adjustment (Feedback on Confidence Threshold)

### Concept
Dynamically adjust `mtp_margin` (e.g. between 1.0 and 2.5) across verification rounds based on acceptance history:
- If previous round was fully accepted, loosen margin down towards 1.0 to encourage longer speculative chains.
- If previous round suffered early rejection, tighten margin up towards 2.5 to be more conservative.

### Empirical Results
- **Throughput:** 70.23 tok/s (+0.01% over baseline, zero meaningful gain).

### Why It Failed
1. **Token Entropy is Locally Markov, Not Inter-Round Stationary:**
   A successful acceptance streak in round N does not mean round N+1 will be predictable. Loosening margin to 1.0 causes immediate mispredictions on the next token boundary.
2. **Oscillatory Hysteresis:**
   A strict margin (2.5) after a misprediction prematurely truncates trivial tokens (braces, keywords, indentation) in the following round, hurting acceptance length. A loose margin (1.0) after a streak drafts speculative garbage that gets rolled back.

**Status: PERMANENTLY DEAD.**

---

## 3. Phase-Aware Drafting (<think> vs Output/Code)

### Concept
Detect generation phase via string matching / token ID state:
- Inside `<think>...</think>` tags: cap draft depth to 4 or 6 (assuming high-entropy reasoning).
- Outside `<think>` (in final answer / code): expand draft depth to 10 or 12.

### Empirical Results
- **Throughput:** 70.36 tok/s (+0.15% gain).

### Why It Failed
1. **Reasoning Models (Qwen 2.5, Qwen 3.5, Qwen4Exp) Spend 85-95% of Tokens in `<think>`:**
   Throttling draft depth inside `<think>` throttles almost the entire generation process.
2. **`<think>` is Not Uniformly High Entropy:**
   Reasoning text contains extensive structured spans: scratchpad code, mathematical formulations, boilerplate phrasing, and markdown formatting. Capping draft depth to 6 artificially curtailed valid 8-10 token speculative streaks.
3. **Logit Margin Already Self-Regulates:**
   Intra-round top-1 vs top-2 logit difference measures certainty directly at the token level, rendering coarse macro-state flags completely obsolete.

**Status: PERMANENTLY DEAD.**

---

## 4. Optimistic Probing / Speculative Exploration

### Concept
If drafting gets stuck at minimum depth (e.g. depth 3) for several consecutive rounds, force an optimistic probe with draft depth 7 or 8 to "break out" of low-depth plateaus.

### Empirical Results
- **Throughput:** 70.06 tok/s (-0.28% degradation).

### Why It Failed
1. **Low Acceptance is Not an Algorithmic Bug:**
   When acceptance length is low, it indicates genuinely high information entropy in the model's output distribution.
2. **Speculation is Strictly Prefix-Dependent:**
   If token 2 is rejected, tokens 3 through 7 are unconditionally discarded. Forcing depth 7 in high-entropy regions resulted in near-100% rejection rates for the speculative tail, needlessly consuming memory bandwidth.

**Status: PERMANENTLY DEAD.**

---

## 5. Cumulative Confidence / Entropy Budget Decay

### Concept
Instead of evaluating each token's margin independently, maintain a running confidence budget:
`budget = budget * P(token_i)` or allow a low-margin token if earlier tokens had exceptionally high confidence (e.g. margin 3.5).

### Empirical Results
- **Throughput:** 64.73 tok/s (-7.87% catastrophic slowdown).

### Why It Failed
1. **Markov Violation:**
   High confidence on `token_0` does not transfer probability mass to `token_1`. If `token_1` is uncertain, guessing it because `token_0` was obvious guarantees a broken chain.
2. **Compounding Tail Hallucinations:**
   Allowing a marginal token poisons all subsequent draft tokens, leading to full rollback cascades and wasted verification batches.
3. **Softmax Overhead:**
   Computing accurate cumulative probabilities requires computing full softmax over the 248k vocabulary, which destroys MTP draft kernel latency.

**Status: PERMANENTLY DEAD.**

---

## 6. Expanding `kMaxLogits` Beyond 16

### Concept
Increase the candidate logit pool size in the draft kernel from 16 to 32 or 64 to allow broader top-k sorting and precise softmax probability reconstruction.

### Empirical Results
- Draft kernel latency increased by 18-24% due to shared memory reduction and register pressure on RDNA 3.5 compute units.
- Acceptance rate was completely unchanged.

### Why It Failed
1. Top-1 vs top-2 margin only requires the two highest logits.
2. The partial top-16 reduction kernel already guarantees finding the top-2 logits with 100% precision. Computing more candidates wastes GPU registers and memory bandwidth.

**Status: PERMANENTLY DEAD.**

---

## 7. Speculative Tree Decoding on AMD Unified Memory (UMA)

### Concept
Draft a branching tree of tokens (e.g. Medusa / Eagle style) instead of a single linear chain, verifying multiple speculative branches concurrently in a single verify forward pass.

### Why It Failed on Strix Halo
1. **Bandwidth-Bound MoE Verify:**
   A tree with 16-32 candidate tokens requires verifying batch sizes of 16-32. On Strix Halo's 256-bit LPDDR5X (273 GB/s peak), loading MoE expert weights for 32 diverse tokens touches nearly all 128 experts per layer, blowing out memory bandwidth and doubling verification latency.
2. **Attention Mask Complexity:**
   Tree attention requires non-trivial custom 2D attention masks and specialized attention kernels, which degrade throughput on RDNA 3.5 compared to contiguous linear causal attention.
3. Linear MTP drafting with Escalating Margin achieves high acceptance (4-7 tokens) at minimal compute and memory overhead.

**Status: DEAD FOR STRIX HALO ARCHITECTURE.**

---

## 8. Prompt Lookup Decoding (PLD) + Neural MTP Gate Hybridization

### Concept
Combine classical prompt n-gram matching with MTP neural drafting in `feat/hybrid-pld-mtp`:
- After MTP step 0 generates candidate d_0, search backwards in the context sequence for an n-gram match (n = 3, 4, 5).
- If matched, draft continuation tokens k (k = 2, 4, 6) from context memory without running subsequent GPU forward passes of the draft head, saving weight streaming bandwidth.

### Empirical Results (n=3 Sweep across 5 Technical Domains on Strix Halo)
- **Baseline (MTP 7 Escalating, Hybrid OFF):** **70.95 +- 0.80 tok/s** (100% Bit-Exact SHA256 match)
- **Hybrid PLD (n=3, k=2):** 68.37 +- 0.02 tok/s (-3.65% regression, Bit-Identity Failed)
- **Hybrid PLD (n=3, k=4):** 68.31 +- 0.01 tok/s (-3.72% regression, Bit-Identity Failed)
- **Hybrid PLD (n=3, k=6):** 67.92 +- 0.00 tok/s (-4.28% regression, Bit-Identity Failed)
- **Hybrid PLD (n=4, k=4):** 69.73 +- 0.00 tok/s (-1.73% regression, Bit-Identity Failed)
- **Hybrid PLD (n=4, k=6):** 69.48 +- 0.00 tok/s (-2.07% regression, Bit-Identity Failed)
- **Hybrid PLD (n=5, k=6):** 70.07 +- 0.02 tok/s (-1.25% regression, Bit-Identity Failed)

### Why It Failed (Empirical & Architectural Root Causes)
1. **Severe Acceptance Deficit vs Neural Draft Head:**
   The MTP neural head maintains a 74.5% - 76.0% acceptance rate because it was trained directly on the model hidden representations. By contrast, PLD context drafts achieved only 0% - 25% acceptance. Generative code and reasoning are not verbatim prompt repetitions; substituting neural drafts with context drafts actively degrades speculative chain length and causes premature verification rollbacks.
2. **GDN Recurrent State Numerical Divergence (Loss of Bit-Identity):**
   Qwen4Exp features Gated Delta Net (GDN) recurrent linear attention layers. In `half-u` precision (F16 hi+lo pairs), `kernels::delta_rule` carries a micro-truncation error (~9e-6) per step. When PLD changes the rollback boundary from draft j=2 to draft j=1, the verify prefix replay runs through a different kernel path (Auto vs ChunkedWmma). Over 150-600 tokens, this micro-error accumulates in the recurrent state matrix and flips near-tied argmax decisions (e.g. `#include <cassert>` vs `<cstring>`), destroying 100% bit identity.
3. **No DRAM Streaming Bottleneck in Draft Head:**
   The MTP draft head is already quantized to Q4 G64 (357 MB total) and pinned in fast SRAM/cache buffers. Replacing 3-4 draft head forward passes saves negligible DRAM bandwidth while dramatically dropping draft accuracy.

**Status: PERMANENTLY DEAD. DO NOT RESURRECT.**

---

## 9. MALL Non-Temporal Loads for MoE Weights (slc dlc Bypass)

### Concept
Hypothesis: 65.9 GiB of streaming MoE weights during decode/verify evicts hot activations, KV cache, and MTP draft head weights from the 32 MB Infinity Cache (MALL) on AMD Strix Halo (GFX1151).
- Implement `__builtin_nontemporal_load` on 128-bit vector chunks (`uint4`) in `kernels/qsplit.hpp`.
- Replace regular loads with non-temporal loads in `kernels/linear_q4.hip` and `kernels/moe_grouped.hip`.
- Emits AMDGPU assembly `global_load_b128 ... slc dlc` (System Level Coherent / Device Level Coherent), explicitly instructing the hardware memory controller to bypass MALL retention.

### Empirical Results (Baseline Prod vs MALL NT on Port 5400, AMD Strix Halo)
- **Bit-Identity:** **PASSED (100% exact SHA256 match across all 5 engineering domains)**
  - C++20 Network Engine: `b633e694a221` (Exact match)
  - Linux Kernel Driver: `30fd0b975f55` (Exact match)
  - Kubernetes Helm Spec: `63bd0ce37f4a` (Exact match)
  - OpenAPI 3.1 YAML Schema: `4df00227f2f9` (Exact match)
  - LSM-Tree Storage Engine: `516e28319923` (Exact match)
- **Baseline Prod (n=5):** **71.32 +- 0.01 tok/s**
  - C++20: 63.00 tok/s | Linux Kernel: 69.84 tok/s | Helm: 76.06 tok/s | OpenAPI: 73.79 tok/s | LSM: 73.89 tok/s
- **MALL Non-Temporal MoE:** **64.79 +- 0.18 tok/s (-9.16% REGRESSION)**
  - C++20: 58.66 tok/s (-6.89%)
  - Linux Kernel: 63.22 tok/s (-9.48%)
  - Kubernetes Helm Spec: 68.33 tok/s (-10.16%)
  - OpenAPI 3.1 YAML Schema: 66.69 tok/s (-9.62%)
  - LSM-Tree Storage Engine: 67.04 tok/s (-9.27%)

### Why It Failed (Empirical & Architectural Root Causes)
1. **MALL (32 MB Infinity Cache) Serves as a Critical Coalescing Buffer on Strix Halo:**
   On AMD Strix Halo (RDNA 3.5 GFX1151, unified LPDDR5X-8533 memory bus), multiple CUs and SIMD waves across dual compute clusters execute expert matrix multiplication simultaneously. When computing SwiGLU gate/up and down projections, different threads and waves hit adjacent 128-bit lines of the same expert weights. The 32 MB MALL provides >1 TB/s internal bandwidth to coalesce and serve these requests.
2. **`slc` (System Level Coherent) Forces Unbuffered DRAM Round-Trips:**
   By marking weight loads non-temporal (`slc`), the GPU bypasses the MALL entirely and sends every 128-bit load directly to the LPDDR5X memory controller queues. This serialized memory transactions across the 256-bit bus, increasing effective read latency and bottlenecking the compute units.
3. **MALL Thrashing Hypothesis Was Incorrect:**
   Hot activations and KV cache fit comfortably in L1 and L2 vector caches during decode. The MALL's high fill rate actively accelerates weight streaming rather than hindering it.

**Status: PERMANENTLY DEAD. DO NOT RESURRECT.**

---

## 10. `v_dot2_f32_bf16` Vectorized MoE Dot Products (Breaks Bit-Identity via FP Non-Associativity)

### Concept
Hypothesis: Vectorizing MoE expert dot product accumulation using AMD RDNA 3.5 dual BF16 dot product instructions (`v_dot2_f32_bf16` / `__builtin_amdgcn_fdot2`) will reduce ALU instruction count in `linear_q4.hip` during SwiGLU projection.

### Empirical Results
- **Throughput:** +1.46% to +5.12% speedup depending on domain.
- **Bit-Identity:** **FAILED (4 out of 5 domains suffered hash mismatches)**
  - C++20 Network Engine: FAILED (hash `e288ebf8a614` != `b633e694a221`)
  - Linux Kernel Driver: PASS (hash `30fd0b975f55` == `30fd0b975f55`)
  - Kubernetes Helm Spec: FAILED (hash `b8d5a1c0d4bb` != `63bd0ce37f4a`)
  - OpenAPI 3.1 YAML Schema: FAILED (hash `5130b429d892` != `4df00227f2f9`)
  - LSM-Tree Storage Engine: FAILED (hash `479c7ee6e544` != `516e28319923`)

### Why It Failed (Empirical & Architectural Root Causes)
1. **IEEE 754 Non-Associativity:**
   - Scalar FMA sequentially evaluates: `acc = ((x0 * w0 + acc) + x1 * w1)`
   - Vectorized `v_dot2_f32_bf16` evaluates: `acc = (x0 * w0 + x1 * w1) + acc`
   - The intermediate pairwise sum differs from sequential accumulation by at most 1 ULP in floating-point round-off.
2. **Amplification in Deep Recurrent States (GDN `half-u`):**
   In deep 48-layer transformer generation, 1-ULP differences in MoE projections propagate into recurrent states and hidden activations. Over dozens to hundreds of autoregressive steps, tiny logit deltas flip near-boundary argmax decisions (e.g. `::std::` vs `std::`), breaking strict bit-identity.
3. **Strict Bit-Identity Requirement:**
   Any kernel optimization that changes the order of floating-point operations in the decode path is strictly forbidden when bit-identity against reference weights is required.

**Status: PERMANENTLY DEAD. DO NOT RESURRECT.**

---

## 11. DRAM-Address-Sorted MoE Expert Iteration (No-Op / Hardware Memory Controller Redundancy)

### Concept
Hypothesis: When multiple tokens route to different experts, iterating over expert weights in monotonic ascending order of their physical DRAM addresses (`expert_id * expert_stride`) will maximize LPDDR5X-8533 burst efficiency, minimize DRAM bank conflict penalties, and improve SwiGLU gather throughput.

### Empirical Results
- **Baseline Prod (n=5, 25 runs x 600 tokens = 15,000 tokens):** **71.26 +- 0.13 tok/s**
- **DRAM-Sorted Gather (n=5, 25 runs x 600 tokens = 15,000 tokens):** **71.35 +- 0.13 tok/s (+0.12% net speedup, within 1-sigma noise)**
- **Bit-Identity:** **100% PASS across all 5 domains** (C++20: `b633e694a221`, Linux: `30fd0b975f55`, Helm: `63bd0ce37f4a`, OpenAPI: `4df00227f2f9`, LSM: `516e28319923`).

### Why It Failed (Empirical & Architectural Root Causes)
1. **Hardware Memory Controller Dynamic Reordering:**
   The AMD RDNA 3.5 (GFX1151) unified memory controller (UMC) already features deep command and transaction queues that dynamically reorder DRAM read requests to maximize bank page hits and burst bandwidth across the 256-bit LPDDR5X-8533 channels.
2. **Software Sorting Overhead:**
   Sorting expert indices in LDS introduces thread divergence and synchronizations (`sort_expert_slots_kernel` consumed 8.69 ms across 3,063 calls in the rocprofv3 trace), canceling out any microscopic physical memory page-hit advantages.
3. **32 MB Infinity Cache (MALL) Masking:**
   Because the MALL acts as a low-latency coalescing layer (>1 TB/s internal bandwidth), adjacent wave memory accesses are coalesced prior to hitting the external LPDDR5X PHYs regardless of dispatch order.

**Status: PERMANENTLY DEAD. DO NOT RESURRECT.**

---

## Summary Comparison Table

| Technique | Goal | Result | Verdict |
| :--- | :--- | :--- | :--- |
| **Escalating Margin (slope=0.08)** | Prune unconfident tails as depth grows | **+2.07% to +4.87% speedup, 0.02 std** | **ACCEPTED (PR #1)** |
| **Parallel 40-CU MTP Pick + Busy-Poll** | 2-stage multi-block top-2 reduction + pause polling | **+0.81% speedup, 100% bit-identical (all 5 domains)** | **ACCEPTED (PR #2)** |
| **AIMD Dynamic Depth** | Dynamically adjust draft depth (4..12) | +0.35% (noise), degrades with escalating margin | **DEAD** |
| **Inter-Round Adaptive Margin** | Shift margin threshold based on past rounds | +0.01% (hysteresis lag) | **DEAD** |
| **Phase-Aware (<think>)** | Throttle depth in thinking blocks | +0.15% (throttles valid streaks) | **DEAD** |
| **Optimistic Probing** | Force deep drafts during low-accept streaks | -0.28% (wasted rollbacks) | **DEAD** |
| **Cumulative Confidence** | Pool confidence across tokens | -7.87% (cascading rollbacks) | **DEAD** |
| **Expand kMaxLogits > 16** | Wider logit pool for sorting | Kernel slowdown, 0% accept gain | **DEAD** |
| **Speculative Tree Decoding** | Branching speculative candidate trees | Saturates LPDDR5X bandwidth | **DEAD** |
| **Hybrid PLD + Neural MTP Gate** | Replace MTP drafts with prompt context continuations | -1.25% to -4.28% slowdown, breaks bit identity via GDN replay | **DEAD** |
| **MALL Non-Temporal MoE Loads** | Bypass 32 MB Infinity Cache (`slc dlc`) to prevent weight thrashing | **-9.16% slowdown** across all domains (100% bit-identical) | **DEAD** |
| **v_dot2_f32_bf16 MoE Vectorization** | Pairwise dot product in SwiGLU | +1.46% to +5.12%, **breaks bit-identity on 4/5 domains** | **DEAD** |
| **DRAM-Sorted MoE Gather** | Sort expert slots in monotonic DRAM order | +0.12% net speedup (no-op, hardware UMC already reorders) | **DEAD** |
| **Attention Softmax Splitting in Prefill** | Multi-split KV keys per query | 15.54% to 36.45% numerical diffs, +4.1% to +18.9% slowdown | **DEAD** |
| **MoE Combine Grouped 2-Deep Pipelining** | 2-deep software pipelining in wide combine | 100% bit-exact, but VGPRs 219->247 drops occupancy 6->5, +18.5% slowdown | **DEAD** |
| **MoE Side-Stream Shared Expert Overlap with Combine** | Overlap shared expert with combine on side stream | 100% bit-exact, but memory bus contention causes +0.43 ms slowdown | **DEAD** |

