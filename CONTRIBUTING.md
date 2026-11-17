# Contributing to strixite (Autonomous Fork)

Pull requests, bug reports, and optimizations are welcome. 

We develop completely in the open. Unlike upstream, we do not operate private development mirrors, we do not lock issues, and we do not squash community work into anonymous sync commits. If your code works and improves the engine, it lands directly in `main` with full author credit.

Contributions are accepted under the project's license, [AGPL-3.0](LICENSE).

---

## What We Care About

We keep the barrier to entry simple:

1. **100.000% Bit-Identity (T=0):**
   Greedy decoding is completely deterministic. Any kernel change, memory tweak, or scheduling optimization must produce identical tokens compared to the reference baseline at `temperature = 0.0`. If tokens drift, the PR cannot be merged.
2. **Real Benchmark Numbers (n=5):**
   If your PR claims a speedup, provide numbers:
   - Run 1 untimed warmup request to prime GPU caches and page tables.
   - Run 5 timed trials and report the mean +- standard deviation.
   - Synthetic 1-run claims will be re-tested on our hardware.
3. **Multi-Agent / Multi-Slot Concurrency:**
   If you modify `GpuArbiter`, memory allocators, or scheduling, test under concurrent multi-slot workloads, not just an isolated single slot.

---

## Code Quality

We don't care how or with what tools you write your code — by hand, with AI agents, or dictating it to your dog. What matters is the result:
- **Clean and readable:** No copy-pasted spaghetti or unmaintainable hacks.
- **Fast and verifiable:** Real performance gains proven on real hardware.
- **Correct:** Zero math compromises, zero memory leaks, and 100% bit-exact outputs.

---

## How to Submit

1. Fork the repository on GitHub.
2. Branch off `main` (name your branch whatever you like).
3. Open a Pull Request targeting `main`. Describe what was changed and attach your benchmark/test results.
4. We will run it on our AMD Strix Halo testbed (`gfx1151`) and merge it.
