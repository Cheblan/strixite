#!/usr/bin/env python3
"""
bench_adaptive_mtp.py:
Benchmark Baseline (Static MTP 7) vs Adaptive Dynamic MTP (depth 2..10)
across 5 heavy-context tasks with strict Bit-Identity ($T=0$ SHA-256) validation.
"""

import sys
import time
import json
import hashlib
import urllib.request
import urllib.parse

PORT = 5400
if len(sys.argv) > 1:
    PORT = int(sys.argv[1])

BASE_URL = f"http://127.0.0.1:{PORT}"

PROMPTS = [
    {
        "domain": "C++20 High-Performance Network Engine",
        "system": "You are a principal systems architect specializing in ultra-low latency C++20 and Linux kernel bypass.",
        "user": (
            "Write a production-grade, zero-allocation C++20 coroutine-based TCP echo engine using io_uring. "
            "Implement a ring buffer memory pool, registered buffer descriptors, socket lifecycle state machine, "
            "and multithreaded work-stealing reactor. Include full comments and error handling."
        ),
        "max_tokens": 1024
    },
    {
        "domain": "Linux Kernel Driver & Diagnostics",
        "system": "You are a Linux kernel maintainer specializing in PCIe device drivers and high-speed DMA engines.",
        "user": (
            "Provide a complete Linux kernel module implementation for a mock PCIe network accelerator. "
            "Include probe/remove, MSI-X interrupt vector allocation, coherent DMA descriptor ring buffers, "
            "sk_buff allocation in NAPI poll context, and sysfs diagnostics statistics."
        ),
        "max_tokens": 1024
    },
    {
        "domain": "Kubernetes Microservices & Helm Spec",
        "system": "You are a lead cloud-native infrastructure engineer specializing in hardened Kubernetes deployments.",
        "user": (
            "Generate a complete, enterprise-grade Helm chart definition for a distributed financial transaction processor. "
            "Include StatefulSet with volume claim templates, PodDisruptionBudget, ServiceAccount with IRSA annotations, "
            "NetworkPolicy with strict egress/ingress rules, and PrometheusRule alerting for latency SLO violations."
        ),
        "max_tokens": 1024
    },
    {
        "domain": "OpenAPI 3.1 REST API & Data Models",
        "system": "You are a fintech API architect designing high-throughput payment settlement APIs.",
        "user": (
            "Write an exhaustive OpenAPI 3.1 YAML specification for a multi-currency FX settlement service. "
            "Include endpoints for quote generation, trade execution, settlement batching, and webhook notifications. "
            "Specify rigorous JSON Schema definitions with idempotency keys, regex patterns, and ISO 4217 currencies."
        ),
        "max_tokens": 1024
    },
    {
        "domain": "LSM-Tree Storage Engine",
        "system": "You are a database engine developer specializing in write-optimized key-value storage.",
        "user": (
            "Design and implement in modern C++ an in-memory Log-Structured Merge (LSM) tree with Leveled Compaction. "
            "Include SkipList memtable, immutable memtable flushing, Block-based SSTable format with Bloom filter "
            "and two-level index, and multi-version concurrency control (MVCC) snapshot reads."
        ),
        "max_tokens": 1024
    }
]

def set_mtp_config(dynamic: bool, draft: int, draft_min: int = 4, draft_max: int = 12):
    params = {
        "dynamic": "1" if dynamic else "0",
        "draft": str(draft),
        "draft_min": str(draft_min),
        "draft_max": str(draft_max),
    }
    url = f"{BASE_URL}/set_mtp?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def get_stats():
    req = urllib.request.Request(f"{BASE_URL}/metrics")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode()
            stats = {}
            for line in text.splitlines():
                if line.startswith("strix_mtp_drafted_tokens_total"):
                    stats["drafted"] = float(line.split()[1])
                elif line.startswith("strix_mtp_accepted_tokens_total"):
                    stats["accepted"] = float(line.split()[1])
            return stats
    except Exception:
        return {}

def run_query(task):
    payload = {
        "model": "thinkingcap-qwen3.8-flash-next",
        "messages": [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["user"]}
        ],
        "temperature": 0.0,
        "max_tokens": task["max_tokens"],
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode())
    t1 = time.perf_counter()
    
    choice = res["choices"][0]
    content = choice["message"]["content"]
    usage = res.get("usage", {})
    comp_tokens = usage.get("completion_tokens", len(content.split()))
    gen_time = t1 - t0
    tok_per_sec = comp_tokens / gen_time if gen_time > 0 else 0
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    return {
        "content": content,
        "tokens": comp_tokens,
        "time_s": gen_time,
        "tok_per_sec": tok_per_sec,
        "sha256": sha256
    }

def main():
    print("=" * 80)
    print("BENCHMARK: Static MTP (draft=7) vs EMA-Smoothed Adaptive MTP (draft=4..12)")
    print(f"Server Target: {BASE_URL}")
    print("=" * 80)

    # 1. Warmup
    print("[1/4] Warming up model server...")
    warmup_task = {
        "system": "You are a helpful assistant.",
        "user": "Explain QuickSort in C++ in 50 words.",
        "max_tokens": 64
    }
    set_mtp_config(dynamic=False, draft=7)
    run_query(warmup_task)
    print("  Warmup done.\n")

    # 2. Run Baseline (Static MTP 7)
    print("[2/4] Running Baseline (Static MTP draft = 7)...")
    set_mtp_config(dynamic=False, draft=7)
    baseline_results = []
    for idx, task in enumerate(PROMPTS):
        print(f"  Task {idx+1}/5: {task['domain']}...", end="", flush=True)
        res = run_query(task)
        print(f" {res['tokens']} tok in {res['time_s']:.2f}s ({res['tok_per_sec']:.2f} tok/s)")
        baseline_results.append(res)
    
    # 3. Run Adaptive Dynamic MTP (4..12, EMA smoothing)
    print("\n[3/4] Running Adaptive Dynamic MTP (draft_min=4, draft_max=12, EMA smoothing)...")
    set_mtp_config(dynamic=True, draft=7, draft_min=4, draft_max=12)
    adaptive_results = []
    for idx, task in enumerate(PROMPTS):
        print(f"  Task {idx+1}/5: {task['domain']}...", end="", flush=True)
        res = run_query(task)
        print(f" {res['tokens']} tok in {res['time_s']:.2f}s ({res['tok_per_sec']:.2f} tok/s)")
        adaptive_results.append(res)

    # 4. Results & Bit Identity Comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON & BIT IDENTITY VERIFICATION")
    print("=" * 80)
    print(f"{'Domain':<38} | {'Baseline':<10} | {'Adaptive':<10} | {'Speedup':<8} | {'Bit Identity'}")
    print("-" * 80)

    all_matched = True
    base_tok_sum = 0
    base_time_sum = 0
    adapt_tok_sum = 0
    adapt_time_sum = 0

    for i in range(len(PROMPTS)):
        b = baseline_results[i]
        a = adaptive_results[i]
        matched = (b["sha256"] == a["sha256"])
        if not matched:
            all_matched = False
        
        speedup = ((a["tok_per_sec"] / b["tok_per_sec"]) - 1.0) * 100.0 if b["tok_per_sec"] > 0 else 0
        match_str = "MATCH (100%)" if matched else "MISMATCH"
        
        base_tok_sum += b["tokens"]
        base_time_sum += b["time_s"]
        adapt_tok_sum += a["tokens"]
        adapt_time_sum += a["time_s"]

        domain_name = PROMPTS[i]["domain"][:37]
        print(f"{domain_name:<38} | {b['tok_per_sec']:>7.2f} t/s | {a['tok_per_sec']:>7.2f} t/s | {speedup:>+6.2f}% | {match_str}")
        if not matched:
            print(f"   Baseline SHA256: {b['sha256']}")
            print(f"   Adaptive SHA256: {a['sha256']}")

    base_avg_tps = base_tok_sum / base_time_sum if base_time_sum > 0 else 0
    adapt_avg_tps = adapt_tok_sum / adapt_time_sum if adapt_time_sum > 0 else 0
    overall_speedup = ((adapt_avg_tps / base_avg_tps) - 1.0) * 100.0 if base_avg_tps > 0 else 0

    print("-" * 80)
    print(f"{'OVERALL AVERAGE':<38} | {base_avg_tps:>7.2f} t/s | {adapt_avg_tps:>7.2f} t/s | {overall_speedup:>+6.2f}% | {'ALL MATCH!' if all_matched else 'FAILED'}")
    print("=" * 80)

    summary = {
        "overall_speedup_pct": overall_speedup,
        "base_avg_tps": base_avg_tps,
        "adapt_avg_tps": adapt_avg_tps,
        "all_matched": all_matched,
        "tasks": []
    }
    for i in range(len(PROMPTS)):
        summary["tasks"].append({
            "domain": PROMPTS[i]["domain"],
            "baseline_tps": baseline_results[i]["tok_per_sec"],
            "adaptive_tps": adaptive_results[i]["tok_per_sec"],
            "speedup_pct": ((adaptive_results[i]["tok_per_sec"] / baseline_results[i]["tok_per_sec"]) - 1.0) * 100.0,
            "bit_identity_match": baseline_results[i]["sha256"] == adaptive_results[i]["sha256"]
        })
    with open("adaptive_mtp_benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results to adaptive_mtp_benchmark_results.json")

if __name__ == "__main__":
    main()
