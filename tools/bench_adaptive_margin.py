#!/usr/bin/env python3
"""
bench_adaptive_margin.py:
Scientific A/B evaluation of Speculative Decoding enhancements on AMD Strix Halo:
  1. Baseline: Static MTP draft = 7 (margin = 1.5)
  2. Pure AIMD: Dynamic MTP depth 4..12 (margin = 1.5)
  3. Option 1: Dynamic MTP depth 4..12 + Adaptive Margin (dynamic margin 1.0..2.5)
  4. Option 2: Dynamic MTP depth 4..12 + Escalating Margin per step (slope = 0.08)
  5. Option 2b: Dynamic MTP depth 4..12 + Escalating Margin (base = 1.3, slope = 0.08)
  6. Option 1+2: Dynamic MTP depth 4..12 + Adaptive Margin + Escalating Margin
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
        "domain": "C++20 Network Engine",
        "system": "You are a principal systems architect specializing in ultra-low latency C++20.",
        "user": "Write a production-grade C++20 coroutine TCP echo engine using io_uring with ring buffers and state machine.",
        "max_tokens": 600
    },
    {
        "domain": "Linux Kernel Driver",
        "system": "You are a Linux kernel maintainer specializing in PCIe device drivers.",
        "user": "Implement a Linux kernel module for a PCIe network accelerator with MSI-X, DMA rings, and NAPI poll.",
        "max_tokens": 600
    },
    {
        "domain": "Kubernetes Helm Spec",
        "system": "You are a lead cloud-native infrastructure engineer.",
        "user": "Generate a complete Helm chart definition for a distributed banking processor with StatefulSet and NetworkPolicy.",
        "max_tokens": 600
    },
    {
        "domain": "OpenAPI 3.1 YAML Schema",
        "system": "You are a fintech API architect.",
        "user": "Write an exhaustive OpenAPI 3.1 YAML specification for an FX settlement service with JSON Schema schemas.",
        "max_tokens": 600
    },
    {
        "domain": "LSM-Tree Storage Engine",
        "system": "You are a database engine developer.",
        "user": "Design and implement in modern C++ an in-memory LSM tree with Leveled Compaction, Bloom filter, and SkipList.",
        "max_tokens": 600
    }
]

def set_mtp_config(dynamic: bool, draft: int, draft_min: int = 4, draft_max: int = 12,
                   margin: float = 1.5, adaptive_margin: bool = False,
                   escalating_margin: bool = False, margin_slope: float = 0.08):
    params = {
        "dynamic": "1" if dynamic else "0",
        "draft": str(draft),
        "draft_min": str(draft_min),
        "draft_max": str(draft_max),
        "margin": f"{margin:.2f}",
        "adaptive_margin": "1" if adaptive_margin else "0",
        "escalating_margin": "1" if escalating_margin else "0",
        "margin_slope": f"{margin_slope:.3f}",
    }
    url = f"{BASE_URL}/set_mtp?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

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
    with urllib.request.urlopen(req, timeout=120) as resp:
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

def eval_experiment(label, **kwargs):
    cfg = set_mtp_config(**kwargs)
    tok_sum = 0
    time_sum = 0
    runs = []
    print(f"\n{'='*20} {label} {'='*20}")
    print(f"Config: {cfg}")
    for idx, task in enumerate(PROMPTS):
        print(f"  [{idx+1}/5] {task['domain']:<26}: ", end="", flush=True)
        res = run_query(task)
        print(f"{res['tokens']} tok in {res['time_s']:.2f}s -> {res['tok_per_sec']:5.2f} tok/s")
        tok_sum += res["tokens"]
        time_sum += res["time_s"]
        runs.append(res)
    avg_speed = tok_sum / time_sum if time_sum > 0 else 0
    print(f"  => Summary Average Speed: {avg_speed:5.2f} tok/s")
    return {
        "label": label,
        "avg_speed": avg_speed,
        "tok_sum": tok_sum,
        "time_sum": time_sum,
        "runs": runs,
        "config": cfg
    }

def main():
    print("=" * 80)
    print("ADAPTIVE SPECULATIVE MTP MARGIN BENCHMARK SUITE")
    print(f"Server Target: {BASE_URL}")
    print("=" * 80)

    # 1. Warmup
    print("Warming up engine...")
    set_mtp_config(dynamic=False, draft=7, margin=1.5)
    run_query({"system": "Assistant", "user": "Hello!", "max_tokens": 32})
    print("Warmup complete.")

    experiments = []

    # 2. Baseline (Static 7)
    base_res = eval_experiment(
        "1. Baseline (Static 7)",
        dynamic=False, draft=7, margin=1.5, adaptive_margin=False, escalating_margin=False
    )
    base_avg = base_res["avg_speed"]
    experiments.append(base_res)

    # 3. Pure AIMD (draft 4..12, margin 1.5)
    aimd_res = eval_experiment(
        "2. Pure AIMD (4..12, static margin 1.5)",
        dynamic=True, draft=7, draft_min=4, draft_max=12, margin=1.5,
        adaptive_margin=False, escalating_margin=False
    )
    experiments.append(aimd_res)

    # 4. Option 1: Dynamic MTP depth 4..12 + Adaptive Margin (margin 1.5 base)
    opt1_res = eval_experiment(
        "3. Option 1: Adaptive Margin (depth 4..12, margin 1.5 base)",
        dynamic=True, draft=7, draft_min=4, draft_max=12, margin=1.5,
        adaptive_margin=True, escalating_margin=False
    )
    experiments.append(opt1_res)

    # 5. Option 2: Escalating Margin (margin 1.5 base, slope 0.08)
    opt2_res = eval_experiment(
        "4. Option 2: Escalating Margin (base 1.5, slope 0.08)",
        dynamic=True, draft=7, draft_min=4, draft_max=12, margin=1.5,
        adaptive_margin=False, escalating_margin=True, margin_slope=0.08
    )
    experiments.append(opt2_res)

    # 6. Option 2b: Escalating Margin (base 1.3, slope 0.08)
    opt2b_res = eval_experiment(
        "5. Option 2b: Escalating Margin (base 1.3, slope 0.08)",
        dynamic=True, draft=7, draft_min=4, draft_max=12, margin=1.3,
        adaptive_margin=False, escalating_margin=True, margin_slope=0.08
    )
    experiments.append(opt2b_res)

    # 7. Option 1 + 2 Combined
    opt12_res = eval_experiment(
        "6. Option 1+2 Combined (Adaptive Margin + Escalating Margin)",
        dynamic=True, draft=7, draft_min=4, draft_max=12, margin=1.5,
        adaptive_margin=True, escalating_margin=True, margin_slope=0.08
    )
    experiments.append(opt12_res)

    # Final Summary Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Experiment':<38} | {'Avg Speed':<11} | {'vs Base (7)':<12} | {'vs AIMD':<10}")
    print("-" * 80)
    aimd_avg = aimd_res["avg_speed"]
    for exp in experiments:
        vs_base = ((exp["avg_speed"] / base_avg) - 1.0) * 100.0
        vs_aimd = ((exp["avg_speed"] / aimd_avg) - 1.0) * 100.0
        print(f"{exp['label']:<38} | {exp['avg_speed']:6.2f} tok/s | {vs_base:>+9.2f}% | {vs_aimd:>+8.2f}%")
    print("=" * 80)

    # Bit-identity check across experiments vs baseline
    print("\nBIT IDENTITY AUDIT (vs Baseline SHA-256):")
    for exp in experiments:
        matches = sum(1 for i in range(len(PROMPTS)) if exp["runs"][i]["sha256"] == base_res["runs"][i]["sha256"])
        print(f"  {exp['label']:<38}: {matches}/{len(PROMPTS)} bit-exact matches")

    with open("bench_adaptive_margin_results.json", "w") as f:
        # Avoid dumping whole content text to keep json compact
        save_data = []
        for exp in experiments:
            save_data.append({
                "label": exp["label"],
                "avg_speed": exp["avg_speed"],
                "speedup_vs_base": ((exp["avg_speed"] / base_avg) - 1.0) * 100.0,
                "speedup_vs_aimd": ((exp["avg_speed"] / aimd_avg) - 1.0) * 100.0,
                "config": exp["config"],
                "tasks": [
                    {
                        "domain": PROMPTS[i]["domain"],
                        "tok_per_sec": exp["runs"][i]["tok_per_sec"],
                        "tokens": exp["runs"][i]["tokens"],
                        "sha256": exp["runs"][i]["sha256"],
                        "match_base": exp["runs"][i]["sha256"] == base_res["runs"][i]["sha256"]
                    }
                    for i in range(len(PROMPTS))
                ]
            })
        json.dump(save_data, f, indent=2)
    print("\nSaved full results to bench_adaptive_margin_results.json")

if __name__ == "__main__":
    main()
