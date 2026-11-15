#!/usr/bin/env python3
import sys
import time
import json
import urllib.request
import urllib.parse
import hashlib
import statistics

PORT = 5400
if len(sys.argv) > 2:
    PORT = int(sys.argv[2])
elif len(sys.argv) == 2 and sys.argv[1].isdigit():
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

def run_prompt(task):
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
    req = urllib.request.Request(f"{BASE_URL}/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    
    usage = res.get("usage", {})
    strix = res.get("strix", {})
    gen_tokens = usage.get("completion_tokens", 0)
    tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
    content = res["choices"][0]["message"]["content"]
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    
    return {
        "domain": task["domain"],
        "gen_tokens": gen_tokens,
        "elapsed_s": elapsed,
        "tok_per_sec": tok_per_sec,
        "hash": content_hash,
        "mtp_drafted": strix.get("mtp_drafted_tokens", 0),
        "mtp_accepted": strix.get("mtp_accepted_tokens", 0),
        "mtp_acc_rate": strix.get("mtp_acceptance_rate", 0.0)
    }

def run_benchmark_suite(label, n_runs=5):
    print("\n" + "=" * 80)
    print(f"BENCHMARKING: {label} (n={n_runs} runs)")
    print("=" * 80)

    # Warmup
    print("  Running warmup request...")
    warmup_task = {
        "domain": "Warmup",
        "system": "You are a helpful assistant.",
        "user": "Write a 50-token Fibonacci sequence in C++.",
        "max_tokens": 50
    }
    w = run_prompt(warmup_task)
    print(f"  Warmup complete: {w['tok_per_sec']:.2f} tok/s")

    suite_results = {
        "label": label,
        "runs": [],
        "task_runs": {task["domain"]: [] for task in PROMPTS}
    }

    for run_idx in range(1, n_runs + 1):
        print(f"\n--- Run {run_idx}/{n_runs} for {label} ---")
        run_tps_list = []
        run_time_list = []
        run_tokens_list = []
        
        for task in PROMPTS:
            dom = task["domain"]
            res = run_prompt(task)
            run_tps_list.append(res["tok_per_sec"])
            run_time_list.append(res["elapsed_s"])
            run_tokens_list.append(res["gen_tokens"])
            suite_results["task_runs"][dom].append(res)
            
            mtp_str = f"MTP: {res['mtp_accepted']}/{res['mtp_drafted']} ({res['mtp_acc_rate']:.1f}%)" if res['mtp_drafted'] > 0 else ""
            print(f"  [{dom:<24}] {res['tok_per_sec']:6.2f} tok/s | {res['gen_tokens']} tok in {res['elapsed_s']:.2f}s | {mtp_str} | hash={res['hash']}")

        run_avg_tps = statistics.mean(run_tps_list)
        run_total_tokens = sum(run_tokens_list)
        run_total_time = sum(run_time_list)
        print(f">> Run {run_idx} Avg: {run_avg_tps:6.2f} tok/s (total {run_total_tokens} tok in {run_total_time:.2f}s)")
        suite_results["runs"].append({
            "run_idx": run_idx,
            "avg_tps": run_avg_tps,
            "total_tokens": run_total_tokens,
            "total_time": run_total_time
        })

    return suite_results

def main():
    if len(sys.argv) < 2:
        print("Usage: bench_n5_mall.py <suite_label> [port]")
        sys.exit(1)
    
    label = sys.argv[1]
    res = run_benchmark_suite(label, n_runs=5)
    
    out_file = f"/tmp/bench_{label.replace(' ', '_').lower()}.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved suite results to {out_file}")

if __name__ == "__main__":
    main()
