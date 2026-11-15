#!/usr/bin/env python3
import sys
import time
import json
import urllib.request
import urllib.parse
import hashlib
import statistics

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

CONFIGS = [
    {
        "name": "Baseline (MTP 7 Escalating, Hybrid OFF)",
        "params": {
            "draft": 7,
            "margin": 1.5,
            "escalating_margin": 1,
            "margin_slope": 0.08,
            "hybrid_draft": 0,
            "pld_n": 3,
            "pld_k": 4
        }
    },
    {
        "name": "Hybrid PLD (n=3, k=2)",
        "params": {
            "draft": 7,
            "margin": 1.5,
            "escalating_margin": 1,
            "margin_slope": 0.08,
            "hybrid_draft": 1,
            "pld_n": 3,
            "pld_k": 2
        }
    },
    {
        "name": "Hybrid PLD (n=3, k=4)",
        "params": {
            "draft": 7,
            "margin": 1.5,
            "escalating_margin": 1,
            "margin_slope": 0.08,
            "hybrid_draft": 1,
            "pld_n": 3,
            "pld_k": 4
        }
    },
    {
        "name": "Hybrid PLD (n=3, k=6)",
        "params": {
            "draft": 7,
            "margin": 1.5,
            "escalating_margin": 1,
            "margin_slope": 0.08,
            "hybrid_draft": 1,
            "pld_n": 3,
            "pld_k": 6
        }
    },
    {
        "name": "Hybrid PLD (n=4, k=4)",
        "params": {
            "draft": 7,
            "margin": 1.5,
            "escalating_margin": 1,
            "margin_slope": 0.08,
            "hybrid_draft": 1,
            "pld_n": 4,
            "pld_k": 4
        }
    },
    {
        "name": "Hybrid PLD (n=4, k=6)",
        "params": {
            "draft": 7,
            "margin": 1.5,
            "escalating_margin": 1,
            "margin_slope": 0.08,
            "hybrid_draft": 1,
            "pld_n": 4,
            "pld_k": 6
        }
    },
    {
        "name": "Hybrid PLD (n=5, k=6)",
        "params": {
            "draft": 7,
            "margin": 1.5,
            "escalating_margin": 1,
            "margin_slope": 0.08,
            "hybrid_draft": 1,
            "pld_n": 5,
            "pld_k": 6
        }
    }
]

def set_server_params(params):
    qs = urllib.parse.urlencode(params)
    url = f"{BASE_URL}/set_mtp?{qs}"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"  [Server Config] Updated: {res}")
        return res

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
        "mtp_drafted": usage.get("mtp_drafted_tokens", 0),
        "mtp_accepted": usage.get("mtp_accepted_tokens", 0),
        "mtp_acc_rate": usage.get("mtp_acceptance_rate", 0.0),
        "pld_rounds": usage.get("pld_rounds", 0),
        "pld_drafted": usage.get("pld_drafted_tokens", 0),
        "pld_accepted": usage.get("pld_accepted_tokens", 0),
        "pld_acc_rate": usage.get("pld_acceptance_rate", 0.0)
    }

def main():
    print("=" * 80)
    print("STRIXITE HYBRID PLD + MTP PARAMETER OPTIMIZATION SWEEP")
    print(f"Target: {BASE_URL}")
    print("=" * 80)

    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as resp:
            h = json.loads(resp.read().decode("utf-8"))
            print(f"Server Health OK: {h.get('model')}, slots: {h.get('slots')}")
    except Exception as e:
        print(f"Error connecting to server at {BASE_URL}: {e}")
        sys.exit(1)

    all_results = {}
    N_RUNS = 3
    baseline_hashes = {}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        print("\n" + "=" * 80)
        print(f"CONFIG: {cfg_name}")
        print("=" * 80)
        set_server_params(cfg["params"])
        
        # Warmup pass
        warmup_task = {
            "domain": "Warmup",
            "system": "You are a helpful assistant.",
            "user": "Write a 50-token Fibonacci sequence in C++.",
            "max_tokens": 50
        }
        w = run_prompt(warmup_task)
        print(f"  Warmup complete: {w['tok_per_sec']:.2f} tok/s")

        cfg_results = {
            "params": cfg["params"],
            "runs": [],
            "task_runs": {task["domain"]: [] for task in PROMPTS}
        }

        for run_idx in range(1, N_RUNS + 1):
            print(f"\n--- Run {run_idx}/{N_RUNS} for {cfg_name} ---")
            run_tps_list = []
            run_time_list = []
            run_tokens_list = []
            
            for task in PROMPTS:
                dom = task["domain"]
                res = run_prompt(task)
                run_tps_list.append(res["tok_per_sec"])
                run_time_list.append(res["elapsed_s"])
                run_tokens_list.append(res["gen_tokens"])
                cfg_results["task_runs"][dom].append(res)
                
                # Check bit-identity against baseline
                if cfg_name.startswith("Baseline"):
                    baseline_hashes[dom] = res["hash"]
                    bit_id_str = "BASE"
                else:
                    base_hash = baseline_hashes.get(dom, "")
                    bit_id_str = "MATCH (100% Bit-Exact)" if res["hash"] == base_hash else f"MISMATCH! (got {res['hash']}, exp {base_hash})"

                pld_info = f"PLD: {res['pld_accepted']}/{res['pld_drafted']} ({res['pld_acc_rate']:.1f}%)" if res['pld_drafted'] > 0 else "PLD: 0"
                print(f"  [{dom:<24}] {res['tok_per_sec']:6.2f} tok/s | {res['gen_tokens']} tok in {res['elapsed_s']:.2f}s | {pld_info} | {bit_id_str}")

            run_avg_tps = statistics.mean(run_tps_list)
            run_total_tokens = sum(run_tokens_list)
            run_total_time = sum(run_time_list)
            print(f">> Run {run_idx} Avg: {run_avg_tps:6.2f} tok/s (total {run_total_tokens} tok in {run_total_time:.2f}s)")
            cfg_results["runs"].append({
                "run_idx": run_idx,
                "avg_tps": run_avg_tps,
                "total_tokens": run_total_tokens,
                "total_time": run_total_time
            })

        all_results[cfg_name] = cfg_results

    # Summary table
    print("\n" + "=" * 90)
    print("HYBRID PLD PARAMETER SWEEP SUMMARY (n=3 runs per configuration)")
    print("=" * 90)
    base_name = CONFIGS[0]["name"]
    base_runs = [r["avg_tps"] for r in all_results[base_name]["runs"]]
    base_mean = statistics.mean(base_runs)
    base_std = statistics.stdev(base_runs)

    print(f"{'Configuration':<42} | {'Mean ± Std':<16} | {'Delta %':<8} | {'Bit Identity':<15}")
    print("-" * 90)

    summary_json = {
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "configs": {}
    }

    for cfg_name, data in all_results.items():
        tps_list = [r["avg_tps"] for r in data["runs"]]
        mean_tps = statistics.mean(tps_list)
        std_tps = statistics.stdev(tps_list)
        delta_pct = ((mean_tps / base_mean) - 1.0) * 100.0 if base_mean > 0 else 0.0
        
        # Verify hashes across all prompts
        all_match = True
        for task in PROMPTS:
            dom = task["domain"]
            for r in data["task_runs"][dom]:
                if r["hash"] != baseline_hashes.get(dom, ""):
                    all_match = False
                    break
        bit_status = "100% IDENTICAL" if all_match else "FAILED!"

        stat_str = f"{mean_tps:6.2f} ± {std_tps:4.2f}"
        print(f"{cfg_name:<42} | {stat_str:<16} | {delta_pct:>+6.2f}% | {bit_status}")
        
        # Aggregate PLD stats
        total_pld_drafted = sum(r["pld_drafted"] for dom_runs in data["task_runs"].values() for r in dom_runs)
        total_pld_accepted = sum(r["pld_accepted"] for dom_runs in data["task_runs"].values() for r in dom_runs)
        pld_acc_rate = (100.0 * total_pld_accepted / total_pld_drafted) if total_pld_drafted > 0 else 0.0

        summary_json["configs"][cfg_name] = {
            "runs": tps_list,
            "mean": mean_tps,
            "std": std_tps,
            "delta_pct": delta_pct,
            "bit_identity_ok": all_match,
            "pld_drafted": total_pld_drafted,
            "pld_accepted": total_pld_accepted,
            "pld_acc_rate": pld_acc_rate
        }

    print("-" * 90)

    out_file = "/home/cheblan/bench_sweep_hybrid_pld_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved full results to {out_file}")

if __name__ == "__main__":
    main()
