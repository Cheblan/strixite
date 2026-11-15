#!/usr/bin/env python3
"""
Full Automated Experiment Runner: Baseline Prod vs MALL Non-Temporal MoE.
1. Safely stops production server (port 5300) to isolate DRAM bandwidth.
2. Runs Baseline Prod (5 runs x 5 domains = 25 requests).
3. Runs MALL Non-Temporal MoE (5 runs x 5 domains = 25 requests).
4. Restores production server on port 5300 and verifies health.
5. Verifies 100% Bit-Identity and calculates throughput metrics.
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import statistics

BASE_DIR = "/home/cheblan"
PROD_SCRIPT = f"{BASE_DIR}/run_thinkingcap.sh"
EXP_CONF = f"{BASE_DIR}/strix-exp.conf"
BASELINE_BIN = f"{BASE_DIR}/strixite-main/build/release-znver5/strix_server"
EXP_BIN = f"{BASE_DIR}/strixite-exp/build/release-znver5/strix_server"
BENCH_SCRIPT = f"{BASE_DIR}/strixite-exp/tools/bench_n5_mall.py"

def run_cmd(cmd, shell=True, check=True):
    print(f"[CMD] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=shell)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed ({res.returncode})", flush=True)
        raise RuntimeError(f"Command failed: {cmd}")
    return res

def wait_for_health(port, timeout_s=120):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    print(f"Waiting for {url} to become healthy (timeout: {timeout_s}s)...")
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        print(f"Server on port {port} is healthy!")
                        return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for port {port} to be healthy")

def stop_exp_server():
    print("Stopping any experimental server running on port 5400...")
    subprocess.run("pkill -15 -f strix-exp.conf || true", shell=True)
    time.sleep(3)
    subprocess.run("pkill -9 -f strix-exp.conf || true", shell=True)
    time.sleep(2)

def main():
    try:
        # Step 1: Stop production server
        print("\n>>> STEP 1: Stopping Production Server on 5300 to eliminate memory/cache contention...")
        run_cmd(f"{PROD_SCRIPT} stop")
        time.sleep(3)

        # Ensure no leftover servers
        stop_exp_server()

        # Step 2: Baseline Benchmark
        print("\n>>> STEP 2: Launching Baseline Prod Binary on Port 5400...")
        log_file = "/tmp/strix_bench_baseline.log"
        run_cmd(f"nohup {BASELINE_BIN} --config {EXP_CONF} </dev/null >{log_file} 2>&1 &")
        wait_for_health(5400, timeout_s=150)

        print("\n>>> Running Baseline Suite (5 runs x 5 domains)...", flush=True)
        run_cmd(f"python3 -u {BENCH_SCRIPT} 'Baseline Prod' 5400")

        print("\n>>> Stopping Baseline Server...", flush=True)
        stop_exp_server()

        # Step 3: Experimental Benchmark (MALL Non-Temporal MoE)
        print("\n>>> STEP 3: Launching MALL Non-Temporal MoE Binary on Port 5400...", flush=True)
        log_file_exp = "/tmp/strix_bench_exp.log"
        run_cmd(f"nohup {EXP_BIN} --config {EXP_CONF} </dev/null >{log_file_exp} 2>&1 &")
        wait_for_health(5400, timeout_s=150)

        print("\n>>> Running MALL Non-Temporal MoE Suite (5 runs x 5 domains)...", flush=True)
        run_cmd(f"python3 -u {BENCH_SCRIPT} 'MALL Non-Temporal MoE' 5400")

        print("\n>>> Stopping Experimental Server...")
        stop_exp_server()

    finally:
        # Step 4: Always restore production server!
        print("\n>>> STEP 4: Restoring Production Server on Port 5300...")
        stop_exp_server()
        run_cmd(f"{PROD_SCRIPT} start")
        try:
            wait_for_health(5300, timeout_s=150)
            print("Production server successfully restored and verified healthy.")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to restore production server: {e}")

    # Step 5: Process and Compare Results
    print("\n" + "=" * 80)
    print("ANALYSIS OF BENCHMARK RESULTS")
    print("=" * 80)

    with open("/tmp/bench_baseline_prod.json") as f:
        base_data = json.load(f)
    with open("/tmp/bench_mall_non-temporal_moe.json") as f:
        exp_data = json.load(f)

    # 1. Bit-Identity Verification
    print("\n--- 1. Bit-Identity Verification ---")
    all_bit_identical = True
    domains = list(base_data["task_runs"].keys())
    for dom in domains:
        base_hashes = [r["hash"] for r in base_data["task_runs"][dom]]
        exp_hashes = [r["hash"] for r in exp_data["task_runs"][dom]]
        
        # Verify deterministic within baseline
        base_match = len(set(base_hashes)) == 1
        # Verify deterministic within exp
        exp_match = len(set(exp_hashes)) == 1
        # Verify cross-match
        cross_match = (base_hashes[0] == exp_hashes[0])
        
        status = "PASS (100% BIT-IDENTICAL)" if (base_match and exp_match and cross_match) else "FAIL (DRIFT DETECTED)"
        if not (base_match and exp_match and cross_match):
            all_bit_identical = False
        print(f"  [{dom:<24}] Base: {base_hashes[0]} | Exp: {exp_hashes[0]} -> {status}")

    print(f"\nOverall Bit-Identity: {'PASSED (100% Exact Match)' if all_bit_identical else 'FAILED'}")

    # 2. Performance Comparison by Domain
    print("\n--- 2. Performance Breakdown by Domain (Mean tok/s over n=5) ---")
    print(f"{'Domain':<26} | {'Baseline':>10} | {'MALL NT':>10} | {'Speedup':>9}")
    print("-" * 65)

    base_domain_means = {}
    exp_domain_means = {}
    for dom in domains:
        b_tps = [r["tok_per_sec"] for r in base_data["task_runs"][dom]]
        e_tps = [r["tok_per_sec"] for r in exp_data["task_runs"][dom]]
        b_mean = statistics.mean(b_tps)
        e_mean = statistics.mean(e_tps)
        base_domain_means[dom] = b_mean
        exp_domain_means[dom] = e_mean
        speedup = ((e_mean - b_mean) / b_mean) * 100
        print(f"{dom:<26} | {b_mean:8.2f} t/s | {e_mean:8.2f} t/s | {speedup:+7.2f}%")

    # 3. Overall Suite Metrics
    base_run_tps = [r["avg_tps"] for r in base_data["runs"]]
    exp_run_tps = [r["avg_tps"] for r in exp_data["runs"]]

    b_mean_all = statistics.mean(base_run_tps)
    b_std_all = statistics.stdev(base_run_tps) if len(base_run_tps) > 1 else 0.0
    e_mean_all = statistics.mean(exp_run_tps)
    e_std_all = statistics.stdev(exp_run_tps) if len(exp_run_tps) > 1 else 0.0
    overall_speedup = ((e_mean_all - b_mean_all) / b_mean_all) * 100

    print("-" * 65)
    print(f"{'OVERALL SUITE (n=5)':<26} | {b_mean_all:8.2f} +/- {b_std_all:4.2f} | {e_mean_all:8.2f} +/- {e_std_all:4.2f} | {overall_speedup:+7.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
