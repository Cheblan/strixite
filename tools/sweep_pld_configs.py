#!/usr/bin/env python3
"""
Comprehensive Config Sweep for MTP + Hybrid Prompt Lookup Decoding (PLD).
Tests permutations of:
  - PLD on/off
  - n-gram match length (n=2, 3, 4)
  - draft depth / max PLD draft tokens (draft=5, 7, 9, 11 / k=4, 6, 8, 10)
  - MTP step-0 confidence margin (margin=1.0, 1.2, 1.5, 1.8)

Checks and enforces Bit Identity (SHA256) across all test runs.
"""

import sys
import os
import json
import time
import urllib.request
import urllib.parse
from bench_heavy_context import BENCH_HEAVY_PROMPTS, run_single

BASE_URL = "http://127.0.0.1:5400"

CONFIG_SWEEP = [
    {
        "id": "baseline_d7_m1.5",
        "name": "1. Baseline (MTP 7, Margin 1.5, PLD OFF)",
        "draft": 7,
        "margin": 1.5,
        "pld": False,
        "pld_n": 3,
        "pld_k": 6
    },
    {
        "id": "pld_n2_k6_d7_m1.5",
        "name": "2. PLD n=2 (2-gram, k=6, draft 7, marg 1.5)",
        "draft": 7,
        "margin": 1.5,
        "pld": True,
        "pld_n": 2,
        "pld_k": 6
    },
    {
        "id": "pld_n3_k6_d7_m1.5",
        "name": "3. PLD n=3 (3-gram, k=6, draft 7, marg 1.5)",
        "draft": 7,
        "margin": 1.5,
        "pld": True,
        "pld_n": 3,
        "pld_k": 6
    },
    {
        "id": "pld_n4_k6_d7_m1.5",
        "name": "4. PLD n=4 (4-gram, k=6, draft 7, marg 1.5)",
        "draft": 7,
        "margin": 1.5,
        "pld": True,
        "pld_n": 4,
        "pld_k": 6
    },
    {
        "id": "pld_n3_k4_d5_m1.5",
        "name": "5. PLD n=3, Short Draft (k=4, draft 5, marg 1.5)",
        "draft": 5,
        "margin": 1.5,
        "pld": True,
        "pld_n": 3,
        "pld_k": 4
    },
    {
        "id": "pld_n3_k8_d9_m1.5",
        "name": "6. PLD n=3, Deep Draft (k=8, draft 9, marg 1.5)",
        "draft": 9,
        "margin": 1.5,
        "pld": True,
        "pld_n": 3,
        "pld_k": 8
    },
    {
        "id": "pld_n2_k8_d9_m1.5",
        "name": "7. PLD n=2, Deep Draft (k=8, draft 9, marg 1.5)",
        "draft": 9,
        "margin": 1.5,
        "pld": True,
        "pld_n": 2,
        "pld_k": 8
    },
    {
        "id": "pld_n3_k10_d11_m1.5",
        "name": "8. PLD n=3, Ultra Draft (k=10, draft 11, marg 1.5)",
        "draft": 11,
        "margin": 1.5,
        "pld": True,
        "pld_n": 3,
        "pld_k": 10
    },
    {
        "id": "pld_n3_k6_d7_m1.2",
        "name": "9. PLD n=3, Margin 1.2 (k=6, draft 7, marg 1.2)",
        "draft": 7,
        "margin": 1.2,
        "pld": True,
        "pld_n": 3,
        "pld_k": 6
    },
    {
        "id": "pld_n3_k6_d7_m1.8",
        "name": "10. PLD n=3, Margin 1.8 (k=6, draft 7, marg 1.8)",
        "draft": 7,
        "margin": 1.8,
        "pld": True,
        "pld_n": 3,
        "pld_k": 6
    }
]

def set_server_config(cfg):
    params = urllib.parse.urlencode({
        "margin": cfg["margin"],
        "draft": cfg["draft"],
        "pld": "1" if cfg["pld"] else "0",
        "pld_n": cfg["pld_n"],
        "pld_k": cfg["pld_k"]
    })
    url = f"{BASE_URL}/set_mtp?{params}"
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def wait_server_ready():
    for _ in range(30):
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def main():
    if not wait_server_ready():
        print(f"Error: Server at {BASE_URL} not reachable.")
        sys.exit(1)

    print("==========================================================================================")
    print("                     MTP + HYBRID PLD CONFIGURATION SWEEP")
    print("==========================================================================================")
    print(f"Total Configurations to test: {len(CONFIG_SWEEP)}")
    print(f"Prompts per config: {len(BENCH_HEAVY_PROMPTS)} (600 tokens greedy)")

    # Baseline hashes for bit identity check
    baseline_hashes = {}
    sweep_results = []

    for idx, cfg in enumerate(CONFIG_SWEEP, 1):
        print(f"\n[{idx:02d}/{len(CONFIG_SWEEP):02d}] Configuring: {cfg['name']}...")
        ack = set_server_config(cfg)
        print(f"Server ack: draft={ack.get('mtp_draft')}, margin={ack.get('mtp_margin')}, pld={ack.get('pld')}, n={ack.get('pld_n')}, k={ack.get('pld_k')}")
        sys.stdout.flush()

        time.sleep(0.5)
        config_runs = []
        for p_idx, p in enumerate(BENCH_HEAVY_PROMPTS, 1):
            res = run_single(BASE_URL, p["prompt"])
            config_runs.append((p, res))
            
            # Bit identity check against baseline
            p_name = p["name"]
            is_match = "N/A"
            if idx == 1:
                baseline_hashes[p_name] = res["sha256"]
                is_match = "BASE"
            else:
                base_h = baseline_hashes.get(p_name)
                is_match = "MATCH" if base_h == res["sha256"] else "MISMATCH!"

            pld_s = f" | PLD: {res['pld_accepted']:2d}/{res['pld_drafted']:2d} ({res['pld_rate']:4.1f}%)" if cfg["pld"] else ""
            print(f"  P{p_idx}: {res['tok_s']:5.2f} tok/s | MTP: {res['mtp_accepted']:3d}/{res['mtp_drafted']:3d} ({res['mtp_rate']:4.1f}%){pld_s} | {is_match} ({res['sha256'][:8]})")
            sys.stdout.flush()

        avg_tok_s = sum(r["tok_s"] for _, r in config_runs) / len(config_runs)
        total_pld_acc = sum(r["pld_accepted"] for _, r in config_runs)
        total_pld_drf = sum(r["pld_drafted"] for _, r in config_runs)
        total_mtp_acc = sum(r["mtp_accepted"] for _, r in config_runs)
        total_mtp_drf = sum(r["mtp_drafted"] for _, r in config_runs)
        all_match = all(baseline_hashes.get(p["name"]) == r["sha256"] for p, r in config_runs)

        res_entry = {
            "config": cfg,
            "avg_tok_s": avg_tok_s,
            "all_match": all_match,
            "total_pld_acc": total_pld_acc,
            "total_pld_drf": total_pld_drf,
            "total_mtp_acc": total_mtp_acc,
            "total_mtp_drf": total_mtp_drf,
            "prompt_runs": [{
                "name": p["name"],
                "tok_s": r["tok_s"],
                "tokens": r["tokens"],
                "time_s": r["time_s"],
                "sha256": r["sha256"],
                "match": baseline_hashes.get(p["name"]) == r["sha256"],
                "pld_accepted": r["pld_accepted"],
                "pld_drafted": r["pld_drafted"],
                "mtp_accepted": r["mtp_accepted"],
                "mtp_drafted": r["mtp_drafted"]
            } for p, r in config_runs]
        }
        sweep_results.append(res_entry)
        print(f"  --> AVERAGE: {avg_tok_s:5.2f} tok/s | Bit Identity: {'OK' if all_match else 'FAIL'}")

    # Final summary table
    print("\n" + "=" * 105)
    print("                                   FINAL CONFIGURATION SWEEP RANKING")
    print("=" * 105)
    print(f"{'Rank':<5} | {'Configuration':<45} | {'Avg tok/s':<10} | {'Delta vs Base':<14} | {'PLD Acc':<10} | {'Bit-Exact'}")
    print("-" * 105)

    base_avg = sweep_results[0]["avg_tok_s"]
    ranked = sorted(sweep_results, key=lambda x: x["avg_tok_s"], reverse=True)

    for rank, item in enumerate(ranked, 1):
        cfg = item["config"]
        speed = item["avg_tok_s"]
        delta = ((speed - base_avg) / base_avg) * 100.0
        sign = "+" if delta >= 0 else ""
        pld_acc_str = f"{item['total_pld_acc']}/{item['total_pld_drf']}" if cfg["pld"] else "OFF"
        match_str = "MATCH (100%)" if item["all_match"] else "MISMATCH"
        print(f"{rank:<5} | {cfg['name']:<45} | {speed:6.2f}     | {sign}{delta:5.2f}%       | {pld_acc_str:<10} | {match_str}")

    print("=" * 105)

    out_path = "/home/cheblan/pld_config_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Sweep results successfully written to {out_path}\n")

if __name__ == "__main__":
    main()
