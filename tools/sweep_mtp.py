#!/usr/bin/env python3
import json
import time
import hashlib
import sys
import os
import urllib.request
import urllib.error

BENCH_PROMPTS = [
    {
        "name": "Code (ThreadPool)",
        "prompt": "Write a complete, thread-safe C++20 ThreadPool class with a fixed worker count, std::latch for synchronization, and tasks returning std::future<T>."
    },
    {
        "name": "Russian Context QA",
        "prompt": """Ниже приведена документация планировщика задач ядра Linux:

Планировщик CFS (Completely Fair Scheduler) использовал концепцию идеального многозадачного процессора на виртуальном оборудовании. Каждому потоку назначается виртуальное время выполнения (vruntime), которое отражает, сколько времени поток провел на физическом процессоре с учетом его приоритета (nice level). В ядре 6.6 CFS был заменен на EEVDF (Earliest Eligible Virtual Deadline First), разработанный Питером Зейлстрой на основе теории Стоики и Абдель-Вахаба 1996 года. EEVDF разделяет понятия 'право на запуск' (eligibility, когда vruntime не отстает от среднего) и 'дедлайн' (deadline, виртуальный момент завершения кванта времени).

На основе приведенного текста подробно объясни:
1) В чём заключалась идея CFS и как использовался vruntime?
2) Почему и когда ядро Linux перешло на планировщик EEVDF?
3) В чём ключевая математическая разница между понятиями eligibility и deadline в EEVDF?"""
    },
    {
        "name": "Math & Proof",
        "prompt": "Prove rigorously: determine all prime numbers p such that p^2 + 2 is prime, and prove whether p^3 + 2 can ever be prime for p > 3."
    },
    {
        "name": "Structured Extraction",
        "prompt": """Here is a raw system log snippet:
[2026-11-14 12:00:01] INFO [cluster-node-01] cpu_util=45.2% mem_used_gib=62.4 mem_total_gib=128.0 gpu_busy=88.1% temp_c=64
[2026-11-14 12:00:02] INFO [cluster-node-02] cpu_util=82.1% mem_used_gib=110.2 mem_total_gib=128.0 gpu_busy=94.5% temp_c=78
[2026-11-14 12:00:03] WARN [cluster-node-03] cpu_util=98.7% mem_used_gib=124.8 mem_total_gib=128.0 gpu_busy=99.2% temp_c=86
[2026-11-14 12:00:04] INFO [cluster-node-01] cpu_util=46.0% mem_used_gib=62.5 mem_total_gib=128.0 gpu_busy=87.9% temp_c=65
[2026-11-14 12:00:05] INFO [cluster-node-02] cpu_util=81.8% mem_used_gib=110.1 mem_total_gib=128.0 gpu_busy=94.2% temp_c=77

Parse each line and format as a clean Markdown table with columns: Timestamp, Node, Status, CPU %, RAM Used (GiB), RAM Total (GiB), GPU Busy %, Temp (C). Then calculate the average CPU utilization across all entries."""
    }
]

def set_mtp(base_url, margin, draft):
    url = f"{base_url}/set_mtp?margin={margin}&draft={draft}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def run_single(base_url, prompt, max_tokens=500):
    payload = {
        "model": "thinkingcap-qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp_body = resp.read().decode("utf-8")
    t_total = time.perf_counter() - t0
    
    res = json.loads(resp_body)
    usage = res.get("usage", {})
    strix = res.get("strix", {})
    completion_tokens = usage.get("completion_tokens", 0)
    content = res["choices"][0]["message"]["content"]
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    tok_per_sec = completion_tokens / t_total if t_total > 0 else 0.0
    
    return {
        "tokens": completion_tokens,
        "time_s": t_total,
        "tok_s": tok_per_sec,
        "sha256": content_hash,
        "mtp_drafted": strix.get("mtp_drafted_tokens", 0),
        "mtp_accepted": strix.get("mtp_accepted_tokens", 0),
        "mtp_rate": strix.get("mtp_acceptance_rate", 0.0)
    }

def main():
    base_url = "http://127.0.0.1:5400"
    out_file = "/home/cheblan/mtp_sweep_results.json"
    
    margins = [0.2, 0.5, 0.8, 1.2, 1.5]
    drafts = [3, 5, 7]
    
    # Also baseline (no MTP: margin=99.0, draft=1)
    grid = [(99.0, 1, "Baseline (MTP Off)")]
    for m in margins:
        for d in drafts:
            grid.append((m, d, f"Margin {m} / Draft {d}"))

    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            pass

    print("================================================================================")
    print(f"  MTP GRID SWEEP: {len(margins)} margins x {len(drafts)} drafts + Baseline = {len(grid)} configs")
    print("================================================================================")
    sys.stdout.flush()

    for idx, (m, d, label) in enumerate(grid, 1):
        key = f"m_{m}_d_{d}"
        if key in results:
            print(f"[{idx}/{len(grid)}] SKIPPING {label} (already recorded: {results[key]['avg_tok_s']:.2f} tok/s)")
            sys.stdout.flush()
            continue

        st = set_mtp(base_url, m, d)
        print(f"\n--- [{idx}/{len(grid)}] Config: {label} (set: margin={st.get('mtp_margin')}, draft={st.get('mtp_draft')}) ---")
        sys.stdout.flush()

        prompt_results = []
        for p in BENCH_PROMPTS:
            res = run_single(base_url, p["prompt"])
            prompt_results.append({
                "name": p["name"],
                **res
            })
            print(f"  {p['name']:<22} : {res['tokens']} tok in {res['time_s']:.2f}s -> {res['tok_s']:5.2f} tok/s | acc: {res['mtp_accepted']:3d}/{res['mtp_drafted']:3d} ({res['mtp_rate']:5.1f}%) | SHA: {res['sha256'][:8]}")
            sys.stdout.flush()

        speeds = [r["tok_s"] for r in prompt_results]
        avg_speed = sum(speeds) / len(speeds)
        total_drafted = sum(r["mtp_drafted"] for r in prompt_results)
        total_accepted = sum(r["mtp_accepted"] for r in prompt_results)
        overall_acc = (100.0 * total_accepted / total_drafted) if total_drafted > 0 else 0.0

        print(f"  >> AVERAGE: {avg_speed:5.2f} tok/s | Overall MTP Accept: {total_accepted}/{total_drafted} ({overall_acc:.1f}%)")
        sys.stdout.flush()

        results[key] = {
            "margin": m,
            "draft": d,
            "label": label,
            "avg_tok_s": avg_speed,
            "overall_accepted": total_accepted,
            "overall_drafted": total_drafted,
            "overall_acceptance_rate": overall_acc,
            "prompts": prompt_results
        }

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    # Print final summary table
    print("\n\n" + "="*95)
    print("                          FINAL MTP BENCHMARK MATRIX")
    print("="*95)
    header = f"{'Margin':<8} | {'Draft 3':<22} | {'Draft 5':<22} | {'Draft 7':<22}"
    print(header)
    print("-" * 95)
    
    for m in margins:
        row = [f"{m:<8}"]
        for d in drafts:
            k = f"m_{m}_d_{d}"
            if k in results:
                s = results[k]["avg_tok_s"]
                acc = results[k]["overall_acceptance_rate"]
                row.append(f"{s:5.2f} tok/s ({acc:4.1f}%)")
            else:
                row.append("N/A")
        print(" | ".join(row))
    print("-" * 95)
    
    if "m_99.0_d_1" in results:
        base = results["m_99.0_d_1"]["avg_tok_s"]
        print(f"Baseline (MTP Off) Average: {base:5.2f} tok/s")
    print("="*95)

if __name__ == "__main__":
    main()
