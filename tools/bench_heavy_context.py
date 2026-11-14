#!/usr/bin/env python3
import json
import time
import hashlib
import sys
import os
import urllib.request
import urllib.error

BENCH_HEAVY_PROMPTS = [
    {
        "name": "1. Heavy C++20 Async RingBuffer & Transport",
        "category": "code",
        "prompt": """Below is the production header definition for an asynchronous network engine in C++20:

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <string_view>
#include <array>
#include <optional>
#include <span>

namespace net::engine {

enum class PacketType : uint16_t {
    HANDSHAKE_INIT = 0x1001,
    HANDSHAKE_ACK  = 0x1002,
    DATA_HEARTBEAT = 0x2001,
    DATA_PAYLOAD   = 0x2002,
    STREAM_RESET   = 0x3001,
    DISCONNECT     = 0x4001
};

enum class TransportError : int32_t {
    OK = 0,
    BUFFER_OVERFLOW = -1,
    INVALID_MAGIC = -2,
    SEQUENCE_DESYNC = -3,
    CRC_MISMATCH = -4,
    SESSION_EXPIRED = -5
};

#pragma pack(push, 1)
struct PacketHeader {
    uint32_t magic{0x53545258}; // 'STRX'
    PacketType type{PacketType::DATA_PAYLOAD};
    uint16_t version{1};
    uint64_t sequence_id{0};
    uint64_t timestamp_ns{0};
    uint32_t payload_len{0};
    uint32_t payload_crc32{0};
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 32, "PacketHeader must be exactly 32 bytes packed");

struct alignas(64) ConnectionMetrics {
    std::atomic<uint64_t> packets_received{0};
    std::atomic<uint64_t> packets_dropped{0};
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint32_t> last_latency_micros{0};
};

template <size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    alignas(64) std::array<uint8_t, Capacity> storage_{};
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
public:
    LockFreeRingBuffer() = default;
    [[nodiscard]] size_t available() const noexcept {
        return Capacity - (head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_relaxed));
    }
    bool push(std::span<const uint8_t> data) noexcept {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_acquire);
        if (Capacity - (h - t) < data.size()) return false;
        for (size_t i = 0; i < data.size(); ++i) {
            storage_[(h + i) & (Capacity - 1)] = data[i];
        }
        head_.store(h + data.size(), std::memory_order_release);
        return true;
    }
};

class AsyncTcpTransport {
    LockFreeRingBuffer<65536> rx_ring_;
    LockFreeRingBuffer<65536> tx_ring_;
    ConnectionMetrics metrics_;
    uint64_t expected_sequence_{0};
public:
    AsyncTcpTransport() = default;
    TransportError handle_incoming_packet(std::span<const uint8_t> raw_bytes);
    TransportError flush_tx_ring(std::span<uint8_t> out_buffer, size_t &bytes_written);
    const ConnectionMetrics &metrics() const noexcept { return metrics_; }
};

} // namespace net::engine
```

Task:
Write the complete `.cpp` implementation for `AsyncTcpTransport::handle_incoming_packet` and `AsyncTcpTransport::flush_tx_ring`. You must strictly check `PacketHeader::magic == 0x53545258`, validate `PacketHeader::payload_len`, check sequence desynchronization against `expected_sequence_`, update `metrics_.packets_received`, `metrics_.packets_dropped`, and `metrics_.bytes_transferred`, and return appropriate `TransportError` enum values."""
    },
    {
        "name": "2. Heavy Kubernetes Helm Values to Deployment YAML",
        "category": "config",
        "prompt": """Here is an enterprise Kubernetes Helm values configuration file for a high-availability banking API gateway and persistence cluster:

```yaml
global:
  environment: production
  clusterName: eu-central-prod-01
  datacenter: frankfurt-dc2

database:
  ha:
    enabled: true
    replicas: 3
    image:
      repository: internal-registry.corp.net/infra/postgresql-ha
      tag: 16.4-alpine
      pullPolicy: IfNotPresent
    credentials:
      secretName: postgres-production-credentials
      userKey: DB_SUPERUSER
      passwordKey: DB_SUPERUSER_PASSWORD
      databaseKey: DB_INITIAL_NAME
    resources:
      limits:
        cpu: 16000m
        memory: 64Gi
      requests:
        cpu: 8000m
        memory: 32Gi
    persistence:
      storageClass: nvme-replication-tier1
      size: 1024Gi
      accessModes: [ReadWriteOnce]
    pgpool:
      port: 5432
      maxPool: 500
      childLifeTime: 300
      connectionLifeTime: 0
    healthCheck:
      livenessPath: /tmp/.pgpool_live
      readinessPath: /tmp/.pgpool_ready
      initialDelaySeconds: 30
      periodSeconds: 10

gateway:
  serviceName: api-gateway-edge
  replicas: 6
  containerPort: 8443
  metricsPort: 9090
  env:
    - name: GATEWAY_ENV
      value: "production"
    - name: MAX_CONCURRENT_STREAMS
      value: "10000"
    - name: RATE_LIMIT_PER_IP_SEC
      value: "250"
    - name: UPSTREAM_TIMEOUT_MS
      value: "1500"
    - name: TLS_CIPHER_SUITE
      value: "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
              - key: app.kubernetes.io/name
                operator: In
                values: [api-gateway-edge]
          topologyKey: "topology.kubernetes.io/zone"
```

Task:
Generate a complete, fully expanded Kubernetes `Deployment` manifest for `api-gateway-edge` and the corresponding `StatefulSet` manifest for the PostgreSQL database cluster. You must explicitly translate all environment variables, ports, volume claims, resource limits/requests, readiness/liveness probes, and affinity rules verbatim from the configuration above."""
    },
    {
        "name": "3. Heavy OpenAPI 3.1 & Nested JSON Schema Payload",
        "category": "json",
        "prompt": """Below is an OpenAPI 3.1 schema specification for an enterprise cloud billing and automated invoice generation system:

```json
{
  "openapi": "3.1.0",
  "components": {
    "schemas": {
      "EnterpriseInvoice": {
        "type": "object",
        "required": ["invoice_id", "billing_period", "customer_profile", "line_items", "tax_breakdowns", "summary"],
        "properties": {
          "invoice_id": { "type": "string", "pattern": "^INV-[0-9]{4}-[A-Z]{3}-[0-9]{6}$" },
          "billing_period": {
            "type": "object",
            "required": ["period_start", "period_end", "currency_code", "payment_terms_days"],
            "properties": {
              "period_start": { "type": "string", "format": "date-time" },
              "period_end": { "type": "string", "format": "date-time" },
              "currency_code": { "type": "string", "enum": ["USD", "EUR", "GBP", "CHF"] },
              "payment_terms_days": { "type": "integer", "enum": [15, 30, 45, 60] }
            }
          },
          "customer_profile": {
            "type": "object",
            "required": ["customer_id", "legal_entity_name", "tax_identification_number", "billing_address"],
            "properties": {
              "customer_id": { "type": "string" },
              "legal_entity_name": { "type": "string" },
              "tax_identification_number": { "type": "string" },
              "billing_address": {
                "type": "object",
                "required": ["street_line_1", "city", "state_province", "postal_code", "country_iso2"],
                "properties": {
                  "street_line_1": { "type": "string" },
                  "city": { "type": "string" },
                  "state_province": { "type": "string" },
                  "postal_code": { "type": "string" },
                  "country_iso2": { "type": "string", "minLength": 2, "maxLength": 2 }
                }
              }
            }
          },
          "line_items": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["item_sku", "service_name", "usage_units", "unit_price", "discount_percentage", "line_subtotal"],
              "properties": {
                "item_sku": { "type": "string" },
                "service_name": { "type": "string" },
                "usage_units": { "type": "number" },
                "unit_price": { "type": "number" },
                "discount_percentage": { "type": "number" },
                "line_subtotal": { "type": "number" }
              }
            }
          },
          "tax_breakdowns": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["jurisdiction_name", "tax_type", "rate_percentage", "taxable_amount", "tax_amount"],
              "properties": {
                "jurisdiction_name": { "type": "string" },
                "tax_type": { "type": "string", "enum": ["VAT", "STATE_SALES", "MUNICIPAL_SURCHARGE"] },
                "rate_percentage": { "type": "number" },
                "taxable_amount": { "type": "number" },
                "tax_amount": { "type": "number" }
              }
            }
          },
          "summary": {
            "type": "object",
            "required": ["subtotal_amount", "discount_total", "tax_total", "grand_total_amount", "escrow_status"],
            "properties": {
              "subtotal_amount": { "type": "number" },
              "discount_total": { "type": "number" },
              "tax_total": { "type": "number" },
              "grand_total_amount": { "type": "number" },
              "escrow_status": { "type": "string", "enum": ["PENDING_CLEARANCE", "HELD_IN_ESCROW", "SETTLED"] }
            }
          }
        }
      }
    }
  }
}
```

Task:
Generate a complete, syntactically valid JSON document adhering strictly to the `EnterpriseInvoice` schema above. Provide 4 realistic line items (compute instances, nvme storage, ingress bandwidth, and managed database hours) with accurate math matching subtotals, tax breakdowns, and grand total."""
    },
    {
        "name": "4. Heavy Linux Kernel Socket Drop Trace Logs Extraction",
        "category": "logs",
        "prompt": """Below is a raw kernel trace log snippet captured using eBPF during high network congestion:

```text
[2026-11-15 00:01:10.104231] CPU#04 [pid=4102/envoy] func=tcp_v4_rcv src=10.240.12.44:54320 dst=10.240.0.10:8443 proto=TCP drop_reason=NO_SOCKET latency_us=14.2
[2026-11-15 00:01:10.104289] CPU#04 [pid=4102/envoy] func=tcp_v4_rcv src=10.240.12.45:54322 dst=10.240.0.10:8443 proto=TCP drop_reason=NO_SOCKET latency_us=12.8
[2026-11-15 00:01:10.105112] CPU#11 [pid=8831/ingress] func=kfree_skb src=192.168.100.12:443 dst=10.240.4.15:39102 proto=TCP drop_reason=QUEUE_PURGE latency_us=88.5
[2026-11-15 00:01:10.105244] CPU#11 [pid=8831/ingress] func=kfree_skb src=192.168.100.14:443 dst=10.240.4.16:39104 proto=TCP drop_reason=QUEUE_PURGE latency_us=92.1
[2026-11-15 00:01:10.106004] CPU#02 [pid=1204/kube-proxy] func=nf_hook_slow src=172.16.0.4:12345 dst=10.96.0.1:443 proto=TCP drop_reason=NETFILTER_DROP latency_us=45.6
[2026-11-15 00:01:10.106201] CPU#02 [pid=1204/kube-proxy] func=nf_hook_slow src=172.16.0.5:12346 dst=10.96.0.1:443 proto=TCP drop_reason=NETFILTER_DROP latency_us=48.2
[2026-11-15 00:01:10.107018] CPU#08 [pid=4102/envoy] func=tcp_v4_do_rcv src=10.240.12.48:54328 dst=10.240.0.10:8443 proto=TCP drop_reason=TCP_OFOMARK latency_us=21.4
[2026-11-15 00:01:10.107150] CPU#08 [pid=4102/envoy] func=tcp_v4_do_rcv src=10.240.12.49:54330 dst=10.240.0.10:8443 proto=TCP drop_reason=TCP_OFOMARK latency_us=23.1
[2026-11-15 00:01:10.108001] CPU#14 [pid=9912/clickhouse] func=ip_rcv_finish src=10.240.8.20:9000 dst=10.240.8.21:9000 proto=TCP drop_reason=PKT_TOO_BIG latency_us=110.4
[2026-11-15 00:01:10.108192] CPU#14 [pid=9912/clickhouse] func=ip_rcv_finish src=10.240.8.22:9000 dst=10.240.8.21:9000 proto=TCP drop_reason=PKT_TOO_BIG latency_us=108.9
```

Task:
Parse all 10 trace log lines above into a Markdown table with columns: Timestamp, CPU, Process, PID, Function, Source IP/Port, Dest IP/Port, Drop Reason, Latency (µs). Then calculate:
1) Total packet drops grouped by Process and PID.
2) Total packet drops grouped by Drop Reason.
3) Average latency for each distinct Kernel Function."""
    },
    {
        "name": "5. Heavy Russian LSM-Tree Storage Architecture QA",
        "category": "russian",
        "prompt": """Ниже приведена подробная инженерная спецификация движка хранения данных на основе архитектуры LSM-дерева (Log-Structured Merge-tree):

Архитектура подсистемы записи и уплотнения (Compaction):
Все входящие мутации (Put, Delete, Merge) сначала записываются в циклический буфер журнала упреждающей записи (Write-Ahead Log, WAL) с вызовом fsync() в соответствии с политикой синхронизации. Одновременно мутация вставляется в активную структуру MemTable в оперативной памяти, реализованную в виде блокирующего Concurrent SkipList с вероятностным фактором ветвления p = 1/4 и максимальной глубиной 32 уровня.

Когда размер активного MemTable превышает пороговое значение `write_buffer_size` (по умолчанию 64 МиБ), он становится доступным только для чтения (Immutable MemTable), а новый пустой MemTable назначается активным. Фоновый пул потоков сбрасывает Immutable MemTable на диск в виде SSTable-файла уровня Level 0 (L0). Для каждого блока данных размером 4 КиБ строится фильтр Блума (Bloom Filter) с выделением 10 бит на ключ, что гарантирует теоретическую вероятность ложноположительного срабатывания (False Positive Rate, FPR) около 0.82% по формуле FPR = (1 - e^(-k*n/m))^k при оптимальном числе хеш-функций k = ln(2) * (m/n) ≈ 0.693 * 10 ≈ 7.

Правила ступенчатого уплотнения (Leveled Compaction):
1) Уровень L0 содержит перекрывающиеся диапазоны ключей. Базовый лимит файлов L0 равен `level0_file_num_compaction_trigger = 4`.
2) Уровни от L1 до Lmax содержат строго непересекающиеся диапазоны ключей в файлах фиксированного размера `target_file_size_base = 64 МиБ`.
3) Максимальный совокупный объем уровня L1 равен `max_bytes_for_level_base = 256 МиБ`. Для каждого последующего уровня Li максимальный объем рассчитывается с коэффициентом умножения `max_bytes_for_level_multiplier = 10`: TargetSize(Li) = TargetSize(L_{i-1}) * 10 (L2 = 2.56 ГиБ, L3 = 25.6 ГиБ, L4 = 256 ГиБ).
4) Write Stall (принудительное замедление записи): если число файлов L0 достигает `level0_slowdown_writes_trigger = 8`, входящие записи искусственно задерживаются на 1 миллисекунду; если число файлов L0 достигает `level0_stop_writes_trigger = 12`, либо число неподтвержденных байт компакшена превышает `max_write_buffer_number_to_maintain * write_buffer_size`, все операции записи полностью блокируются до завершения сброса.

На основе приведенной спецификации дай подробный технический ответ на следующие вопросы:
1) Объясни математику фильтра Блума: почему при m/n = 10 число хеш-функций k = 7 минимизирует FPR, и какой процент ненужных дисковых чтений отсекается?
2) При каких точных условиях и пороговых значениях включается замедление (slowdown) и полная остановка (stop stall) записи?
3) Какова формула расчета совокупного объема для уровня L3 и сколько максимум файлов SSTable по 64 МиБ может на нем храниться?
4) Опиши пошаговый процесс аварийного восстановления (Crash Recovery), если сбой питания произошел в момент сброса Immutable MemTable на диск."""
    }
]

def set_engine_params(base_url, margin=1.5, draft=7, pld=False, pld_n=3, pld_k=6, dynamic=False, draft_min=2, draft_max=12):
    pld_param = "1" if pld else "0"
    dyn_param = "1" if dynamic else "0"
    url = f"{base_url}/set_mtp?margin={margin}&draft={draft}&pld={pld_param}&pld_n={pld_n}&pld_k={pld_k}&dynamic={dyn_param}&draft_min={draft_min}&draft_max={draft_max}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def run_single(base_url, prompt, max_tokens=600):
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
        "mtp_rate": strix.get("mtp_acceptance_rate", 0.0),
        "pld_rounds": strix.get("pld_rounds", 0),
        "pld_drafted": strix.get("pld_drafted_tokens", 0),
        "pld_accepted": strix.get("pld_accepted_tokens", 0),
        "pld_rate": strix.get("pld_acceptance_rate", 0.0)
    }

def run_suite(base_url, pld_enabled, label, n_runs=2):
    print(f"\n{'='*95}")
    print(f"  RUNNING SUITE: {label} (N={n_runs})")
    print(f"{'='*95}")
    sys.stdout.flush()

    suite_results = {p["name"]: [] for p in BENCH_HEAVY_PROMPTS}
    
    for r in range(1, n_runs + 1):
        print(f"\n--- Iteration {r} of {n_runs} ---")
        for p in BENCH_HEAVY_PROMPTS:
            res = run_single(base_url, p["prompt"])
            suite_results[p["name"]].append(res)
            pld_info = f" | PLD: {res['pld_accepted']}/{res['pld_drafted']} ({res['pld_rate']:.1f}%)" if pld_enabled else ""
            print(f"  {p['name']:<48} : {res['tokens']} tok in {res['time_s']:.2f}s -> {res['tok_s']:5.2f} tok/s | MTP: {res['mtp_accepted']}/{res['mtp_drafted']} ({res['mtp_rate']:.1f}%){pld_info} | SHA: {res['sha256'][:8]}")
            sys.stdout.flush()
        time.sleep(1)

    summary = {}
    for p in BENCH_HEAVY_PROMPTS:
        name = p["name"]
        runs = suite_results[name]
        speeds = [x["tok_s"] for x in runs]
        hashes = [x["sha256"] for x in runs]
        avg_s = sum(speeds) / len(speeds)
        pld_acc = sum(x["pld_accepted"] for x in runs)
        pld_drf = sum(x["pld_drafted"] for x in runs)
        mtp_acc = sum(x["mtp_accepted"] for x in runs)
        mtp_drf = sum(x["mtp_drafted"] for x in runs)
        summary[name] = {
            "avg_tok_s": avg_s,
            "speeds": speeds,
            "hashes": hashes,
            "hash_uniform": len(set(hashes)) == 1,
            "pld_accepted": pld_acc,
            "pld_drafted": pld_drf,
            "mtp_accepted": mtp_acc,
            "mtp_drafted": mtp_drf
        }
    return summary

def main():
    base_url = "http://127.0.0.1:5400"
    out_file = "/home/cheblan/heavy_context_benchmark_results.json"
    
    print("===============================================================================================")
    print("      HEAVY CONTEXT / REAL-WORLD CODE BENCHMARK: STATIC MTP vs DYNAMIC ADAPTIVE MTP")
    print("===============================================================================================")
    sys.stdout.flush()

    # 1. Baseline: Static MTP Draft 7, Margin 1.5, Dynamic OFF, PLD OFF
    print("\nSetting up Baseline: Static MTP Draft 7, Dynamic OFF, PLD OFF...")
    st = set_engine_params(base_url, margin=1.5, draft=7, pld=False, dynamic=False)
    print(f"Server acknowledged: {st}")
    baseline_summary = run_suite(base_url, pld_enabled=False, label="Baseline (Static MTP Draft 7)", n_runs=2)

    # 2. Dynamic Adaptive MTP: initial=7, min=2, max=12, Margin 1.5, Dynamic ON, PLD OFF
    print("\nSetting up Dynamic Adaptive MTP (min=2, max=12, initial=7, Dynamic ON)...")
    st = set_engine_params(base_url, margin=1.5, draft=7, pld=False, dynamic=True, draft_min=2, draft_max=12)
    print(f"Server acknowledged: {st}")
    dyn_summary = run_suite(base_url, pld_enabled=False, label="Dynamic Adaptive MTP (Draft 2..12)", n_runs=2)

    # Comparison and Bit Identity Check
    print("\n\n" + "="*105)
    print(f"{'Task / Domain':<50} | {'Static (7)':<12} | {'Dynamic (2..12)':<15} | {'Speedup':<9} | {'Bit Identity'}")
    print("-" * 105)

    base_avg_list = []
    dyn_avg_list = []

    for p in BENCH_HEAVY_PROMPTS:
        name = p["name"]
        b = baseline_summary[name]
        d = dyn_summary[name]
        b_speed = b["avg_tok_s"]
        d_speed = d["avg_tok_s"]
        delta = (d_speed - b_speed) / b_speed * 100.0
        base_avg_list.append(b_speed)
        dyn_avg_list.append(d_speed)
        
        # Check hash match between baseline and Dynamic MTP
        bit_exact = (b["hashes"][0] == d["hashes"][0])
        bit_str = "MATCH (Bit-Exact)" if bit_exact else f"MISMATCH ({b['hashes'][0][:8]} vs {d['hashes'][0][:8]})"
        
        print(f"{name:<50} | {b_speed:5.2f} tok/s | {d_speed:5.2f} tok/s   | {delta:+6.2f}% | {bit_str}")

    print("-" * 105)
    total_base = sum(base_avg_list) / len(base_avg_list)
    total_dyn = sum(dyn_avg_list) / len(dyn_avg_list)
    total_delta = (total_dyn - total_base) / total_base * 100.0
    print(f"{'OVERALL AVERAGE':<50} | {total_base:5.2f} tok/s | {total_dyn:5.2f} tok/s   | {total_delta:+6.2f}% |")
    print("=" * 105)

    # Save to JSON
    with open(out_file, "w") as f:
        json.dump({
            "baseline": baseline_summary,
            "hybrid_pld": pld_summary,
            "overall_baseline": total_base,
            "overall_pld": total_pld,
            "overall_speedup_percent": total_delta
        }, f, indent=2)
    print(f"\nFull benchmark results saved to {out_file}")

if __name__ == "__main__":
    main()
