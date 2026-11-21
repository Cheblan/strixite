import subprocess, os, re

magic = b"__CLANG_OFFLOAD_BUNDLE__"
with open("/tmp/server_fatbin5.bin", "rb") as f:
    data = f.read()

offsets = []
idx = 0
while True:
    pos = data.find(magic, idx)
    if pos == -1: break
    offsets.append(pos)
    idx = pos + len(magic)
offsets.append(len(data))

all_kernels = {}
total_kernels = 0
spilled_kernels = []

for i in range(len(offsets) - 1):
    chunk = data[offsets[i]:offsets[i+1]]
    b_path = f"/tmp/bundles5/b_{i}.bin"
    dev_path = f"/tmp/bundles5/dev_{i}.o"
    res = subprocess.run(["clang-offload-bundler", "--type=o", "--unbundle", f"--inputs={b_path}", "--targets=hip-amdgcn-amd-amdhsa--gfx1151", f"--outputs={dev_path}"], capture_output=True)
    if res.returncode != 0:
        continue
    out = subprocess.check_output(["llvm-readelf", "--notes", dev_path]).decode("utf-8", errors="ignore")
    kernels = out.split("  - .args:")
    for k in kernels:
        m_name = re.search(r"\.name:\s+([^\n]+)", k)
        if not m_name: continue
        name = m_name.group(1)
        vgpr = re.search(r"\.vgpr_count:\s+(\d+)", k)
        spill = re.search(r"\.vgpr_spill_count:\s+(\d+)", k)
        priv = re.search(r"\.private_segment_fixed_size:\s+(\d+)", k)
        v = int(vgpr.group(1)) if vgpr else 0
        s = int(spill.group(1)) if spill else 0
        p = int(priv.group(1)) if priv else 0
        total_kernels += 1
        
        cat = "other"
        if "attention" in name: cat = "attention"
        elif "linear_q4_experts" in name: cat = "moe_experts (q4/q5)"
        elif "grouped_wmma" in name: cat = "moe_grouped"
        elif "hc_down" in name: cat = "hc_down"
        elif "linear_q4_kernel" in name: cat = "linear_q4"
        elif "linear_q5_kernel" in name: cat = "linear_q5"
        elif "linear_q6_kernel" in name: cat = "linear_q6"
        elif "linear_q8_kernel" in name: cat = "linear_q8"
        elif "linear_wmma" in name: cat = "linear_wmma"
        elif "delta_rule" in name: cat = "gdn"
        elif "mtp" in name: cat = "mtp"
        
        if cat not in all_kernels:
            all_kernels[cat] = {"count": 0, "spills": 0, "max_vgpr": 0, "min_vgpr": 999}
        st = all_kernels[cat]
        st["count"] += 1
        st["max_vgpr"] = max(st["max_vgpr"], v)
        st["min_vgpr"] = min(st["min_vgpr"], v)
        if p > 0:
            st["spills"] += 1
            spilled_kernels.append((cat, name, v, s, p))

print(f"=== COMPREHENSIVE KERNEL AUDIT REPORT ({total_kernels} KERNELS) ===")
header = f"{'Module / Family':<25} | {'Kernels':>7} | {'Spills':>6} | {'VGPR Range':>12}"
print(header)
print("-" * len(header))
for cat, st in sorted(all_kernels.items()):
    vgpr_str = f"{st['min_vgpr']}..{st['max_vgpr']}"
    print(f"{cat:<25} | {st['count']:>7} | {st['spills']:>6} | {vgpr_str:>12}")

print("\n--- ALL REMAINING SPILLED KERNELS IN ENTIRE BINARY ---")
if not spilled_kernels:
    print("NONE! 100% Zero Spills!")
else:
    for cat, name, v, s, p in spilled_kernels:
        print(f"[{cat}] VGPR: {v:>3} | Spill: {s:>2} | Private: {p:>3} B | {name[:75]}")
