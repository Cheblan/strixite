#!/usr/bin/env python3
"""
mtp_model.py - PyTorch mirror of Qwen 3.8-Flash-Next MTP Head.

Designed for Strategy C1 training (90.6M Non-MoE trainable, 512 experts frozen).
Runs on AMD Radeon RX 7900 XTX (ROCm 10.1 / gfx1100).
"""

import os
import sys

# Ensure jemalloc alignment is preloaded to prevent ROCm AVX-512 unaligned segfault
if "/usr/lib/libjemalloc.so" not in os.environ.get("LD_PRELOAD", ""):
    if os.path.exists("/usr/lib/libjemalloc.so"):
        os.environ["LD_PRELOAD"] = "/usr/lib/libjemalloc.so"
        os.execv(sys.executable, [sys.executable] + sys.argv)

if os.environ.get("ROCPROFILER_REGISTER_ENABLED") != "0":
    raise RuntimeError(
        "ROCPROFILER_REGISTER_ENABLED=0 must be set in environment before running this script!\n"
        "Example: ROCPROFILER_REGISTER_ENABLED=0 python3 ..."
    )

import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extract_overlay import parse_strixw


class RMSNormZC(nn.Module):
    """Zero-centered RMSNorm: y = (x * rsqrt(mean(x^2) + eps)) * (1 + w)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)) * (1.0 + self.weight)


class HyperConnectionMix(nn.Module):
    """
    Hyper-Connections mixing module.
    mix_down: down projection [R, H*d] with silu for h and optional 2*sigmoid for w_in
    mix_up: up projection [H*d, r] with sigmoid gate and (1 + hc_norm)
    """
    def __init__(self, H: int = 4, d: int = 2560, r: int = 320, inject: bool = True, eps: float = 1e-6):
        super().__init__()
        self.H = H
        self.d = d
        self.r = r
        self.inject = inject
        self.eps = eps
        R = r + (H if inject else 0)
        self.w_down = nn.Parameter(torch.zeros(R, H * d, dtype=torch.float32))
        self.w_up = nn.Parameter(torch.zeros(H * d, r, dtype=torch.float32))
        self.hc_norm = nn.Parameter(torch.zeros(H, d, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        # x: [..., H, d]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        x_flat = x_norm.reshape(*x.shape[:-2], self.H * self.d)
        
        a = F.linear(x_flat, self.w_down)
        z = a / float(self.H)
        h = F.silu(z[..., :self.r])
        w_in = 2.0 * torch.sigmoid(z[..., self.r:self.r + self.H]) if self.inject else None

        acc = F.linear(h, self.w_up).reshape(*x.shape[:-2], self.H, self.d)
        gate = torch.sigmoid(acc)
        normed_stream = x_norm * (1.0 + self.hc_norm)
        contrib = gate * normed_stream
        u = contrib.sum(dim=-2) / float(self.H)
        return u, w_in


class MtpAttention(nn.Module):
    """Dense Causal Gated Attention (24 Q heads, 2 KV heads, head_dim 256, rot_dim 64)."""
    def __init__(self, d: int = 2560, hq: int = 24, hkv: int = 2, hd: int = 256, rot: int = 64, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.hq = hq
        self.hkv = hkv
        self.hd = hd
        self.rot = rot
        self.eps = eps
        self.scale = 1.0 / (hd ** 0.5)

        # qkv [13952, 2560]: 12288 (Q+Gate), 512 (K), 512 (V), 640 (Indexer unused for seq <= 2048)
        self.qkv = nn.Linear(d, 13952, bias=False)
        self.o_proj = nn.Linear(hq * hd, d, bias=False)
        self.q_norm = RMSNormZC(hd, eps=eps)
        self.k_norm = RMSNormZC(hd, eps=eps)

        # Precompute rotary frequencies
        inv_freq = 1.0 / (10000000.0 ** (torch.arange(0, rot // 2, dtype=torch.float32) * 2.0 / float(rot)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_rope(self, x: torch.Tensor, pos: int):
        # x: [..., num_heads, 256]
        x_rot = x[..., :self.rot]
        x_pass = x[..., self.rot:]
        
        angle = pos * self.inv_freq.to(x.device)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        cos_rot = torch.cat([cos, cos], dim=-1)
        sin_rot = torch.cat([sin, sin], dim=-1)

        x1 = x_rot[..., :self.rot // 2]
        x2 = x_rot[..., self.rot // 2:self.rot]
        rotated = torch.cat([-x2, x1], dim=-1)
        x_embed = (x_rot * cos_rot) + (rotated * sin_rot)
        return torch.cat([x_embed, x_pass], dim=-1)

    def forward(self, u: torch.Tensor, pos: int, k_cache: torch.Tensor = None, v_cache: torch.Tensor = None):
        # u: [..., d] (single token step: [d] or [1, d])
        qkv_out = self.qkv(u)
        
        q_all = qkv_out[..., :self.hq * 2 * self.hd].reshape(*u.shape[:-1], self.hq, 2 * self.hd)
        q = q_all[..., :self.hd]
        q_gate = q_all[..., self.hd:]
        k = qkv_out[..., self.hq * 2 * self.hd:self.hq * 2 * self.hd + self.hkv * self.hd].reshape(*u.shape[:-1], self.hkv, self.hd)
        v = qkv_out[..., self.hq * 2 * self.hd + self.hkv * self.hd:self.hq * 2 * self.hd + 2 * self.hkv * self.hd].reshape(*u.shape[:-1], self.hkv, self.hd)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q_rope = self.apply_rope(q, pos)
        k_rope = self.apply_rope(k, pos)

        # Update KV cache
        if k_cache is not None and v_cache is not None:
            k_full = torch.cat([k_cache, k_rope.unsqueeze(0) if k_rope.dim() == 2 else k_rope], dim=0)
            v_full = torch.cat([v_cache, v.unsqueeze(0) if v.dim() == 2 else v], dim=0)
        else:
            k_full = k_rope.unsqueeze(0) if k_rope.dim() == 2 else k_rope
            v_full = v.unsqueeze(0) if v.dim() == 2 else v

        # Repeat KV heads for GQA (2 -> 24)
        gqa_ratio = self.hq // self.hkv
        k_rep = k_full.repeat_interleave(gqa_ratio, dim=1) # [T_kv, 24, 256]
        v_rep = v_full.repeat_interleave(gqa_ratio, dim=1) # [T_kv, 24, 256]

        # Manual Attention without enable_gqa=True (to prevent ROCm gfx1100 segfault)
        scores = torch.einsum('hd,thd->ht', q_rope, k_rep) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.einsum('ht,thd->hd', attn_weights, v_rep)

        core = attn_out * torch.sigmoid(q_gate)
        core_flat = core.reshape(*u.shape[:-1], self.hq * self.hd)
        y = self.o_proj(core_flat)
        return y, k_full, v_full


class MtpMoE(nn.Module):
    """MoE block with 512 routed experts + 1 shared expert (Strategy C1: frozen)."""
    def __init__(self, d: int = 2560, experts: int = 512, top_k: int = 10, inter: int = 640):
        super().__init__()
        self.d = d
        self.experts = experts
        self.top_k = top_k
        self.inter = inter

        self.router = nn.Linear(d, experts + 1, bias=False)
        self.shared_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.shared_down = nn.Linear(inter, d, bias=False)

        # Routed experts storage (dequantized on-demand or pre-loaded)
        self.expert_gu_table = None
        self.expert_dn_table = None

    def set_expert_weights(self, gu_table, dn_table):
        self.expert_gu_table = gu_table
        self.expert_dn_table = dn_table

    def forward(self, u: torch.Tensor):
        # u: [d]
        router_logits = self.router(u)
        routed_logits = router_logits[..., :self.experts]
        topk_weights, topk_idx = torch.topk(routed_logits, k=self.top_k, dim=-1)
        topk_probs = F.softmax(topk_weights, dim=-1)

        y_moe = torch.zeros_like(u)
        for i in range(self.top_k):
            e_idx = int(topk_idx[..., i].item())
            w_gu = self.expert_gu_table[e_idx] # [1280, 2560]
            w_dn = self.expert_dn_table[e_idx] # [2560, 640]
            gu = F.linear(u, w_gu)
            gate = gu[..., :self.inter]
            up = gu[..., self.inter:]
            h_exp = F.silu(gate) * up
            out_exp = F.linear(h_exp, w_dn)
            y_moe = y_moe + topk_probs[..., i] * out_exp

        # Shared expert
        sh_gu = self.shared_gate_up(u)
        sh_gate = sh_gu[..., :self.inter]
        sh_up = sh_gu[..., self.inter:]
        sh_h = F.silu(sh_gate) * sh_up
        sh_out = self.shared_down(sh_h)
        sh_coeff = torch.sigmoid(router_logits[..., self.experts:self.experts + 1])

        return y_moe + sh_coeff * sh_out


class QwenMtpHead(nn.Module):
    """Complete Qwen 3.8-Flash-Next MTP Layer."""
    def __init__(self, d: int = 2560, H: int = 4, vocab_size: int = 248320):
        super().__init__()
        self.d = d
        self.H = H
        self.vocab_size = vocab_size

        # Trainable Non-MoE parameters (Strategy C1: ~90.6M params)
        self.pre_fc_norm_embedding = RMSNormZC(d)
        self.pre_fc_norm_hidden = nn.Parameter(torch.zeros(H, d, dtype=torch.float32))
        self.fc_embedding = nn.Linear(d, d, bias=False)
        self.fc_hidden = nn.Linear(d, d, bias=False)

        self.attn_hc = HyperConnectionMix(H=H, d=d, r=320, inject=True)
        self.self_attn = MtpAttention(d=d, hq=24, hkv=2, hd=256, rot=64)

        self.mlp_hc = HyperConnectionMix(H=H, d=d, r=320, inject=True)
        self.moe = MtpMoE(d=d, experts=512, top_k=10, inter=640)

        self.hc_mixer = HyperConnectionMix(H=H, d=d, r=320, inject=False)

        # Frozen components
        self.embed_tokens = None # [vocab_size, d]
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    @classmethod
    def from_bundle(cls, bundle_path: str, device: torch.device = torch.device("cuda")):
        print(f"Loading MTP bundle from: {bundle_path}")
        meta, tensors = parse_strixw(bundle_path)
        t_dict = {t["name"]: t for t in tensors}

        model = cls().to(device)

        with open(bundle_path, "rb") as f:
            def read_comp(c):
                f.seek(c["offset"])
                return f.read(c["bytes"])

            def dequant_q4(t):
                N, K = t["shape"]
                G = t["group_size"]
                q_b = np.frombuffer(read_comp(t["components"][0]), dtype=np.uint8).reshape(N, K // 2)
                sc = np.frombuffer(read_comp(t["components"][1]), dtype=np.uint16).reshape(N, K // G)
                mn = np.frombuffer(read_comp(t["components"][2]), dtype=np.uint16).reshape(N, K // G)
                q_t = torch.from_numpy(q_b.copy()).to(device)
                sc = torch.from_numpy(sc.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                mn = torch.from_numpy(mn.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                low = (q_t & 0x0f).to(torch.float32)
                high = ((q_t >> 4) & 0x0f).to(torch.float32)
                unpacked = torch.stack([low, high], dim=-1).reshape(N, K)
                return unpacked * sc.repeat_interleave(G, dim=1) + mn.repeat_interleave(G, dim=1)

            def dequant_q8_row(t):
                N, K = t["shape"]
                G = t["group_size"]
                q = np.frombuffer(read_comp(t["components"][0]), dtype=np.uint8).reshape(N, K)
                sc = np.frombuffer(read_comp(t["components"][1]), dtype=np.uint16).reshape(N, K // G)
                mn = np.frombuffer(read_comp(t["components"][2]), dtype=np.uint16).reshape(N, K // G)
                q = torch.from_numpy(q.copy()).to(torch.float32).to(device)
                sc = torch.from_numpy(sc.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                mn = torch.from_numpy(mn.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                return q * sc.repeat_interleave(G, dim=1) + mn.repeat_interleave(G, dim=1)

            def dequant_q8_cm(t):
                N, K = t["shape"]
                G = t["group_size"]
                nc = K // 32
                q = np.frombuffer(read_comp(t["components"][0]), dtype=np.uint8).reshape(nc, N, 32)
                sc = np.frombuffer(read_comp(t["components"][1]), dtype=np.uint16).reshape(K // G, N)
                mn = np.frombuffer(read_comp(t["components"][2]), dtype=np.uint16).reshape(K // G, N)
                q = torch.from_numpy(q.copy()).permute(1, 0, 2).reshape(N, K).to(torch.float32).to(device)
                sc = torch.from_numpy(sc.copy()).permute(1, 0).view(torch.bfloat16).to(torch.float32).to(device)
                mn = torch.from_numpy(mn.copy()).permute(1, 0).view(torch.bfloat16).to(torch.float32).to(device)
                return q * sc.repeat_interleave(G, dim=1) + mn.repeat_interleave(G, dim=1)

            def get_f32(t):
                return torch.from_numpy(np.frombuffer(read_comp(t["components"][0]), dtype=np.float32).copy()).to(device)

            def get_bf16(t):
                return torch.from_numpy(np.frombuffer(read_comp(t["components"][0]), dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).to(device)

            print("Dequantizing Non-MoE parameters...")
            with torch.no_grad():
                model.pre_fc_norm_embedding.weight.copy_(get_f32(t_dict["mtp.pre_fc_norm_embedding.weight"]))
                model.pre_fc_norm_hidden.copy_(get_f32(t_dict["mtp.pre_fc_norm_hidden.weight"]).reshape(4, 2560))
                model.fc_embedding.weight.copy_(dequant_q4(t_dict["mtp.fc_embedding.weight"]))
                model.fc_hidden.weight.copy_(dequant_q4(t_dict["mtp.fc_hidden.weight"]))

                model.attn_hc.w_down.copy_(dequant_q8_row(t_dict["mtp.layers.0.attn_hyper_connection.mix_down"]))
                model.attn_hc.w_up.copy_(dequant_q8_cm(t_dict["mtp.layers.0.attn_hyper_connection.mix_up"]))
                model.attn_hc.hc_norm.copy_(get_f32(t_dict["mtp.layers.0.attn_hyper_connection.hc_norm.weight"]).reshape(4, 2560))

                model.self_attn.qkv.weight.copy_(dequant_q8_row(t_dict["mtp.layers.0.self_attn.qkv"]))
                model.self_attn.o_proj.weight.copy_(dequant_q8_row(t_dict["mtp.layers.0.self_attn.o_proj.weight"]))
                model.self_attn.q_norm.weight.copy_(get_f32(t_dict["mtp.layers.0.self_attn.q_norm.weight"]))
                model.self_attn.k_norm.weight.copy_(get_f32(t_dict["mtp.layers.0.self_attn.k_norm.weight"]))

                model.mlp_hc.w_down.copy_(dequant_q8_row(t_dict["mtp.layers.0.mlp_hyper_connection.mix_down"]))
                model.mlp_hc.w_up.copy_(dequant_q8_cm(t_dict["mtp.layers.0.mlp_hyper_connection.mix_up"]))
                model.mlp_hc.hc_norm.copy_(get_f32(t_dict["mtp.layers.0.mlp_hyper_connection.hc_norm.weight"]).reshape(4, 2560))

                model.moe.router.weight.copy_(get_bf16(t_dict["mtp.layers.0.mlp.router"]).reshape(513, 2560))
                model.moe.shared_gate_up.weight.copy_(dequant_q8_row(t_dict["mtp.layers.0.mlp.shared_expert.gate_up"]))
                model.moe.shared_down.weight.copy_(dequant_q8_row(t_dict["mtp.layers.0.mlp.shared_expert.down_proj.weight"]))

                model.hc_mixer.w_down.copy_(dequant_q8_row(t_dict["mtp.hyper_connection_mixer.mix_down"]))
                model.hc_mixer.w_up.copy_(dequant_q8_cm(t_dict["mtp.hyper_connection_mixer.mix_up"]))
                model.hc_mixer.hc_norm.copy_(get_f32(t_dict["mtp.hyper_connection_mixer.hc_norm.weight"]).reshape(4, 2560))

                model.lm_head.weight.copy_(dequant_q8_row(t_dict["lm_head.weight"]))

                # Load embeddings table
                t_emb = t_dict["model.language_model.embed_tokens.weight"]
                f.seek(t_emb["components"][0]["offset"])
                emb_raw = np.frombuffer(f.read(t_emb["components"][0]["bytes"]), dtype=np.uint16)
                model.embed_tokens = torch.from_numpy(emb_raw.copy()).view(torch.bfloat16).to(torch.float32).to(device).reshape(model.vocab_size, model.d)

                # Pre-dequantize routed experts table for top-k fast forward
                print("Pre-dequantizing 512 MoE experts...")
                t_gu = t_dict["mtp.layers.0.mlp.experts_gate_up"]
                t_dn = t_dict["mtp.layers.0.mlp.experts_down"]
                gu_table = []
                dn_table = []
                N_gu, K_gu, G_gu = 1280, 2560, 128
                N_dn, K_dn, G_dn = 2560, 640, 128

                for idx in range(512):
                    f.seek(t_gu["components"][0]["offset"] + idx * (N_gu * K_gu // 2))
                    q_b = np.frombuffer(f.read(N_gu * K_gu // 2), dtype=np.uint8).reshape(N_gu, K_gu // 2)
                    f.seek(t_gu["components"][1]["offset"] + idx * (N_gu * (K_gu // G_gu) * 2))
                    sc = np.frombuffer(f.read(N_gu * (K_gu // G_gu) * 2), dtype=np.uint16).reshape(N_gu, K_gu // G_gu)
                    f.seek(t_gu["components"][2]["offset"] + idx * (N_gu * (K_gu // G_gu) * 2))
                    mn = np.frombuffer(f.read(N_gu * (K_gu // G_gu) * 2), dtype=np.uint16).reshape(N_gu, K_gu // G_gu)
                    q_t = torch.from_numpy(q_b.copy()).to(device)
                    sc = torch.from_numpy(sc.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                    mn = torch.from_numpy(mn.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                    low = (q_t & 0x0f).to(torch.float32)
                    high = ((q_t >> 4) & 0x0f).to(torch.float32)
                    unpacked = torch.stack([low, high], dim=-1).reshape(N_gu, K_gu)
                    w_gu = unpacked * sc.repeat_interleave(G_gu, dim=1) + mn.repeat_interleave(G_gu, dim=1)

                    f.seek(t_dn["components"][0]["offset"] + idx * (N_dn * K_dn // 2))
                    q_b_dn = np.frombuffer(f.read(N_dn * K_dn // 2), dtype=np.uint8).reshape(N_dn, K_dn // 2)
                    f.seek(t_dn["components"][1]["offset"] + idx * (N_dn * (K_dn // G_dn) * 2))
                    sc_dn = np.frombuffer(f.read(N_dn * (K_dn // G_dn) * 2), dtype=np.uint16).reshape(N_dn, K_dn // G_dn)
                    f.seek(t_dn["components"][2]["offset"] + idx * (N_dn * (K_dn // G_dn) * 2))
                    mn_dn = np.frombuffer(f.read(N_dn * (K_dn // G_dn) * 2), dtype=np.uint16).reshape(N_dn, K_dn // G_dn)
                    q_t_dn = torch.from_numpy(q_b_dn.copy()).to(device)
                    sc_dn = torch.from_numpy(sc_dn.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                    mn_dn = torch.from_numpy(mn_dn.copy()).view(torch.bfloat16).to(torch.float32).to(device)
                    low_dn = (q_t_dn & 0x0f).to(torch.float32)
                    high_dn = ((q_t_dn >> 4) & 0x0f).to(torch.float32)
                    unpacked_dn = torch.stack([low_dn, high_dn], dim=-1).reshape(N_dn, K_dn)
                    w_dn = unpacked_dn * sc_dn.repeat_interleave(G_dn, dim=1) + mn_dn.repeat_interleave(G_dn, dim=1)

                    gu_table.append(w_gu)
                    dn_table.append(w_dn)

                model.moe.set_expert_weights(gu_table, dn_table)

        print("QwenMtpHead loaded successfully!")
        return model

    def freeze_for_strategy_c1(self):
        """Freezes MoE router, shared expert, and lm_head for Strategy C1."""
        for p in self.moe.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False

    def forward_step(self, x: torch.Tensor, token_id: int, pos: int, k_cache: torch.Tensor = None, v_cache: torch.Tensor = None):
        """
        Executes single autoregressive draft step k.
        x: [4, 2560]
        token_id: int
        pos: int sequence position
        k_cache, v_cache: [pos, 2, 256]
        """
        # 1. Input embedding & normalization
        token_emb = self.embed_tokens[token_id]
        e_norm = self.pre_fc_norm_embedding(token_emb)

        eps = 1e-6
        var_h = x.pow(2).mean(dim=-1, keepdim=True)
        h_norm = (x * torch.rsqrt(var_h + eps)) * (1.0 + self.pre_fc_norm_hidden)

        proj_e = self.fc_embedding(e_norm)
        proj_h = self.fc_hidden(h_norm)
        mtp_x = proj_h + proj_e.unsqueeze(0)

        # 2. Attention
        u_attn, w_in_attn = self.attn_hc(mtp_x)
        y_attn, k_full, v_full = self.self_attn(u_attn, pos, k_cache, v_cache)
        mtp_x = mtp_x + w_in_attn.unsqueeze(1) * y_attn.unsqueeze(0)

        # 3. MLP / MoE
        u_mlp, w_in_mlp = self.mlp_hc(mtp_x)
        y_mlp = self.moe(u_mlp)
        mtp_x = mtp_x + w_in_mlp.unsqueeze(1) * y_mlp.unsqueeze(0)

        # 4. Mixer & LM Head
        u_mix, _ = self.hc_mixer(mtp_x)
        logits = self.lm_head(u_mix)

        return mtp_x, logits, k_full, v_full
