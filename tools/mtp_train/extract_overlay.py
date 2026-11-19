#!/usr/bin/env python3
"""
extract_overlay.py - Extract MTP tensors from a .strixw file into an overlay .strixw container.

By default extracts the 22 Non-MoE MTP tensors (~91 MB), leaving MoE routed experts in base model.
Can also extract all 24 MTP tensors if --all-mtp is specified.
"""

import os
import sys
import struct
import argparse

# ROCm 10.1 (HIP 7.15) requirement: must be set before runtime initialization
if os.environ.get("ROCPROFILER_REGISTER_ENABLED") != "0":
    raise RuntimeError(
        "ROCPROFILER_REGISTER_ENABLED=0 must be set in environment before running this script!\n"
        "Example: ROCPROFILER_REGISTER_ENABLED=0 python3 extract_overlay.py ..."
    )

kMagic = b'STRIXWT\0'
kStrixwVersion = 4
kStrixwHeaderBytes = 128
kStrixwTensorAlign = 4096
kStrixwComponentAlign = 256

kK1 = 0x9e3779b97f4a7c15
kK2 = 0xd6e8feb86659fd93
MASK64 = 0xffffffffffffffff


def align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def mix(h: int, w: int) -> int:
    h = (h ^ ((w * kK1) & MASK64)) & MASK64
    return (((h ^ (h >> 31)) & MASK64) * kK2) & MASK64


class StrixHasher:
    def __init__(self, total: int):
        self.total = total
        self.fed = 0
        self.h = [
            (total ^ kK1) & MASK64,
            (total ^ kK2) & MASK64,
            ((~total) ^ kK1) & MASK64,
            ((~total) ^ kK2) & MASK64,
        ]
        self.carry = bytearray()

    def consume_block(self, block):
        w0, w1, w2, w3 = struct.unpack('<4Q', block)
        self.h[0] = mix(self.h[0], w0)
        self.h[1] = mix(self.h[1], w1)
        self.h[2] = mix(self.h[2], w2)
        self.h[3] = mix(self.h[3], w3)

    def update(self, data):
        data = memoryview(data)
        n = len(data)
        self.fed += n
        if len(self.carry) > 0:
            take = min(n, 32 - len(self.carry))
            self.carry.extend(data[:take])
            data = data[take:]
            if len(self.carry) == 32:
                self.consume_block(self.carry)
                self.carry = bytearray()
        while len(data) >= 32:
            self.consume_block(data[:32])
            data = data[32:]
        if len(data) > 0:
            self.carry.extend(data)

    def final(self) -> int:
        c_len = len(self.carry)
        i = 0
        lane = 0
        while i + 8 <= c_len:
            w, = struct.unpack('<Q', self.carry[i:i+8])
            self.h[lane] = mix(self.h[lane], w)
            i += 8
            lane += 1
        if i < c_len:
            rem = self.carry[i:]
            w = int.from_bytes(rem, 'little')
            w = (w ^ ((c_len - i) << 56)) & MASK64
            self.h[0] = mix(self.h[0], w)
        r = mix(mix(mix(self.h[0], self.h[1]), self.h[2]), self.h[3])
        r = (r ^ (r >> 33)) & MASK64
        r = (r * kK1) & MASK64
        return (r ^ (r >> 29)) & MASK64


def strix_hash64(data: bytes) -> int:
    h = StrixHasher(len(data))
    h.update(data)
    return h.final()


class Cursor:
    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def get_u8(self) -> int:
        v = self.data[self.at]
        self.at += 1
        return v

    def get_u16(self) -> int:
        v, = struct.unpack_from('<H', self.data, self.at)
        self.at += 2
        return v

    def get_u32(self) -> int:
        v, = struct.unpack_from('<I', self.data, self.at)
        self.at += 4
        return v

    def get_u64(self) -> int:
        v, = struct.unpack_from('<Q', self.data, self.at)
        self.at += 8
        return v

    def get_i64(self) -> int:
        v, = struct.unpack_from('<q', self.data, self.at)
        self.at += 8
        return v

    def get_str16(self) -> str:
        slen = self.get_u16()
        s = self.data[self.at:self.at+slen].decode('utf-8', errors='replace')
        self.at += slen
        return s


def parse_strixw(path: str):
    with open(path, 'rb') as f:
        hdr = f.read(kStrixwHeaderBytes)
        if len(hdr) < kStrixwHeaderBytes or hdr[:8] != kMagic:
            raise ValueError(f"'{path}' is not a valid strixw file")
        ver, hdr_bytes = struct.unpack_from('<II', hdr, 8)
        idx_off, idx_bytes, data_off, file_bytes, idx_hash = struct.unpack_from('<QQQQQ', hdr, 16)
        f.seek(idx_off)
        idx_raw = f.read(idx_bytes)
        if strix_hash64(idx_raw) != idx_hash:
            raise ValueError(f"Index hash mismatch in '{path}'")

    cur = Cursor(idx_raw)
    n_meta = cur.get_u32()
    meta = {}
    for _ in range(n_meta):
        k = cur.get_str16()
        vlen = cur.get_u32()
        v = cur.data[cur.at:cur.at+vlen].decode('utf-8', errors='replace')
        cur.at += vlen
        meta[k] = v

    n_tensors = cur.get_u32()
    tensors = []
    for _ in range(n_tensors):
        tname = cur.get_str16()
        enc = cur.get_u8()
        n_dims = cur.get_u8()
        shape = [cur.get_i64() for _ in range(n_dims)]
        group_size = cur.get_i64()
        experts = cur.get_i64()
        n_parts = cur.get_u16()
        parts = []
        for _ in range(n_parts):
            src = cur.get_str16()
            roff = cur.get_i64()
            rows = cur.get_i64()
            parts.append({'source': src, 'row_offset': roff, 'rows': rows})
        n_comps = cur.get_u8()
        comps = []
        for _ in range(n_comps):
            role = cur.get_u8()
            off = cur.get_u64()
            cbytes = cur.get_u64()
            chash = cur.get_u64()
            comps.append({'role': role, 'offset': off, 'bytes': cbytes, 'hash': chash})
        tensors.append({
            'name': tname,
            'encoding': enc,
            'shape': shape,
            'group_size': group_size,
            'experts': experts,
            'parts': parts,
            'components': comps,
        })
    return meta, tensors


def encode_index(meta: dict, tensors: list) -> bytes:
    b = bytearray()
    b.extend(struct.pack('<I', len(meta)))
    for k, v in meta.items():
        kb = k.encode('utf-8')
        vb = v.encode('utf-8')
        b.extend(struct.pack('<H', len(kb)))
        b.extend(kb)
        b.extend(struct.pack('<I', len(vb)))
        b.extend(vb)
    b.extend(struct.pack('<I', len(tensors)))
    for t in tensors:
        nb = t['name'].encode('utf-8')
        b.extend(struct.pack('<H', len(nb)))
        b.extend(nb)
        b.extend(struct.pack('<B', t['encoding']))
        b.extend(struct.pack('<B', len(t['shape'])))
        for d in t['shape']:
            b.extend(struct.pack('<q', d))
        b.extend(struct.pack('<q', t['group_size']))
        b.extend(struct.pack('<q', t['experts']))
        b.extend(struct.pack('<H', len(t['parts'])))
        for p in t['parts']:
            pb = p['source'].encode('utf-8')
            b.extend(struct.pack('<H', len(pb)))
            b.extend(pb)
            b.extend(struct.pack('<q', p['row_offset']))
            b.extend(struct.pack('<q', p['rows']))
        b.extend(struct.pack('<B', len(t['components'])))
        for c in t['components']:
            b.extend(struct.pack('<B', c['role']))
            b.extend(struct.pack('<Q', c['offset']))
            b.extend(struct.pack('<Q', c['bytes']))
            b.extend(struct.pack('<Q', c['hash']))
    return bytes(b)


def extract_overlay(src_path: str, dst_path: str, all_mtp: bool = False, include_heads: bool = False):
    print(f"Reading source strixw: {src_path}")
    meta, tensors = parse_strixw(src_path)
    
    # Filter tensors
    selected = []
    for t in tensors:
        name = t['name']
        if include_heads and name in ('model.language_model.embed_tokens.weight', 'lm_head.weight'):
            selected.append(t)
            continue
        if not name.startswith("mtp."):
            continue
        if not all_mtp and name.startswith("mtp.layers.0.mlp.experts_"):
            continue
        selected.append(t)

    print(f"Selected {len(selected)} tensors for overlay:")
    total_data_bytes = 0
    for t in selected:
        t_bytes = sum(c['bytes'] for c in t['components'])
        total_data_bytes += t_bytes
        print(f"  {t['name']} ({t_bytes / 1024 / 1024:.2f} MB, {len(t['components'])} components)")
    print(f"Total data bytes: {total_data_bytes:,} B ({total_data_bytes / 1024 / 1024:.2f} MB)")

    # Prepare overlay meta
    overlay_meta = {
        'arch': meta.get('arch', 'qwen4exp'),
        'type': 'mtp-overlay',
        'generator': 'tools/mtp_train/extract_overlay.py',
    }

    # First pass: encode index with dummy offsets to find exact index size
    dummy_tensors = []
    for t in selected:
        dt = dict(t)
        dt['components'] = [{'role': c['role'], 'offset': 0, 'bytes': c['bytes'], 'hash': c['hash']} for c in t['components']]
        dummy_tensors.append(dt)
    
    idx_bytes = len(encode_index(overlay_meta, dummy_tensors))
    data_off = align_up(kStrixwHeaderBytes + idx_bytes, kStrixwTensorAlign)

    # Assign final offsets
    cur_off = data_off
    final_tensors = []
    for t in selected:
        cur_off = align_up(cur_off, kStrixwTensorAlign)
        new_comps = []
        for c in t['components']:
            cur_off = align_up(cur_off, kStrixwComponentAlign)
            new_comps.append({
                'role': c['role'],
                'offset': cur_off,
                'bytes': c['bytes'],
                'hash': c['hash'],
                '_src_offset': c['offset'],
            })
            cur_off += c['bytes']
        ft = dict(t)
        ft['components'] = new_comps
        final_tensors.append(ft)
    
    file_bytes = cur_off

    # Encode real index
    idx_encoded = encode_index(overlay_meta, final_tensors)
    assert len(idx_encoded) == idx_bytes, f"Index size changed ({len(idx_encoded)} vs {idx_bytes})"
    idx_hash = strix_hash64(idx_encoded)

    # Construct 128-byte header
    hdr = bytearray(kStrixwHeaderBytes)
    hdr[:8] = kMagic
    struct.pack_into('<I', hdr, 8, kStrixwVersion)
    struct.pack_into('<I', hdr, 12, kStrixwHeaderBytes)
    struct.pack_into('<Q', hdr, 16, kStrixwHeaderBytes)
    struct.pack_into('<Q', hdr, 24, idx_bytes)
    struct.pack_into('<Q', hdr, 32, data_off)
    struct.pack_into('<Q', hdr, 40, file_bytes)
    struct.pack_into('<Q', hdr, 48, idx_hash)
    hdr_hash = strix_hash64(bytes(hdr[:120]))
    struct.pack_into('<Q', hdr, 120, hdr_hash)

    # Write output file
    print(f"Writing overlay to: {dst_path} ({file_bytes / 1024 / 1024:.2f} MB)")
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with open(src_path, 'rb') as f_src, open(dst_path, 'wb') as f_dst:
        # Pre-allocate
        f_dst.truncate(file_bytes)
        # Write header & index
        f_dst.seek(0)
        f_dst.write(hdr)
        f_dst.write(idx_encoded)

        # Copy component data
        for t in final_tensors:
            for c in t['components']:
                f_src.seek(c['_src_offset'])
                data = f_src.read(c['bytes'])
                assert len(data) == c['bytes'], "Short read from source"
                assert strix_hash64(data) == c['hash'], f"Component hash verification failed for {t['name']}"
                f_dst.seek(c['offset'])
                f_dst.write(data)

    print(f"Successfully extracted {len(final_tensors)} tensors into {dst_path}!")


def main():
    parser = argparse.ArgumentParser(description="Extract MTP overlay from strixw")
    parser.add_argument("src", help="Path to source weights.strixw")
    parser.add_argument("dst", help="Path to output overlay .strixw")
    parser.add_argument("--all-mtp", action="store_true", help="Extract all 24 MTP tensors including MoE experts")
    parser.add_argument("--include-heads", action="store_true", help="Also extract embed_tokens and lm_head")
    args = parser.parse_args()

    extract_overlay(args.src, args.dst, all_mtp=args.all_mtp, include_heads=args.include_heads)


if __name__ == "__main__":
    main()
