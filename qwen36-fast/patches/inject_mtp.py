#!/usr/bin/env python3
"""Inject MTP tensors into an existing Qwen3.6 GGUF to produce a combined file."""
import sys, os, json, struct
sys.path.insert(0, '/home/everlier/code/mlm/qwen36-fast/deps/llama.cpp/gguf-py')
import gguf
import numpy as np

SRC_GGUF = '/home/everlier/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-UD-Q2_K_XL.gguf'
DST_GGUF = '/home/everlier/code/mlm/qwen36-fast/build-artifacts/qwen36-27b-mtp-merged.gguf'
SHARDS = [
    '/tmp/qwen36-mtp-shards/model-00013-of-00015.safetensors',
    '/tmp/qwen36-mtp-shards/model-00015-of-00015.safetensors',
]

os.makedirs(os.path.dirname(DST_GGUF), exist_ok=True)

# Read source
print(f"Reading source GGUF: {SRC_GGUF}")
reader = gguf.GGUFReader(SRC_GGUF)

# Check arch
arch_field = reader.get_field('general.architecture')
arch_str = bytes(arch_field.parts[-1]).decode()
print(f"  arch: {arch_str}")
block_count = int(reader.get_field(f'{arch_str}.block_count').parts[-1][0])
print(f"  block_count: {block_count}")

# Set up writer with QWEN35 arch, 65 layers
NEW_BLOCK_COUNT = block_count + 1
MTP_LAYER_IDX = block_count  # 64

writer = gguf.GGUFWriter(DST_GGUF, arch_str)

# Copy all KV pairs from source, but override block_count and add nextn_predict_layers
for field in reader.fields.values():
    name = field.name
    if name == f'{arch_str}.block_count':
        writer.add_block_count(NEW_BLOCK_COUNT)
        continue
    if name in ('GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'):
        continue  # auto-written
    # Parse the field
    types = field.types
    if len(types) == 0:
        continue
    value = None
    try:
        if types[0] == gguf.GGUFValueType.STRING:
            value = bytes(field.parts[-1]).decode('utf-8', errors='replace')
            writer.add_string(name, value)
        elif types[0] == gguf.GGUFValueType.ARRAY:
            inner = types[1]
            if inner == gguf.GGUFValueType.STRING:
                # Array of strings
                strs = []
                # parts: [length, type, length_each, data_each...]
                # simpler: use field.data (indices into parts)
                for idx in field.data:
                    strs.append(bytes(field.parts[idx]).decode('utf-8', errors='replace'))
                writer.add_array(name, strs)
            else:
                # Numeric array
                arr = np.concatenate([field.parts[i] for i in field.data]).tolist()
                writer.add_array(name, arr)
        else:
            # Scalar numeric/bool
            value = field.parts[-1]
            if hasattr(value, 'tolist'):
                value = value.tolist()
                if isinstance(value, list):
                    value = value[0]
            # Pick correct add method per type
            t = types[0]
            if t == gguf.GGUFValueType.UINT32:
                writer.add_uint32(name, value)
            elif t == gguf.GGUFValueType.INT32:
                writer.add_int32(name, value)
            elif t == gguf.GGUFValueType.UINT64:
                writer.add_uint64(name, value)
            elif t == gguf.GGUFValueType.INT64:
                writer.add_int64(name, value)
            elif t == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(name, value)
            elif t == gguf.GGUFValueType.FLOAT64:
                writer.add_float64(name, value)
            elif t == gguf.GGUFValueType.BOOL:
                writer.add_bool(name, value)
            elif t == gguf.GGUFValueType.UINT16:
                writer.add_uint16(name, value)
            elif t == gguf.GGUFValueType.INT16:
                writer.add_int16(name, value)
            elif t == gguf.GGUFValueType.UINT8:
                writer.add_uint8(name, value)
            elif t == gguf.GGUFValueType.INT8:
                writer.add_int8(name, value)
    except Exception as e:
        print(f"  WARN: couldn't copy field {name}: {e}")

# Add nextn_predict_layers
writer.add_uint32(f'{arch_str}.nextn_predict_layers', 1)
print(f"Added nextn_predict_layers=1")

# Copy all existing tensors
print(f"Copying {len(reader.tensors)} tensors from source...")
for t in reader.tensors:
    writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

# Now read MTP tensors from safetensors
print("Reading MTP tensors from safetensors shards...")
def bf16_to_fp32(data_bytes, shape):
    """Convert bytes of bf16 to fp32 numpy array."""
    u16 = np.frombuffer(data_bytes, dtype=np.uint16).copy()
    # shift left 16 bits to get fp32 bit pattern
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32).reshape(shape)
    return f32

# Remap MTP tensor name -> GGUF tensor name (matching PR 20700 converter logic)
def mtp_name_to_gguf(name, mtp_layer):
    """name is like 'mtp.layers.0.self_attn.q_proj.weight' or 'mtp.fc.weight'"""
    b = mtp_layer  # absolute block idx (64)
    # Shared MTP weights
    mapping = {
        'mtp.fc.weight': f'blk.{b}.nextn.eh_proj.weight',
        'mtp.pre_fc_norm_embedding.weight': f'blk.{b}.nextn.enorm.weight',
        'mtp.pre_fc_norm_hidden.weight': f'blk.{b}.nextn.hnorm.weight',
        'mtp.norm.weight': f'blk.{b}.nextn.shared_head_norm.weight',
    }
    if name in mapping:
        return mapping[name]
    # Per-layer MTP block tensors — match existing backbone layer naming
    if name.startswith('mtp.layers.0.'):
        rest = name[len('mtp.layers.0.'):]
        # Standard transformer naming pattern
        r = rest
        r = r.replace('self_attn.q_proj.weight', 'attn_q.weight')
        r = r.replace('self_attn.k_proj.weight', 'attn_k.weight')
        r = r.replace('self_attn.v_proj.weight', 'attn_v.weight')
        r = r.replace('self_attn.o_proj.weight', 'attn_output.weight')
        r = r.replace('self_attn.q_norm.weight', 'attn_q_norm.weight')
        r = r.replace('self_attn.k_norm.weight', 'attn_k_norm.weight')
        r = r.replace('input_layernorm.weight', 'attn_norm.weight')
        r = r.replace('post_attention_layernorm.weight', 'post_attention_norm.weight')
        r = r.replace('mlp.gate_proj.weight', 'ffn_gate.weight')
        r = r.replace('mlp.down_proj.weight', 'ffn_down.weight')
        r = r.replace('mlp.up_proj.weight', 'ffn_up.weight')
        return f'blk.{b}.{r}'
    return None

mtp_tensors_written = 0
for shard in SHARDS:
    print(f"  shard: {shard}")
    with open(shard, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_len)
        data_offset = 8 + header_len
        header = json.loads(header_bytes)
    for tname, meta in header.items():
        if not tname.startswith('mtp'):
            continue
        if meta.get('dtype') != 'BF16':
            print(f"    SKIP {tname}: unexpected dtype {meta.get('dtype')}")
            continue
        gguf_name = mtp_name_to_gguf(tname, MTP_LAYER_IDX)
        if not gguf_name:
            print(f"    SKIP unknown tensor name: {tname}")
            continue
        shape = tuple(meta['shape'])
        start, end = meta['data_offsets']
        size = end - start
        # Read tensor data
        with open(shard, 'rb') as f:
            f.seek(data_offset + start)
            raw = f.read(size)
        data_fp32 = bf16_to_fp32(raw, shape)
        # Norm weights (1D small tensors) MUST be F32 to match backbone convention
        # (otherwise ggml_cuda_op_mul fails with alignment assertion against F32 activations).
        # Weights (2D matmul operands) are written as F16 to save space.
        is_norm = ('norm' in gguf_name)
        if is_norm:
            writer.add_tensor(gguf_name, data_fp32)
            mtp_tensors_written += 1
            print(f"    {tname} -> {gguf_name} {shape} F32 (norm)")
        else:
            data_f16 = data_fp32.astype(np.float16)
            writer.add_tensor(gguf_name, data_f16)
            mtp_tensors_written += 1
            print(f"    {tname} -> {gguf_name} {shape} F16")

print(f"MTP tensors written: {mtp_tensors_written}")
print(f"Writing GGUF to: {DST_GGUF}")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()
print("DONE")
print(f"Final file: {os.path.getsize(DST_GGUF) / (1024**3):.2f} GiB")
