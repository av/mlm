# Drafter strategy decision — HF safetensors inspection

Date: 2026-04-22 (iteration 3)
Source: HuggingFace `Qwen/Qwen3.6-27B` metadata (no weight download).
Artifacts: `/tmp/qwen36-27b-meta/{model.safetensors.index.json,config.json,generation_config.json}`

## 1. MTP head in HF safetensors: YES

The upstream `Qwen/Qwen3.6-27B` checkpoint ships a full one-layer MTP head.
15 tensors under the `mtp.*` prefix, distinct from the backbone and from `lm_head`:

```
mtp.fc.weight                                  # fusion of (embedding, hidden) -> hidden
mtp.pre_fc_norm_embedding.weight               # RMSNorm on embedding stream
mtp.pre_fc_norm_hidden.weight                  # RMSNorm on hidden stream
mtp.layers.0.input_layernorm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.mlp.down_proj.weight
mtp.norm.weight
```

Architecturally this is the DeepSeek-style MTP layout: a single transformer
block that takes `(embedding_of_next_token_guess, last_hidden_state)`, passes
each through its own pre-norm, fuses them via a concat->linear (`mtp.fc`),
runs one self-attn + MLP block, norms the result, and reuses the main
`lm_head.weight` to produce logits for the next+1 token.

Notably this MTP block uses **plain `self_attn` (full attention)**, NOT
`linear_attn` (Gated DeltaNet). That is good news for llama.cpp porting:
the drafter only needs the full-attention kernel path that llama.cpp
already has.

The MTP head does **not** carry dedicated embeddings (`mtp_use_dedicated_embeddings: false`),
so it reuses `model.language_model.embed_tokens.weight`, and reuses
`lm_head.weight` for output. Cost is just the 15 extra tensors; roughly
equivalent to one extra transformer layer (~2-3% parameter overhead).

## 2. Config hints

`config.json` -> `text_config`:
- `"mtp_num_hidden_layers": 1`
- `"mtp_use_dedicated_embeddings": false`
- `"num_hidden_layers": 64`
- 48 `linear_attention` layers + 16 `full_attention` layers (the 3:1 hybrid).
- `"vocab_size": 248320`, `"hidden_size": 5120`, `"intermediate_size": 17408`.
- `"head_dim": 256`, `"num_attention_heads": 24`, `"num_key_value_heads": 4`.
- Linear (Gated DeltaNet) attn: `linear_num_key_heads: 16`, `linear_num_value_heads: 48`,
  key/value head dim 128, conv kernel 4.
- `"attn_output_gate": true`, `"output_gate_type": "swish"`.
- Root `architectures: ["Qwen3_5ForConditionalGeneration"]`, i.e. a VLM;
  language model lives under `model.language_model.*` and the checkpoint
  also includes a ViT (`model.visual.*`). That is why the GGUF conversion
  script needs a VLM-aware path, which may be the reason the current
  Unsloth GGUF drops the MTP head — it likely uses a text-only conversion path.

## 3. Small Qwen3.6 siblings: NONE

HF query `author=Qwen&search=Qwen3.6` returns exactly 4 models:
```
Qwen/Qwen3.6-27B
Qwen/Qwen3.6-27B-FP8
Qwen/Qwen3.6-35B-A3B
Qwen/Qwen3.6-35B-A3B-FP8
```
No 0.5B / 1.5B / 3B / 4B / 7B dense sibling with the 248,320 vocab exists.
Community releases visible in the search are all quant/finetune variants of
the 27B or 35B. So a "use a small sibling as draft" path is not available —
we cannot reuse an off-the-shelf dense Qwen3.6 as drafter.

Relevant third-party releases:
- `z-lab/Qwen3.6-35B-A3B-DFlash` — the DFlash drafter from the paper, but
  trained against the 35B-A3B MoE, not the 27B dense. Also custom_code
  (Python), not llama.cpp-compatible.
- `Jackrong/Qwen3.6-27B-GGUF` — alternative GGUF source; unlikely to
  preserve MTP either since `convert_hf_to_gguf.py` would need a patch.

## 4. 27B download status

Blob `5ed60d0af...incomplete`, 16,817,244,384 bytes on disk.
Target size from the GGUF manifest is ~16.6 GB (Q4_K_M), so this is nominally
complete, but the `.incomplete` suffix and no entries under `snapshots/*/`
indicate hf_hub did not finalize (background PID 2180645 is no longer
running; the log shows no completion marker). Not blocking for this
iteration; future iterations can either restart the pull or rename the blob
and hand-link the snapshot.

## 5. Decision: **Path A — patch `convert_hf_to_gguf.py` to preserve the MTP head**

### Rationale

- Free drafter. Already trained, already in the checkpoint, ~2-3% weight
  overhead. The alternatives all cost significant time:
  - B (n-gram / prompt-lookup): works out-of-the-box in llama.cpp but
    acceptance is typically low (~20-30%) on open-domain decoding, giving
    only a 1.2-1.4x decode speedup. Not enough to reach the 40 tps target
    from a 15-18 tps Q4 baseline.
  - C (self-speculative / layer-skip): requires llama.cpp patches anyway,
    acceptance depends on the specific skip pattern and is usually worse
    than a proper trained drafter.
  - D (train EAGLE from scratch): weeks of work, needs a calibration
    corpus, needs distillation on a machine beefier than Strix Halo.
- Drafter is full-attention only. No Gated DeltaNet kernel needed for the
  drafter path — only the backbone needs linear-attn support (which
  llama.cpp already handles for the main 27B via the existing `qwen3_5`
  arch support if it exists, or the forthcoming port).
- MTP is vocab-consistent by construction. It shares the embedding and
  `lm_head`, so the draft-model-vs-target vocab mismatch problem that
  killed Qwen3-0.6B as a drafter does not exist here.
- Single-layer head means small and fast: one transformer block decode
  per draft token + one backbone decode per verify. With typical
  acceptance rates reported for MTP (~60-75%), net decode throughput can
  plausibly 1.7-2.2x, which gets us to 30-40 tps from a 16-18 tps Q4
  baseline.
- Blast radius of the patch is small: add (a) a new tensor-name mapping
  class in `convert_hf_to_gguf.py` that recognizes `mtp.*` keys and
  emits them under a new `gguf.Keys.*.MTP_*` namespace, (b) a GGUF
  metadata bump declaring `mtp.num_layers=1`, and (c) llama.cpp
  inference changes to load the MTP block and run it as the draft
  model in the existing speculative pipeline. The conversion patch
  alone is a 1-day job; the inference patch is the heavier half and
  may take 2-3 days on this machine.

### Fallback if A proves infeasible

If llama.cpp's speculative-decode pipeline cannot be coerced into
calling a within-model drafter (it assumes a separate `gguf` file for
the draft model), the easiest workaround is to **emit a second GGUF
containing only the MTP head + a copy of the shared `token_embd` and
`output` tensors**, and run it as a standard draft-model spec decode
with a matching vocab. That preserves the "free drafter" property
without needing llama.cpp runtime changes.

## 6. Concrete next steps (for future iterations)

1. Clone llama.cpp somewhere under Harbor's workspace and read
   `convert_hf_to_gguf.py` to locate the Qwen3.6 / `qwen3_5` converter
   class. If it doesn't exist yet, this research needs a llama.cpp
   version bump first.
2. Implement the MTP-head preservation patch (emit two GGUFs: backbone
   and drafter, sharing `token_embd` + `output`).
3. Restart the Unsloth 27B GGUF download OR convert upstream HF weights
   ourselves with the patch; the latter is more valuable because it
   produces the MTP drafter in one shot.
4. Re-dump to confirm MTP tensors round-tripped.
5. Wire up spec decode via Harbor's llama.cpp server (ROCm 7.2 backend),
   measure acceptance rate and tps on a fixed prompt mix.

## 7. Files touched this iteration

- Created: `/home/everlier/code/mlm/qwen36-fast/notes/03-drafter-strategy.md` (this file)
- Fetched (outside repo): `/tmp/qwen36-27b-meta/model.safetensors.index.json`,
  `/tmp/qwen36-27b-meta/config.json`,
  `/tmp/qwen36-27b-meta/generation_config.json`.
