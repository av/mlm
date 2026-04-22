# 02 — MTP head inspection + draft model compatibility

Iteration 2 — Thu 2026-04-23, early AM CEST.

## Goal

Determine whether:
1. Unsloth's Qwen3.6 GGUF conversions preserve the MTP (multi-token-prediction) head
   that was reportedly trained as part of Qwen3.6. If yes, we get a cheap drafter "for
   free" and can skip drafter training.
2. A small Qwen3 dense GGUF (Qwen3-0.6B) is a viable draft model for Qwen3.6-27B under
   vanilla llama.cpp speculative decoding (needs matching tokenizer vocab).

## Methodology

- 27B GGUF not yet cached — started a background download (see section "Downloads" below).
- Used Unsloth's Qwen3.6-35B-A3B-GGUF (already cached from prior work, same model family
  and same Unsloth conversion pipeline) as a proxy. If the MTP head is stripped from
  35B-A3B's GGUF, it's almost certainly stripped from the 27B GGUF too, because Unsloth
  uses one conversion script per Qwen3.6 family release.
- `pip install --user gguf` (0.18.0) then `python -m gguf.scripts.gguf_dump <file>`.

## 1. Unsloth Qwen3.6-35B-A3B-GGUF tensor inventory

File: `/home/everlier/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

Metadata highlights:

- `general.architecture = 'qwen35moe'`
- `qwen35moe.block_count = 40`
- `qwen35moe.embedding_length = 2048`
- `qwen35moe.attention.head_count = 16`, `head_count_kv = 2`
- `qwen35moe.expert_count = 256`, `expert_used_count = 8`
- `qwen35moe.ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.group_count = 16`,
  `ssm.time_step_rank = 32`, `ssm.inner_size = 4096` (Gated DeltaNet layers)
- `qwen35moe.full_attention_interval = 4` (confirms 3:1 DeltaNet:attention ratio)
- `tokenizer.ggml.tokens` length = **248,320** (Qwen3.6 expanded vocab)
- `tokenizer.ggml.eos_token_id = 248046`, bos = 248044
- `tokenizer.ggml.pre = 'qwen35'`
- Total tensor count: 733

### Non-block tensors (only 3)

```
  1:  508559360 | 2048, 248320, 1, 1 | Q8_0 | output.weight
  2:       2048 | 2048, 1, 1, 1      | F32  | output_norm.weight
  3:  508559360 | 2048, 248320, 1, 1 | Q8_0 | token_embd.weight
```

### Unique per-block tensor names (after stripping `blk.N.` prefix)

```
attn_gate.weight
attn_k_norm.weight
attn_k.weight
attn_norm.weight
attn_output.weight
attn_qkv.weight
attn_q_norm.weight
attn_q.weight
attn_v.weight
ffn_down_exps.weight
ffn_down_shexp.weight
ffn_gate_exps.weight
ffn_gate_inp_shexp.weight
ffn_gate_inp.weight
ffn_gate_shexp.weight
ffn_up_exps.weight
ffn_up_shexp.weight
post_attention_norm.weight
ssm_a
ssm_alpha.weight
ssm_beta.weight
ssm_conv1d.weight
ssm_dt.bias
ssm_norm.weight
ssm_out.weight
```

All standard transformer / MoE / SSM tensors. Grepping the full tensor list for
`mtp`, `nextn`, `draft`, `medusa`, `eagle` returns **zero matches**.

### Interpretation

**MTP head is NOT preserved** in Unsloth's Qwen3.6-35B-A3B-GGUF conversion. The
released GGUF is the "main model" only: token_embd → 40 hybrid blocks → output_norm
→ output (LM head). No extra transformer layer or LM head pointing to a
shifted-token objective.

Implications for the 27B:

- Very high prior that the 27B Unsloth GGUF is also MTP-stripped (same conversion
  pipeline, same model family, same upstream HF `architectures` field which llama.cpp's
  converter does not recognize MTP for). We will still verify once the 27B download
  completes.
- If upstream `Qwen/Qwen3.6-27B` HF safetensors ship an MTP head, we would need to
  either (a) patch llama.cpp's `convert_hf_to_gguf.py` to carry it across and
  register a new MTP inference path, or (b) skip MTP and either use an external dense
  draft or train an EAGLE-3 / Medusa head ourselves.
- Plan-B (EAGLE-3 head training / vanilla draft-model spec decode) must be assumed
  the default path from here. MTP "free drafter" is almost certainly not available
  via llama.cpp on this stack.

## 2. Draft-model compatibility: Qwen3-0.6B vs Qwen3.6-27B

File: `/home/everlier/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf`
(~397 MB, Q4_K_M, downloaded this iteration)

Metadata highlights:

- `general.architecture = 'qwen3'`
- `qwen3.block_count = 28`
- `qwen3.embedding_length = 1024`
- `qwen3.attention.head_count = 16`, `head_count_kv = 8`
- `tokenizer.ggml.tokens` length = **151,936**
- `tokenizer.ggml.eos_token_id = 151645`

### Compatibility verdict: INCOMPATIBLE as a draft for Qwen3.6-27B

Vocab mismatch: 151,936 (Qwen3) vs 248,320 (Qwen3.6). Special token IDs differ too
(eos 151645 vs 248046). llama.cpp spec-decode requires matching vocabs (it uses the
draft's sampled token IDs directly against the target's vocab, and refuses to launch
if `n_vocab` differs). Even a permissive mode would produce garbage.

### Alternatives considered

- `Qwen/Qwen3.6-0.6B` or `Qwen/Qwen3.6-1.7B` — **not released** as of 2026-04-22 per
  quick recollection of the Qwen3.6 launch lineup (only 27B dense + 35B-A3B MoE +
  122B-A10B MoE were first-wave). Need to confirm by a HF search in a later iteration.
- `Qwen/Qwen3.5-*` small — Qwen3.5 used a different vocab from Qwen3.6 per the
  `qwen35moe` arch tag here vs Qwen3.6 arch naming in 27B; specifically the 248k
  vocab may or may not be shared. Need to confirm after 27B download lands.
- Self-drafting via n-gram / prompt lookup (llama.cpp's `--draft-max` with lookup
  cache) — vocab-independent, works unconditionally, gives modest 1.2-1.5× on repetitive
  text. Cheap fallback.
- Training a ~0.5-1B EAGLE-3 head against the 27B target — highest ceiling but
  requires an H100-class GPU for a few hours, which we don't have on this box.
- Layer-skip / self-speculative (same model, run only every Nth layer for draft) —
  available in llama.cpp via `--cache-type-k q8_0 --draft-max ...`? Need to verify
  flag surface. Would be vocab-trivially-compatible and is worth trying.

Recommended fallback ordering once 27B GGUF is local:

1. Prompt-lookup / n-gram draft (`--lookup-cache-dynamic`) — zero cost to try.
2. Check Hugging Face for `Qwen3.6-*` <=2B dense; if any exists with 248k vocab, use it.
3. Self-speculative / layer-skip in llama.cpp if supported on ROCm build.
4. Train an EAGLE-3 head.

## 3. Downloads in flight

### Qwen3.6-27B-GGUF Q4_K_M (PRIMARY TARGET)

- Started: 2026-04-23 00:26 CEST.
- Background task ID: **bg361bq6z** (Claude Code background), underlying
  `hf download` PID **2180645** (parent bash PID 2180568).
- Log: `/tmp/qwen36-downloads/27b.log`
- Target blob: `/home/everlier/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/blobs/5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0.incomplete`
- Full size: 16,588,606,287 bytes (~16.6 GB preallocated).
- Progress at ~1min in: ~4.2 GB downloaded. No HF_TOKEN set (anonymous rate limit).
- Snapshot dir (will appear once done): `/home/everlier/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/<hash>/Qwen3.6-27B-*Q4_K_M*.gguf`

### Qwen3-0.6B-GGUF Q4_K_M (potential draft — confirmed incompatible)

- Completed this iteration. Kept on disk for reference / self-speculative decoding
  experiments (any Qwen3 dense as a sanity-check target).

## 4. Next high-impact steps

- Wait for 27B download. Dump its tensors too and confirm the same MTP-stripped
  picture (expected).
- Verify 27B vocab size + tokenizer model string match 35B-A3B (both should be
  qwen35 pre / 248320 tokens) — this governs whether a future Qwen3.6-small draft
  could in principle be used.
- HF search for any released Qwen3.6 small-dense variant (<=2B).
- Once 27B is cached, wire it into Harbor's llama.cpp (ROCm 7.2 image, gfx1151) and
  run the baseline decode benchmark (tps short + long ctx) that Phase 1 requires.
- In parallel, start an n-gram / prompt-lookup spec-decode experiment (vocab-free).
