# Iter-17: MTP K>1 cascade — architectural dead-end confirmed

- Target model: `qwen36-27b-mtp-merged.gguf` (UD-Q2_K_XL backbone + MTP layer 64 injected, iter-15 F32-norm fix)
- Runtime: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2` with `/bld/bin` from `build-rocm/` (PR #20700 applied, iter-14 build)
- Prompt: `merge_sorted_lists` coding task (same as iter-15 — short prompt, matched generation regime)
- Flags: `--spec-type mtp --draft-max K --draft-min 1 -ngl 99 -fa on -fit off -c 4096 --no-warmup`
- Generation: `n_predict=256`, `temperature=0` (greedy)
- Server endpoint: `POST /completion`

## Hypothesis tested

Iter-15 noted "tps K=2 = same as K=1, PR #20700 hardcodes `n_max=1` at
`tools/server/server.cpp:1309`". This iteration attempts to REMOVE that
hardcode and measure real K=2/3/4.

## What actually lives at that line

```cpp
// tools/server/server-context.cpp:1307 (post-merge line numbering)
backend_sampling &= !(slot.can_speculate() && task.params.speculative.n_max > 0);
```

— this is a boolean gate on backend-accelerated sampling, NOT a value cap.
`n_max` is passed through unchanged.

**The real constraint is in `common/speculative.cpp`**
`common_speculative_state_mtp::draft()` (lines 603-649):

```cpp
void draft(const common_params_speculative & params, ..., llama_tokens & result) override {
    // ... cooldown logic ...
    const float * mtp_logits = llama_get_mtp_logits(ctx_tgt);  // ONE position's logits
    // ... argmax over mtp_n_vocab ...
    result.push_back(draft_token);            // pushes EXACTLY 1 token
    GGML_UNUSED(params);                       // n_max not consulted
}
```

The function is architecturally single-token. No amount of `--draft-max N`
can make it return more than 1 token.

## Measured results

All four runs at `temperature=0`, identical prompt, identical n_predict:

| K | tps | drafts | accepted | α | drafts/step | vs 13.82 tps base | vs iter-13 lookup 30.05 | output |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 (iter-15 control) | 7.80 | 127 | 127 | 1.00 | 0.50 | 0.56× | 0.26× | coherent |
| 1 (re-run) | 7.66 | 127 | 127 | 1.00 | 0.50 | 0.55× | 0.25× | coherent |
| 2 | 7.61 | 127 | 127 | 1.00 | 0.50 | 0.55× | 0.25× | coherent, identical |
| 3 | 7.56 | 127 | 127 | 1.00 | 0.50 | 0.55× | 0.25× | coherent, identical |
| 4 | 7.64 | 127 | 127 | 1.00 | 0.50 | 0.55× | 0.25× | coherent, identical |

**Identity signature**: `draft_n=127` for all K, `α=1.00` for all K, output
token stream identical for all K. K is completely ignored by the MTP path.

## Why this is not fixable by a server patch

To produce K>1 tokens from a single MTP forward, we'd need the graph to
emit `t_logits_mtp` of shape `[n_vocab, K]` — K position logits. Currently
it emits `[n_vocab, 1]` (last-position logit only; see
`src/llama-context.cpp:1820-1835`).

Three paths considered:

1. **Chained single-token MTP micro-decodes** (N = 1 each, iterated K
   times, feeding prior draft into next call). Requires a new API
   `llama_decode_mtp_only()` that runs only the 65th layer + attention
   against the in-memory KV state, without invoking the 64-layer backbone.
   No such split exists; `llama_decode` is monolithic. ~200+ LoC to add
   and it would be fragile on hybrid (SSM + attn) models due to the
   recurrent-state entanglement at each MTP call.

2. **Re-architect `build_mtp_head` to unroll K steps inside one graph.**
   The loop at `src/models/qwen35.cpp:447-542` iterates over
   `nextn_predict_layers`, but does so to CHAIN DEEPER heads into a
   better single-token prediction (DeepSeek-MTP convention), not to
   produce multiple future tokens. Each iteration overwrites
   `res->t_logits_mtp = mtp_logits` (line 533). Even the "greedy_tokens"
   feed-forward at lines 536-539 is intra-cascade (i.e., "use the token
   predicted by layer k as input to layer k+1 for the SAME output
   position"), not true look-ahead.

   Genuine K-lookahead would need:
   - Retaining K separate `mtp_logits_k` tensors
   - Per-K argmax and embedding to feed the NEXT TOKEN position
     (not deeper into the current prediction)
   - Per-K position encoding with `inp_pos + k`
   - Per-K buffer extraction in `llama-context.cpp`
   - Per-K recurrent-state checkpoint for the Gated DeltaNet layers
     (Qwen3.6's 3:1 GDN + attention hybrid)

   Estimated ~400-600 LoC across `qwen35.cpp`, `llama-graph.h`,
   `llama-context.cpp`, `speculative.cpp`, and `llama.h`. Out of scope
   for a 60-min timebox.

3. **Replicate the single MTP layer K times in the GGUF**
   (`nextn_predict_layers = K`). The graph-level cascade runs, but still
   only writes the LAST layer's logits to `t_logits_mtp`. And even if
   intermediate logits were extracted, they'd all predict the SAME next
   position — the MTP head was trained for 1-step lookahead, not for
   shift-k prediction. Acceptance would collapse for k>0.

Additionally: the Qwen3.6-27B HF checkpoint has `mtp_num_hidden_layers: 1`
(verified iter-3, iter-14). There are no additional pretrained MTP weights
to inject; any replication would be of the same single layer.

## Did we exceed iter-13's 30.05 tps? NO

| Metric | Value |
|---|---|
| **Best tps tonight** | **30.05 (iter-13 lookup)** — unchanged |
| Best MTP tps this iteration | 7.66 (K=1,2,3,4 all tied) |
| MTP regression vs baseline | −45% |
| MTP regression vs iter-13 lookup | −75% |

**Bandwidth economics**: iter-15's diagnosis holds. On 256 GB/s Strix Halo,
the cost of running an extra MTP forward per step (≈45ms) exceeds the
benefit of saving one decode per accepted draft (≈83ms × 1 = 83ms ÷ 2 = 42ms
amortized at α=1.00, K=1). The budget ratio is roughly 1.0:1.1, regressive
even at perfect acceptance.

For MTP to win on this hardware we'd need either:
- K≥3 with α≥0.70 (not achievable — architectural cap above)
- A much cheaper MTP head (fewer params, smaller vocab matmul)
- Much higher memory bandwidth (>1 TB/s, where the fixed MTP cost gets
  amortized under the backbone read)

## Interpretation

1. **The "n_max=1 hardcode" in iter-15's writeup was a mis-attribution.**
   No such hardcode exists. The actual constraint is structural.

2. **PR #20700 MTP is a single-token drafter by design.** Even on
   CUDA/datacenter GPUs where the PR author reports wins, the "win" is
   K=1 speedup (1 MTP logit replacing 1 backbone forward when accepted),
   not K>1 lookahead.

3. **True multi-token MTP for Qwen3.6 requires retraining the head for
   shift-k prediction.** The released 15-tensor MTP head only supports
   1-step lookahead. Retraining would need gradients through K forward
   shifts — a big project.

4. **For Strix Halo, the path to 40 tps is not MTP.** Options:
   - EAGLE-3 head (PR #21437) — isolates draft from target forward,
     better bandwidth profile
   - Lookup with richer/longer dynamic n-gram cache (iter-16 pushed to
     31 tps; headroom is limited by prompt diversity)
   - Q2 + aggressive quant of less-sensitive tensors (iter-8 showed
     Q2_K_XL already essentially Q-saturated at 13.82 tps)

## Files

- `bench/10-mtp-cascade-K1.json` through `K4.json` — raw server `/completion` responses
- `bench/10-mtp-cascade-K*.log` — statistics and draft metrics from server log
- `patches/llamacpp-unlock-mtp-k.patch` — empty-by-design patch file documenting
  why removal of the "cap" is a no-op (the cap is architectural, not a hardcode)

## Commands

Reproducible via:

```bash
for K in 1 2 3 4; do
  docker run -d --name mtp-test ... \
    --spec-type mtp --draft-max $K --draft-min 1 ...
  curl -sX POST http://localhost:8080/completion \
    -d '{"prompt":"<merge_sorted_lists task>","n_predict":256,"temperature":0}'
  docker rm -f mtp-test
done
```

Server image: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`.
Binaries: `qwen36-fast/deps/llama.cpp/build-rocm/bin/` (iter-14 build,
PR #20700 applied).
GGUF: `build-artifacts/qwen36-27b-mtp-merged.gguf` (11.83 GiB,
iter-15 F32-norm fix).
