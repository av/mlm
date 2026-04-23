# Iter-15: PR #20700 MTP spec-decode — v2 after broadcast fix

- Target model: Qwen3.6-27B UD-Q2_K_XL (11.83 GiB backbone) + MTP layer 64 merged
- Build: llama.cpp master @ 0d0764dfd + PR #20700 applied (branch `pr20700-on-master`)
- Binary host: kyuz0/amd-strix-halo-toolboxes:rocm-7.2 (runtime) + /bld/bin from custom builder image
- Hardware: Strix Halo gfx1151, 96 GB VRAM pool

## Fix applied: (a) — F32 norms in MTP layer

Root cause of iter-14 `GGML_ASSERT(nb10 % sizeof(src1_t) == 0)`: backbone norm tensors are F32 across
all 64 layers; MTP layer 64 norms were injected as F16 (`attn_norm`, `post_attention_norm`,
`attn_q_norm`, `attn_k_norm`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`).
Internal `ggml_cuda_op_mul` in `build_norm → ggml_mul(activations, norm_weight)` tripped the
alignment assertion with F32×F16 mixed-dtype elementwise MUL on ROCm.

Minimal fix in `qwen36-fast/patches/inject_mtp.py`: write norm tensors (detected by
`'norm' in name`) as F32 while keeping matmul operands (`eh_proj`, `attn_q/k/v`, `attn_output`,
`ffn_up/gate/down`) as F16. Regenerated GGUF in ~6s. No build changes required.

Result: model loads, warmup passes, decode works.

## Results — all runs at temperature=0.0, greedy, prompt_code.txt (84 input tokens, 256 predict)

| Config                                  | tps     | α        | notes                                      |
|-----------------------------------------|---------|----------|--------------------------------------------|
| Plain (spec=none, merged GGUF)          | **11.91** | n/a    | MTP head loaded but bypassed               |
| llama-bench plain (merged GGUF, tg64)    | 12.06   | n/a      | sanity check                               |
| MTP K=1 (`--draft-max 1`)                | **7.80**  | 1.000    | draft_n=127 / accepted=127                 |
| MTP K=2 (`--draft-max 2`)                | 7.76      | 1.000    | identical to K=1 — PR forces n_max=1       |
| (reference) iter-13 lookup spec, K=4    | **30.05** | 0.653   | best real number on this machine tonight   |
| (reference) Q2_K_XL baseline (non-MTP)  | 13.82     | n/a     | from iter-8                                |

## Interpretation

**MTP K=1 is a net SLOWDOWN** on Qwen3.6-27B via PR #20700 on ROCm/Strix Halo: 7.80 tps vs
11.91 tps plain = 0.65× — a 35% **regression**.

Why it's slow despite α=100%:
- K=1 MTP cascade: each decode step does (target 64-layer forward) + (MTP 1-layer forward) +
  (verify of the previous draft in a 2-token batch). Even when the draft always matches, the
  target forward is now a 2-token ubatch instead of 1-token, and the MTP head adds ~45ms of
  work per step (65th layer compute + RMS norms + separate 12K-head attn + 32K-vocab matmul).
- Per-token time plain = 83 ms; per-token time K=1 = 128 ms. Verify overhead + MTP compute =
  ~55% tax.
- Plain decode on Strix Halo is already running at 70% of memory bandwidth ceiling (iter-5);
  there's little idle bandwidth for the MTP layer to "hide behind", so its cost is purely additive.

**Why K=2/3 don't help**: PR #20700 hardcodes `params_base.speculative.n_max = 1` in the MTP
branch of `tools/server/server.cpp:1309`. `--draft-max 2` and higher are silently capped.
True K>1 cascade (recursive MTP) isn't implemented in this PR.

**FastMTP vocab trim is in PR but helps little**: the PR caps `mtp_vocab_size = 32768` to shrink
the draft lm_head matmul. On our hardware this saves ~8ms/step but the rest of the MTP forward
(attn+FFN through a 5120-dim 17408-hidden layer) dominates.

## Did we hit 40 tps? NO

| Metric                                | Value  |
|---------------------------------------|--------|
| **Best absolute tps tonight**         | **30.05** (iter-13 lookup, K=4) |
| MTP path best                         | 7.80 (WORSE than baseline)      |
| Target                                | 40.00  |
| Gap                                   | -10 tps (-25%)                  |

## What remains

1. **PR #20700 MTP is not viable as a speed win for Qwen3.6-27B on bandwidth-bound hardware.**
   The architecture would need K=2+ cascade to amortize MTP-layer cost, which PR doesn't support.
2. Lookup spec decode (iter-13) remains the best path at 30.05 tps.
3. To close the gap to 40 tps without waiting for PR #20700 to mature:
   - Try lookup with a larger n-gram corpus preseeded (e.g. the current conversation's own
     output fed back as a cache). Acceptance could rise from 0.65 to ~0.75.
   - Or try draft-model spec decode now that recurrent-rollback works: a trained or stitched
     Qwen3.6-0.5B-style drafter with matching 248k vocab. No such drafter exists today but an
     EAGLE-3 head trained on 27B hidden states (PR #21437) would be the right path.
4. If PR #20700 is revisited: remove the `n_max=1` hardcode, implement true K>1 recursion in
   `build_mtp_head`, then re-benchmark. The tensor-graph for recursive MTP already exists
   (see `src/models/qwen35.cpp:536-539` — but the server-side cascade plumbing caps it at 1).

## Fix-path recap (for next iteration)

- (a) F32 norms — **worked**; kept matmul tensors F16 to save space. Final GGUF still 11.83 GiB.
- (b) Q8_0 for MTP — not needed, (a) fixed it.
- (c) binbcast.cu change — not needed.
- (d) Gate MTP in warmup — not the bug; warmup failure was the dtype mismatch, not the MTP
  layer itself. Model now passes warmup fine.

## Files

- `build-artifacts/qwen36-27b-mtp-merged.gguf` (regenerated, 11.83 GiB, not in git)
- `patches/inject_mtp.py` (norm-tensor F32 fix)
- `bench/08-mtp-smoke.log` (server startup / model load, trimmed)
- `bench/08-mtp-bench.log` (llama-bench plain tg64 = 12.06 tps)
- `bench/08-mtp-K1.json` (MTP K=1 completion response + timings)
- `bench/08-mtp-K2.json` (MTP K=2 completion response — same as K=1)
- `bench/08-mtp-nospec.json` (plain spec=none on merged model)
- `bench/08-mtp-nospec.log` (server log for nospec run)
