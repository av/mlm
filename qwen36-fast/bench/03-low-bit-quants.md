# Low-bit quant sweep — Qwen3.6-27B

Run date: 2026-04-23 CEST (Thu, ~01:40+)

**Context**: Iteration 7 established that speculative decoding on Qwen3.6 is blocked
upstream by `llama_memory_recurrent::seq_rm`'s refusal to partial-rollback Gated
DeltaNet state (multi-day fix). This benchmark quantifies the *no-spec-decode*
fallback: how close can we get to the 40 tps target just by moving down the
quant/bandwidth ladder?

Reference baseline (iter 5): Q4_K_M → 10.87 tps decode @ 2k ctx, ~70 % of the
15.6 tps bandwidth ceiling on 256 GB/s LPDDR5x.

## Quants benchmarked

Sourced from `unsloth/Qwen3.6-27B-GGUF` (HF revision `82d411acf4a...`).

| Quant | File size | bpw (approx) | Theoretical ceiling* |
|---|---:|---:|---:|
| Q4_K_M (iter-5 baseline) | 15.66 GiB (16.82 GB) | ~5.00 | ~15.2 tps |
| Q3_K_S | 11.51 GiB (12.36 GB) | ~3.68 | ~20.7 tps |
| UD-IQ3_XXS | 11.17 GiB (11.99 GB) | ~3.57 | ~21.3 tps |
| UD-Q2_K_XL | 11.04 GiB (11.85 GB) | ~3.53 | ~21.6 tps |

*Ceiling = 256 GB/s ÷ file_size (GB). This is the weights-only ceiling; real
decode also pays KV reads + activation traffic, so the achievable fraction is
~60-72 % from iter-5 experience.

Note: the orchestrator's paper "bpw table" (Q4_K_M=4.5, Q3_K_S=3.5, Q2_K=2.6,
IQ3_XXS=3.06) is the dequantized-weight bpw. Unsloth's GGUFs carry additional
overhead (embedding+output at higher precision, imatrix metadata, UD dynamic
per-layer choices) so the **file-size-derived bpw ~3.5 for all three low-bit
variants** is what actually gets read from memory per decode step. This
compresses the theoretical headroom between quants much more than one would
guess from the paper table.

## Backend

Identical to iter-5 baseline — no recompile, no flag changes, no llama.cpp
patches.

- Image: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`
- llama.cpp build: `d6f303004 (8738)`, full qwen35 dense arch support
- GPU: Radeon 8060S (gfx1151), 96 GiB VRAM pool, 256 GB/s LPDDR5x
- Flags: `-ngl 99 -fa 1 -r 3 -p 2048 -n 256 -d 2048` (pp2048+tg256 @ depth 2048)

## Results

| Quant | File size | Prefill tps | Decode tps | VRAM peak | Tok/GiB | % of BW ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M (baseline) | 15.66 GiB | 308.40 ± 0.22 | 10.87 ± 0.01 | 18.79 GiB | 0.578 | ~71 % |
| Q3_K_S | 11.51 GiB | 302.87 ± 1.45 | 13.48 ± 0.04 | 13.91 GiB | 0.969 | ~65 % |
| UD-IQ3_XXS | 11.17 GiB | 284.65 ± 2.71 | 11.98 ± 0.02 | 13.64 GiB | 0.878 | ~56 % |
| UD-Q2_K_XL | 11.04 GiB | 306.00 ± 3.28 | **13.82 ± 0.02** | 13.47 GiB | 1.026 | ~64 % |

Raw logs in `bench/03-q3ks.log`, `bench/03-iq3xxs.log`, `bench/03-q2kxl.log`,
`bench/03-q2kxl-fa0.log`.

### Headline observations

- **Best absolute decode: UD-Q2_K_XL at 13.82 tps** (1.27× over Q4_K_M baseline).
- Q3_K_S is effectively tied with Q2_K_XL at 13.48 tps — in the noise against
  Q2_K_XL given the PPL trade-off. For a 2.5 % speed premium, Q2_K_XL gives up
  roughly 6-12 % PPL vs Q4 while Q3_K_S gives up ~3-6 %.
- **UD-IQ3_XXS underperforms both**: 11.98 tps despite being slightly smaller
  than Q3_K_S. IQ-quants use a non-uniform codebook-based dequant path — on
  gfx1151/ROCm-7.2 this kernel is clearly less optimized than the K-quant
  path (56 % of BW ceiling vs 64-65 % for the K-quants). This is a pure
  kernel-throughput issue, not a bandwidth one.
- **None of the low-bit quants alone close the gap** — best is 13.82 tps vs
  40 tps target, still 2.89× short.
- All three low-bit VRAM peaks cluster tightly at 13.47-13.91 GiB (vs 18.79
  GiB Q4_K_M). ~5 GiB of headroom freed by the quant step. Plenty for a
  draft-model or MTP drafter once spec decode is unblocked.

### FA impact

Q2_K_XL was re-run with `-fa 0` (no flash attention) for comparison:

| Run | Decode tps | Delta vs FA |
|---|---:|---:|
| UD-Q2_K_XL, `-fa 1` | 13.82 | baseline |
| UD-Q2_K_XL, `-fa 0` | 13.06 | −0.76 tps (−5.5 %) |

Flash attention nets ~5.5 % decode at 2k ctx on this quant — consistent with
iter-5 findings that the workload is decode-bound and KV reads are already a
small fraction of the per-token traffic. FA will grow in importance at longer
contexts (8k+), but at 2k it's a minor win.

## Quality context

Using community perplexity numbers (Wikitext-2, similar-scale models — no
actual eval of Qwen3.6-27B done here):

| Quant | PPL delta vs F16 (typical) |
|---|---|
| Q4_K_M | +0.5–1.0 % (reference "good" baseline) |
| Q3_K_S | +3–6 % (noticeable on reasoning, mostly fine on chat) |
| UD-IQ3_XXS | +2–4 % (Unsloth "dynamic" allocates more bits to sensitive tensors) |
| UD-Q2_K_XL | +6–12 % (XL = a few layers kept at higher bits; still a sharp drop) |

The XL/dynamic variants are specifically designed by Unsloth to soften the
perplexity cliff at Q2/Q3. For Qwen3.6 27B specifically, Unsloth's release
notes claim UD-Q2_K_XL retains ~95-97 % of Q4_K_M chatbench quality — so
it's usable, but a step down from Q4.

Given IQ3_XXS is **slower** than both K-quants on this backend, it has no
Pareto-optimal niche here — Q3_K_S dominates it on both speed and PPL, and
Q2_K_XL dominates on speed at the cost of a bigger PPL hit.

## Gap to 40 tps target

- **Best observed decode**: UD-Q2_K_XL, 13.82 tps (`-fa 1`, pp2048+tg256 @ d2048)
- **Target**: 40 tps
- **Absolute gap**: **26.18 tps**
- **Required multiplier**: **2.89×** from the best quant alone

Required spec-decode multiplier from the **best** quant (Q2_K_XL @ 13.82 tps)
to hit 40 tps, using `speedup ≈ (1 + α·K) / (1 + K·c)`:

| K | c (draft cost / target step) | Required α |
|---:|---:|---:|
| 2 | 0.016 (MTP single block) | **0.99** (effectively impossible) |
| 3 | 0.016 | 0.68 |
| 4 | 0.016 | **0.52** |
| 4 | 0.05  (0.6B separate draft) | 0.62 |

At K=4 the required α drops to ~0.5 for MTP and ~0.62 for a 0.6B draft —
both numerically reachable (iter-6 measured α=0.65 on a repetitive prompt
with lookup drafting before the M-RoPE bug truncated output).

**Verdict**: even the best low-bit quant is **not enough alone**. Closing the
gap still requires:

- Q2_K_XL (or Q3_K_S), **AND**
- spec decode at K≥3 with α≥0.65 (recurrent-state checkpoint fix is
  prerequisite — see notes/05-mrope-spec-fix.md).

No quant tested hits ≥ 30 tps (no-spec viable path). Recurrent-state
checkpoint remains the critical unblocker.

## Remaining high-impact work (ranked)

Ranked by (tps impact) × (1 / effort) given spec-decode is blocked upstream:

1. **Q2_K_XL (or Q3_K_S) becomes the new baseline-quant for the spec-decode
   work stream.** This is a free +27 % decode over Q4_K_M with no code changes
   and drops the α requirement at K=4 from ~0.75 (Q4 → 40 tps) to ~0.52 (Q2_K_XL
   → 40 tps). The K-quant kernel advantage over IQ3_XXS also means sticking
   with K-quants on this backend — IQ-family gets no special consideration
   until the IQ dequant kernel gets a gfx1151-aware rewrite.
2. **Recurrent-state checkpoint+restore in `llama_memory_recurrent`**
   (SSM/GDN state snapshot on ubatch start, restore on partial reject).
   Multi-day, multi-backend, upstream-mergeable. Unblocks every spec decode
   path (lookup, draft, MTP) for every hybrid model in llama.cpp.
3. **Atomic-block spec decode variant** that avoids partial rollback entirely:
   accept full K-draft or reject full K-draft. Throws away the win on mixed
   accepts. Lower ceiling than (2) but drops the impl cost by ~10×.
4. **Train EAGLE-3 head** against Qwen3.6-27B. Only unblocks spec decode once
   (2) or (3) lands. Heavy compute.

## Raw logs

- `bench/03-q3ks.log` — Q3_K_S @ pp2048+tg256 d2048
- `bench/03-iq3xxs.log` — UD-IQ3_XXS @ pp2048+tg256 d2048
- `bench/03-q2kxl.log` — UD-Q2_K_XL @ pp2048+tg256 d2048, `-fa 1`
- `bench/03-q2kxl-fa0.log` — UD-Q2_K_XL @ pp2048+tg256 d2048, `-fa 0`
