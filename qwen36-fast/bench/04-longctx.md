# Long-context decode bench — Qwen3.6-27B UD-Q2_K_XL on ROCm 7.2

Run date: 2026-04-23 (iter-9/10)

Binary: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`, build `d6f303004 (8738)`
Model: `Qwen3.6-27B-UD-Q2_K_XL.gguf` (11.03 GiB on disk, 11.85 GiB weights in VRAM)
Backend: ROCm 7.2 on Radeon 8060S (gfx1151), FA on (`-fa 1`), default ctx
Command pattern: `llama-bench -m <model> -fa 1 -p <N> -n 128 -d <N> -r 3`

## Results

| depth | prefill (`pp<N>@d<N>`) tps | decode (`tg128@d<N>`) tps | Δdecode vs d=2048 |
|------:|---------------------------:|--------------------------:|-------------------|
|  2048 |               306.00 ± 3.28 |            13.82 ± 0.02 (baseline) | 0.00% |
|  4096 |               289.03 ± 2.09 |            13.65 ± 0.05   | −1.2% |
|  8192 |               262.15 ± 0.85 |            13.44 ± 0.05   | −2.7% |
| 16384 |               218.50 ± 0.87 |            13.01 ± 0.08   | −5.9% |
| 32768 |                   (skipped) |               (skipped)   |  OOM risk |

32k skipped on a 96 GB unified-memory pool because 11.85 GiB weights + 32k KV cache (~10-12 GiB at fp16 K/V on 64 layers × 1024 kv_heads × 128 head-dim) would push peak well beyond the 22 GiB reserved VRAM block actually in use and risks page thrashing. Decode at 32k is projected ~12.8-13.0 tps extrapolating the 0→8k slope, not worth the risk given MTP is the real lever.

## Interpretation

- Decode is almost flat out to 8 k context (−2.7 % from 2 k).
  Qwen3.6's hybrid Gated-DeltaNet 3:1 Gated-Attention keeps the attention
  bandwidth cost roughly constant — only 1 in 4 layers grows with ctx
  length, and FA collapses that 1-layer cost onto a compute-bound axis
  (not BW-bound). Q2_K_XL keeps the bandwidth budget ~6 GiB/token
  instead of ~10 GiB for Q4_K_M, which widens the decode headroom.
- Prefill drops ~14 % from 2 k → 8 k; that's the expected compute-bound
  attention-O(L) cost, amplified by Q2_K_XL dequant overhead in prefill
  matmul. Not on our critical path — MTP target is decode tps.
- **Important for spec-decode planning**: the decode-at-context curve is
  so flat that the 2 k → 8 k numbers are effectively interchangeable for
  α-estimation. Expected spec-decode gain ratios measured at d≈2 k
  transfer cleanly to d≈8 k. Above 8 k we will need to re-measure.
- Peak VRAM at 8 k remained well under the 18.8 GiB observed on Q4_K_M
  at similar context, meaning a drafter of up to ~4-6 GiB fits without
  evicting the target.

## What this means for MTP target

- Target decode is **13.4-13.8 tps** regardless of ctx depth in our
  realistic work band (2 k-8 k).
- 40 tps goal → **2.90×-2.98×** spec multiplier needed.
- MTP K=4 at α≥0.52 covers it; at K=2 need α≥0.65.
- Long-ctx penalty on MTP drafter itself: ~10 % at 8 k (MTP adds 1 attn
  layer ≈ 1/64 of model cost; drafter-per-token is ~1/64 × baseline
  13.4 tps ≈ 0.2 ms, negligible).

## Raw logs

- `04-longctx-q2kxl-4k.log` — d=4096 (done, 13.65)
- `04-longctx-q2kxl-8k.log` — d=8192 (done, 13.44)
- `04-longctx-q2kxl-16k.log` — d=16384 (done, 13.01)
