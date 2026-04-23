# Upstream PR / issue survey — hybrid spec decode + MTP + DFlash

Run date: 2026-04-23 CEST (iter-9)

## Bottom line

**Most of the upstream work we need already exists.** Specifically, the exact
`llama_memory_recurrent::seq_rm` partial-rollback problem diagnosed in iter-7
was fixed upstream by PR #19493, **merged 2026-04-19**. Our local llama.cpp
clone (`d6f303004 (8738)` at `deps/llama.cpp/`, upstream HEAD `0d0764d`
2026-04-22) **already contains** the checkpoint-based rollback code — verified
by grepping `common/speculative.cpp:148` for `common_speculative_checkpoint`
and `draft_create_checkpoint` / `draft_restore_checkpoint` helpers.

The iter-6 lookup / draft spec decode garbled-output test was run against a
build **from before** #19493 landed. Re-running it on the current clone, with
the correct `--spec-ckpt-num-tries N --ctx-checkpoints M` flags, should now
produce clean output. That means **the hybrid-spec-decode blocker we thought
we owned is already unblocked upstream**; our iter-7 "300-500 LoC multi-day"
estimate is moot.

What is still needed (and what we can contribute):

1. Wire MTP (the head already present in Qwen3.6 HF safetensors) into the
   llama.cpp pipeline — there's a WIP PR (#20700, Qwen3.5 MTP+FastMTP, 890 LoC,
   currently open with WIP status) that does almost exactly what we need. It
   targets **Qwen3.5**, but Qwen3.6 uses the same backbone arch and the same
   MTP tensor layout, so adapting it should be small.
2. EAGLE3 support for Qwen3.5/3.6 (#21437, builds on #18039) is actively
   iterated, multiple PRs stacked — if we want EAGLE3 we wait rather than
   re-invent.
3. DFlash is vLLM-only still on the llama.cpp side (#22105 open, experimental).

## Key PRs & issues

### THE unblocker (already merged, already in our clone)

- **[llama.cpp #19493](https://github.com/ggml-org/llama.cpp/pull/19493) —
  "server : speculative checkpointing"** — **MERGED 2026-04-19**, +420/-179 LoC
  across `common/speculative.{h,cpp}` and `tools/server/server-*.cpp`.
  Introduces `common_speculative_checkpoint`, save/restore of recurrent state
  via `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, and `--spec-ckpt-num-tries` +
  `--ctx-checkpoints` CLI flags. Author reports 96→258 tps on Qwen3-Coder-Next
  quicksort benchmark with ngram_map_k; acceptance rate 62-71%. Shipped with
  `common_speculative_is_compat` now falling back to the checkpoint path
  instead of refusing.

- [llama.cpp #16382](https://github.com/ggml-org/llama.cpp/pull/16382) —
  "implement context checkpointing for hybrid and recurrent models" — **MERGED
  2025-10-03**, +89/-74 LoC. Renamed `--swa-checkpoints` → `--ctx-checkpoints`,
  added `llama_model_is_hybrid`, generalized checkpoint support to recurrent/
  hybrid. Prerequisite for #19493.

- [llama.cpp #18391](https://github.com/ggml-org/llama.cpp/pull/18391) —
  "server : fix crash when seq_rm fails for hybrid/recurrent models" — MERGED
  2025-12-26. Defensive guard that predates the full checkpoint solution.

### Closed alternatives (so we know why the merged one won)

- [llama.cpp #20080](https://github.com/ggml-org/llama.cpp/pull/20080) —
  "speculative : enable checkpoint-based rollback for hybrid/recurrent models"
  — **CLOSED 2026-03-03** with +420/-6 LoC. Identical concept to the merged
  #19493 (save recurrent state with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`,
  restore on reject, re-decode accepted prefix). Rejected explicitly because
  "contains a large amount of AI-generated code" per ngxson. Useful reference
  for API surface and for seeing that two independent contributors converged
  on the same design.
- [llama.cpp #19670](https://github.com/ggml-org/llama.cpp/pull/19670) —
  "Allow partial success of seq_rm for hybrid memory" — OPEN, +8/-6 LoC,
  stalled 2026-03-12. Narrower attempt: make `seq_rm` return true on partial,
  combined with saving recurrent state externally. Superseded by #19493's
  cleaner full-checkpoint path.

### MTP work (WIP upstream, directly relevant to us)

- **[llama.cpp #20700](https://github.com/ggml-org/llama.cpp/pull/20700) —
  "feat: MTP support for dense Qwen 3.5 with FastMTP vocabulary trimming"** —
  **OPEN, +890/-113 LoC**, author @itigges22, last update 2026-04-21. Adds
  full MTP attention head graph, FastMTP (248K→32K vocab trim for 3.7× draft
  throughput), `--two-phase-decode` option for safer hybrid rollback, fixes
  `copy_cell` (element-count vs byte-count bug), fuzzy `seq_rm` checkpoint
  matching. Reports 82% acceptance on Qwen3.5-9B Q4_K_M at temp=0, but *net*
  throughput 28.1 vs 30 tps unaccelerated on RTX 5060 Ti 16 GB (author is
  VRAM-constrained — memory-bandwidth ceiling rather than compute).
  Author explicitly says "WIP, please help find a better solution". This is
  the PR to **build on** for Qwen3.6 — Qwen3.6 shares the `qwen35` arch and
  MTP tensor layout (confirmed in iter-3).
- [llama.cpp #15225](https://github.com/ggml-org/llama.cpp/pull/15225) —
  "server: implement GLM-style MTP" — OPEN, hot, last update 2026-01-13.
  GLM4 MTP — orthogonal architecture but establishes the MTP scaffold in the
  server state machine.

### EAGLE3 work (adjacent, not our direct target)

- [llama.cpp #18039](https://github.com/ggml-org/llama.cpp/pull/18039) —
  "[Speculative decoding] feat: add EAGLE3 speculative decoding support" —
  OPEN, hot, +1137 LoC. Framework-level EAGLE3 addition — "close collaboration
  between NVIDIA and GGML teams", 2-3× speedup claimed.
- [llama.cpp #21437](https://github.com/ggml-org/llama.cpp/pull/21437) —
  "eagle3: add qwen3.5 4B 9B 35B-A3B support" — OPEN, +1764 LoC, builds on
  #18039. Reports 1.68× speedup on Qwen3.5-9B-BF16 (9.8 → 16.5 tps) with 61%
  acceptance. The two PRs are stacked; #21437 will rebase once #18039 merges.

### DFlash (llama.cpp status: experimental only)

- [llama.cpp #22105](https://github.com/ggml-org/llama.cpp/pull/22105) —
  "[Speculative decoding] feat: add DFlash support" — OPEN, +1970 LoC, last
  update 2026-04-20. Built on top of #18039 EAGLE3 infra. Reports up to 8×
  speedup on Qwen3-8B non-thinking, 3.4× on Qwen3.5-4B; author explicitly
  documents that **hybrid targets cap DFlash speedup** due to the same
  recurrent-rollback tax (draft block verify writes recurrent state before
  acceptance; restore requires replay). Still needs `--draft-max 16`,
  `--dflash` flag wiring. Not yet merge-ready.

### Competing fix PR for hybrid SSM/MoE (also open)

- [llama.cpp #20075](https://github.com/ggml-org/llama.cpp/pull/20075) —
  "fix: speculative decoding broken on hybrid SSM/MoE (Qwen3.5 MoE)" — OPEN,
  +188/-22 LoC, last update 2026-04-15. Adds its own rolling checkpoint
  buffer depth=8 per sequence, plus `empty_cell.src` and `copy_cell` fixes
  (separate from what #19493 shipped). Reports 20.4 → 23.5-29.7 tps on
  Qwen3.5-122B-A10B with 63-89% acceptance. Testing comments (FatheredPuma81)
  report only 44% acceptance and looping bugs — fix may be incomplete or
  interacting badly with #19493 now that it's merged.

### User-impact issues (prove this is painful for others, not just us)

- [llama.cpp #20039](https://github.com/ggml-org/llama.cpp/pull/20039) —
  "Feature Request: Speculative decoding on Qwen3.5" — OPEN, points directly
  at #19493 and #20075 as candidates.
- [llama.cpp #21840](https://github.com/ggml-org/llama.cpp/issues/21840) —
  "Qwen3-Code-Next not support speculative decoding" — CLOSED by
  "#19493 has been merged".

## vLLM landscape (for comparison / leverage)

vLLM has a **much more mature stack** for Qwen3.5/3.6 hybrid + spec decode:

- [vllm #36847](https://github.com/vllm-project/vllm/pull/36847) —
  "[Feat][Spec Decode] DFlash" — **MERGED 2026-03-30**.
- [vllm #38300](https://github.com/vllm-project/vllm/pull/38300) —
  "[Speculative Decoding] Add DFlash speculators config parsing" — **MERGED
  2026-04-15**.
- [vllm #39703](https://github.com/vllm-project/vllm/pull/39703) — "[Feat]
  dflash support for ROCm" — **MERGED 2026-04-21**. ROCm path for DFlash
  landed just two days before iter-9.
- [vllm #37514](https://github.com/vllm-project/vllm/pull/37514) — "[MODEL]
  Cherry-pick: Adding Support for Qwen3.5 Models" — OPEN, +1502 LoC, wires
  Qwen3.5 + Qwen3.5-MoE with their MTP heads for vLLM V1 inference.
- [vllm #40472](https://github.com/vllm-project/vllm/pull/40472) — "[CI] Add
  MTP coverage: Qwen3.5 correctness + no-sync spec decode" — OPEN but "ready",
  confirms MTP on Qwen3.5-0.8B works end-to-end on GB200 + CUDA.
- [vllm #38200](https://github.com/vllm-project/vllm/pull/38200) — "Qwen3.5
  0325 mtp" — OPEN, needs rebase.
- [vllm #40334](https://github.com/vllm-project/vllm/pull/40334) — "[Model]
  fix(dflash): dtype mismatch in combine_hidden_states" — OPEN, active.
- [vllm #40425](https://github.com/vllm-project/vllm/pull/40425) — "[Model]
  Fix quantized DFlash Qwen3 draft support" — OPEN, active.
- [vllm #36649](https://github.com/vllm-project/vllm/pull/36649) — "[Hybrid]
  [GDN] Enable prefix caching 'all' mode for Qwen3.5/Qwen3Next" — OPEN.

**Takeaway**: if the goal were "run Qwen3.6-27B with spec decode today on
mid-tier hardware", **vLLM ROCm** is the shipping answer. The blocker for us
is Strix Halo / gfx1151 — vLLM ROCm is only just getting functional for this
GPU family (check Harbor for the MI300/gfx94x-targeted build), and vLLM DFlash
still relies on Triton kernels that may not have gfx1151 parity.

## Conclusions & recommendations

### Revised key takeaway

**We do NOT own the recurrent-state rollback work.** Upstream merged #19493
four days ago (2026-04-19) and it's already in our build. Our iter-6 garbage-
output experiment needs to be re-run with `--spec-ckpt-num-tries 2
--ctx-checkpoints 16` before we conclude anything about lookup/draft spec
decode viability on Qwen3.6.

### What we *do* own

1. **Adapt #20700 to Qwen3.6.** It's WIP with a humble "please find a better
   solution" note. Qwen3.6 uses the same `qwen35` arch + same `mtp.*` tensor
   layout, so it's a close port. Getting MTP working at 50-80% acceptance on
   the existing #19493 checkpoint infra gets us the 2.89× multiplier needed.

2. **gfx1151-specific quant kernel tuning** (IQ3_XXS is 56% BW vs K-quants
   65%; iter-8). Upstream doesn't care about Strix Halo specifically — this
   is our bandwidth-bound territory.

3. **Empirical α measurement on #19493's checkpoint path** (and on #20075
   and #20700 branches) on Strix Halo — upstream benchmarks are all Apple M-
   series or NVIDIA, nobody has reported numbers on gfx1151.

4. **Re-establish the Strix Halo measurement baseline** for speculative
   decode on the post-#19493 codebase. The only current Strix Halo data
   points in upstream are bug reports (#22052 ROCm MAF on Qwen3.6-35B-A3B
   with Oculink + 7900 XTX setup).

### Deferred / won't-own

- Recurrent-state checkpoint in `llama_memory_recurrent` — **done upstream**.
- EAGLE3 for Qwen3.5/3.6 — **upstream actively merging**, wait.
- DFlash in llama.cpp — too early, multi-month timeline, wait for #22105.

### Immediate next step recommendation

Re-run iter-6 lookup spec decode test on the current `deps/llama.cpp/` build
with the new flags. If clean output and ≥50% acceptance, the gap to 40 tps
narrows further even without MTP.
