# Qwen3.6-27B fast-decode on Strix Halo (gfx1151)

Night run: 2026-04-23 00:20 - 06:00 CEST (~5h40m, 24 iterations, 22
commits). Orchestrator log: `/tmp/timeboxed-qwen36-fast-1776896422.md`.
Iteration notes in `notes/`, benchmark numbers + raw logs in `bench/`,
local patches in `patches/`. Chronological history: `CHANGELOG.md`.

## TL;DR

- Baseline Q4_K_M: **10.87 tps** (iter-5, bench/01-baseline-q4km.md).
- Baseline UD-Q2_K_XL: **13.82 tps** (iter-8, bench/03-low-bit-quants.md).
- **Realistic tps with lookup speculative decoding: 21--32 tps depending on
  workload, mean ~27 tps** (iter-26, bench/15-workload-diversity.md,
  4 workload types x 2 reps). Workload breakdown: code-review ~27,
  chat-with-history ~29, code-generation-from-spec ~25, translation/summary
  ~26. Speedup range: **1.54x--2.32x** over Q2_K_XL baseline,
  **1.96x--2.94x** over Q4_K_M baseline.
- **High end: 30.21 tps** reproducible via `bench/run-best.sh` on the
  self-referential code-review prompt (iter-20 full-mode verification).
  That is the prompt-shape best case, not the typical user experience.
- **Low end: ~21 tps** on natural-language translation / summary where the
  n-gram cache has no useful matches (alpha drops to 0.22--0.47).
- User target: **40 tps** — **NOT reached on any workload**. Lookup saturates
  at ~32 tps best case on this hardware (iter-16 clean +
  iter-26 chat rep2 at 32.03).
- MTP head (PR #20700) works end-to-end after a local dtype fix but
  regresses to 7.80 tps on Strix Halo (iter-15,
  bench/08-mtp-spec-v2.md) — the PR is tuned for >1 TB/s CUDA
  datacenter GPUs, not our 256 GB/s bandwidth-bound APU.
- Reproduce the high-end 30 tps result: `./bench/run-best.sh`
  (see "Reproduce" below; ~40-60 s wall on 256 tokens). For other workloads,
  swap the prompt file — see the prompts/ directory and
  bench/15-workload-diversity.md.

## Hardware / setup

- CPU/GPU: AMD Ryzen AI Max+ 395 / Radeon 8060S iGPU, arch gfx1151.
- Memory: 96 GiB VRAM pool, shared 256 GB/s LPDDR5x.
- OS: Fedora, Docker, llama.cpp built inside
  `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`.
- llama.cpp base commit: `0d0764d` (2026-04-22 master) with the iter-11
  `can_seq_rm` relaxation patch
  (`patches/llamacpp-qwen36-spec-decode.patch`) compiled into
  `libllama-common.so.0`. For MTP runs, PR #20700 applied on top
  (`patches/llamacpp-pr20700-applied.patch`).
- Bandwidth ceilings (weights-only, 256 GB/s / quant size):
  - Q4_K_M ~15.66 GiB -> 15.6 tps peak. Measured decode 10.87 tps = ~70%.
  - UD-Q2_K_XL ~11.0 GiB -> 21.6 tps peak. Measured 13.82 tps = ~64%.
- Real decode stays at 64-72% of weights-only ceiling across quants. The
  missing 30% is KV-cache reads, dispatch overhead, and the hybrid Gated
  DeltaNet layers being less kernel-optimised on ROCm than pure attention.

## Results

All tps numbers are decode-side (token-generation), pp=prefill excluded.
"Clean" = output verified coherent by eyeballing the log; "degen" = token
loops / gibberish.

| Config | Model | tps | alpha | Clean? | Source |
|---|---|---:|---:|:---:|---|
| Plain decode baseline | Q4_K_M, d=2048 | 10.87 | - | yes | bench/01-baseline-q4km.md |
| Plain decode baseline | Q3_K_S, d=2048 | 13.48 | - | yes | bench/03-low-bit-quants.md |
| Plain decode baseline | UD-IQ3_XXS, d=2048 | 11.98 | - | yes | bench/03-low-bit-quants.md |
| Plain decode baseline | UD-Q2_K_XL, d=2048 | 13.82 | - | yes | bench/03-low-bit-quants.md |
| **Lookup spec (iter-13)** | UD-Q2_K_XL, dm=4, dynamic | **30.05** | 0.65 | yes | bench/06-patched-lookup.md |
| Lookup spec (iter-16) | UD-Q2_K_XL, dm=5, dynamic greedy | 29.02 | 0.92 | yes (cleanest) | bench/09-lookup-tuning.md |
| Lookup spec (iter-16) | UD-Q2_K_XL, dm=5, dmin=2, dynamic | 31.13 | 0.84 | partial | bench/09-lookup-tuning.md |
| Lookup spec dm=8 | UD-Q2_K_XL, dm=8, greedy | 36-52 wall | 0.15-0.37 | **NO (degen)** | bench/09-lookup-tuning.md |
| Lookup on NL-QA prompt | UD-Q2_K_XL, dm=5 | 28.03 | 0.61 | yes | bench/09-lookup-tuning.md |
| Lookup on code-gen prompt | UD-Q2_K_XL, dm=5 | 19.73 | 0.08 | no | bench/09-lookup-tuning.md |
| Plain on MTP-merged GGUF | Q2_K_XL+MTP F16 | 11.91 | - | yes | bench/08-mtp-spec-v2.md |
| MTP K=1 (PR #20700) | Q2_K_XL+MTP merged | 7.80 | 1.00 | yes | bench/08-mtp-spec-v2.md |
| MTP K=2 (PR #20700) | Q2_K_XL+MTP merged | 7.76 | 1.00 | yes (K=1 effective) | bench/08-mtp-spec-v2.md |
| MTP K=3 (iter-18) | Q2_K_XL+MTP merged | ~7.6 | 1.00 | yes (K=1 effective) | iter-18 log |
| MTP K=4 (iter-18) | Q2_K_XL+MTP merged | ~7.6 | 1.00 | yes (K=1 effective) | iter-18 log |

**Note on the MTP K-sweep (iter-18):** K={1,2,3,4} produce **bit-identical
output** and identical `draft_n=127, accepted=127` counters. `--draft-max`
is a **no-op** for the MTP path: draft length is structurally 1 per step
regardless of CLI argument.

## What worked

1. **Lookup / n-gram speculative decode on UD-Q2_K_XL** is the single
   meaningful speed win this run. The recurrent-state checkpoint
   infrastructure merged upstream as PR #19493 (2026-04-19) makes the
   hybrid Gated DeltaNet rollback safe-enough at `--draft-max 4`. The
   visible M-RoPE `X < Y` decode failures are handled by
   checkpoint-restore-then-retry inside `common_speculative` and do not
   corrupt output at dm<=5.
2. **Quant-only drop**: moving from Q4_K_M -> UD-Q2_K_XL alone gives +27%
   (10.87 -> 13.82). Unsloth UD schemes keep higher-precision on sensitive
   layers so perplexity stays usable. VRAM also drops 18.79 GiB ->
   13.91 GiB, leaving headroom for a drafter.
3. **Tight docker-run bypass** avoids Harbor's broken compose template
   (see "Known issues"). Same image (`kyuz0/amd-strix-halo-toolboxes:rocm-7.2`)
   as Harbor, known-good ROCm 7.2 stack on gfx1151.
4. **MTP GGUF merge** via `patches/inject_mtp.py` successfully injects the
   15 `mtp.*` tensors from the Qwen3.6 HF safetensors as "block 64" on
   top of the Unsloth Q2_K_XL backbone. Norms written as F32 to match
   backbone dtype; matmul tensors kept F16. Loads and runs end-to-end on
   the PR #20700 build. This is reusable infrastructure for any future
   MTP variant (EAGLE-3 head, distilled drafter, ...).

## What didn't and why

1. **MTP via PR #20700 regresses on Strix Halo — AND is structurally
   single-token by design (iter-18 definitive ruling).** At K=1 with
   alpha=1.00, decode drops from 11.91 -> 7.80 tps (-35%). Per-step
   cost breakdown: plain step 83 ms; MTP K=1 step 128 ms. The extra
   cost is the 65th MTP transformer block (attention + 5120x17408 FFN
   + vocab matmul on 32768 trimmed vocab) + a 2-token verify ubatch on
   the 27B backbone. On CUDA datacenter parts (>1 TB/s HBM) this
   overhead hides behind memory stalls and alpha=1.00 wins. On 256
   GB/s LPDDR5x it is purely additive: we are already at 64-70% of BW,
   no idle bandwidth for the MTP layer to hide behind.

   **Iter-18 also ruled out the K>=2 cascade hypothesis structurally.**
   The earlier "n_max=1 hardcode at `tools/server/server.cpp:1309`" note
   was wrong: that line is a boolean gate, not a clamp. The real cap
   lives in `common/speculative.cpp:603-649`
   (`common_speculative_state_mtp::draft()`), which argmaxes **ONE**
   vocab-sized vector returned by `llama_get_mtp_logits()` and pushes
   exactly one draft token per step — `params.n_max` is declared
   `GGML_UNUSED`. The MTP logits tensor itself is single-vocab-row by
   construction in `src/llama-context.cpp:1819-1835`: one forward pass,
   one candidate. Running the merged GGUF with
   `--draft-max ∈ {1,2,3,4}` produces bit-identical output, tps ~=7.6,
   alpha=1.00, `draft_n=127, accepted=127` — `--draft-max` is a no-op
   for MTP, confirmed empirically.

   Unlocking a real K>1 cascade would require ~400-600 LoC across
   `src/models/qwen35.cpp`, `src/llama-graph.h`, `src/llama-context.cpp`,
   `common/speculative.cpp`, `include/llama.h` **plus retraining the
   MTP head for shift-k lookahead** (the released head predicts only
   position t+1 given hidden state at t; no ground truth for t+2..t+K).
   This is not a llama.cpp bug — it is the shape of the head Qwen3.6
   was trained with. On bandwidth-bound hardware, MTP K=1 **cannot**
   beat lookup; any MTP-based 40-tps story on this APU requires a
   retrained drafter, not a patch.
2. **Gated DeltaNet rollback degenerates at dm>=6.** The iter-16 sweep
   reaches 36-52 tps wall-clock at dm>=8 but the output is token loops
   (" wants wants", "111111", " ** ** **"). PR #20700's fuzzy `seq_rm`
   + 8-cell checkpoint ring in `llama-memory-recurrent.cpp` copes up to
   dm=5 on Qwen3.6; above that the rejection cascade outstrips the ring
   size and state desyncs. Fixing this is a real engineering job on
   `llama_memory_recurrent` (hundreds of LoC, multi-backend regression
   testing) — out of scope for one night.
3. **IQ-quants underperform K-quants of similar size** on this stack:
   UD-IQ3_XXS runs at 11.98 tps vs Q3_K_S at 13.48 tps (both ~3.2 bpw).
   gfx1151 ROCm 7.2 dequant kernels are unoptimised for IQ-class
   block formats; only 56% of BW ceiling vs 64-65% for K-quants
   (bench/03-low-bit-quants.md).
4. **Draft-model spec decode is blocked.** Qwen3.6 vocab size is 248320;
   no <=1B sibling exists (Qwen3-0.6B is 151936-vocab + different arch,
   rejected by llama.cpp compat check). EAGLE-3 or a trained drafter
   would unblock but neither is available today.
5. **Self-referential prompts inflate alpha.** The 0.65-0.92 alpha on
   the code-review prompt comes from the prompt literally asking the
   model to repeat a quoted module verbatim, which is ideal for n-gram
   lookup. On NL-QA alpha is 0.61 and on cold-start code-gen alpha
   collapses to 0.08. Lookup is a workload-shape amplifier, not a
   workload-agnostic speedup.

## Reproduce the 30 tps result

One-shot: `./bench/run-best.sh` (full 256 tokens, ~40-60 s).
Short sanity check: `./bench/run-best.sh --short` (64 tokens, ~15-25 s).

Exit code is 0 if measured tps >= 25, else 1.

Preconditions the script checks:

1. `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/` contains
   `Qwen3.6-27B-UD-Q2_K_XL.gguf`. If missing, the script prints the
   `huggingface-cli download` command and exits 2.
2. `deps/llama.cpp/build-rocm/bin/llama-lookup` exists (the iter-11
   patched binary). If missing, the script prints build-from-source
   instructions and exits 3.
3. `/dev/kfd` + `/dev/dri` devices are accessible.

Manual equivalent of the main run (exactly what produced 30.05 tps in
iter-13):

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video \
    -v $HOME/.cache/huggingface/hub:/models:ro \
    -v $PWD/deps/llama.cpp/build-rocm:/bld:ro \
    -v $PWD/prompts:/prompts:ro \
    -e LD_LIBRARY_PATH=/bld/bin:/opt/rocm-7.2.0/lib \
    kyuz0/amd-strix-halo-toolboxes:rocm-7.2 \
    /bld/bin/llama-lookup \
        -m /models/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-UD-Q2_K_XL.gguf \
        -ngl 99 -fa on \
        -f /prompts/prompt_code.txt \
        -n 256 --draft-max 4
```

Expected output line at the end:
`decoded  259 tokens in   8.618 seconds, speed:   30.053 t/s`
and `accept       = 65.289%` give-or-take a couple percent across runs.
90 "decode: failed to initialize batch" lines will appear in the log;
this is expected (M-RoPE retries handled by PR #19493 checkpoint rollback).
Output is a coherent code review in Markdown.

## What's plausible to reach 40 tps (ordered by effort:gain)

The numbers in parentheses are this-author's best estimate, not promises.
All require real engineering, none are 1-day fixes.

1. **EAGLE-3 port (PR #21437 on top of #18039)** — **the single most
   promising path**. EAGLE-3's draft head is a dedicated 1-layer
   transformer that runs in parallel with target verify, NOT embedded
   in the backbone forward. Per-step drafter cost is a small fraction
   of the 27B backbone (vs MTP's 45 ms full block). #21437
   specifically adds Qwen3.5 linear-attention EAGLE3 support (Qwen3.6
   should be a small delta since it shares the `qwen35` arch). No
   EAGLE-3 drafter weights exist for Qwen3.6-27B yet — training one
   is part of the effort. Details: `notes/09-eagle3-future-path.md`.
   Estimate: 2-4 weeks total (C++ port + drafter training + hybrid
   rollback tuning). Projected: alpha ~= 0.55-0.70 -> ~35-45 tps on
   Q2_K_XL (upper end plausibly hits target).
2. **Fix GDN rollback for high draft-max** — at dm=5 we get 29-31 tps
   clean. The dm=8+ sweep shows 36-52 tps wall-clock is physically
   present but degenerate. Fixing `llama_memory_recurrent::seq_rm`
   checkpoint-ring sizing + verify-batch bounds for dm>=6 should unlock
   another ~20-30% cleanly. Estimate: 200-400 LoC in
   `src/llama-memory-recurrent.cpp`, 3-5 days incl. regression tests
   across ROCm/CUDA/Metal. Projected: ~36-40 tps.
3. **Prompt-specific static lookup caches** — if the workload is
   repeat-heavy (e.g. editing a specific codebase, doc rewriting,
   code review over a fixed corpus), pre-built static caches matched
   to the workload should recover the dynamic-cache alpha (0.85-0.92)
   without warm-up cost. Our generic code-corpus static cache did not
   overlap the benchmark prompt and hurt. Per-project caches are a few
   hours of glue. Projected: ~33-36 tps on matched workloads.
4. **Retrain MTP head for shift-k lookahead** — theoretical-only path
   for unlocking K>1 on the current PR #20700 infrastructure. Would
   require: (a) 400-600 LoC in llama.cpp to produce and consume K
   logits tensors from a single MTP forward (graph + context API +
   speculative.cpp), (b) a retrained drafter head that predicts t+1..t+K
   given hidden state at t — Qwen's released MTP head only predicts
   t+1 (single-shift). No training data or recipe is public; this is
   research, not engineering. Even if it worked, per-step MTP overhead
   on this APU would likely keep it below EAGLE-3 on an effort-to-gain
   basis. Documented for completeness; **do not prioritise**.
5. **Lightweight custom MTP** — train a *new* MTP head much smaller
   than the released one (<10% of backbone cost), run K=1. Same
   training+integration burden as above but less graph work. Requires
   training compute. Projected: 12-15 tps, worse than EAGLE-3.
6. **vLLM-ROCm migration (DFlash, PR #22105 upstream-side)** —
   vLLM has DFlash in mainline and z-lab publishes drafters for many
   Qwen3.5/3.6 variants, though NO Qwen3.6-27B DFlash drafter exists
   at the time of writing (only Qwen3.5-27B-DFlash and
   Qwen3.6-35B-A3B-DFlash). Hybrid-target speedup is intrinsically
   limited per PR #22105's own evaluation. Different inference stack
   entirely; weeks of porting / ops work to match Harbor integration.
7. **Upstream our `can_seq_rm` patch** (ready to file, see
   `patches/upstream-pr-draft/`) — zero-effort for the upstream
   community: 32 added / 2 removed lines in `common/common.cpp`,
   validated end-to-end here via `llama-server` (iter-21, α=1.00) and
   `llama-lookup` (iter-13, 30.05 tps). Doesn't move the needle on
   *our* local number, but unblocks every hybrid/recurrent model
   (Qwen3.5/3.6, Qwen3-Next, GLM4-MoE, Mamba, LFM2, Plamo2,
   Kimi-Linear) for any drafter kind (lookup, ngram-cache, draft-model,
   future MTP/EAGLE3). Orthogonal to the Qwen3.5 MTP WIP #20700.
   Patch + PR-body drafted, applies cleanly to master as of
   `86db42e`; review and file tomorrow.

## Files

- `README.md` — this file.
- `CHANGELOG.md` — chronological 24-iteration night log with commit hashes.
- `MORNING.md` — 30-second briefing, reproduce instructions, the ONE decision.
- `notes/00-context.md` — orchestrator snapshot + phased plan + final
  outcome footer.
- `notes/01-harbor-state.md` — Harbor state when we started, compose bug.
- `notes/02-mtp-inspection.md` — MTP absent in Unsloth GGUF.
- `notes/03-drafter-strategy.md` — HF safetensors MTP tensors found.
- `notes/04-llamacpp-qwen36-support.md` — upstream support survey.
- `notes/05-mrope-spec-fix.md` — first diagnosis of the rollback issue.
- `notes/06-upstream-survey.md` — PR #19493 / PR #20700 / EAGLE3 status.
- `notes/07-pr20700-port-plan.md` — file-by-file PR #20700 inventory.
- `notes/08-final-state.md` — postmortem + lessons for future runs.
- `notes/09-eagle3-future-path.md` — EAGLE-3 (PR #21437) + DFlash
  (PR #22105) research for the next run at 40 tps.
- `bench/01-baseline-q4km.md` — Q4_K_M baseline.
- `bench/02-lookup-spec.md` — first lookup run (iter-6, bug noted).
- `bench/03-low-bit-quants.md` — Q3/Q2/IQ3 sweep.
- `bench/04-longctx.md` — Q2_K_XL 2k/4k/8k/16k.
- `bench/05-lookup-with-checkpoint.md` — iter-9 server-side lookup
  failure (common_context_can_seq_rm rejection).
- `bench/06-patched-lookup.md` — **the 30.05 tps run** (iter-13).
- `bench/07-mtp-spec.md` — first MTP attempt (broadcast assert).
- `bench/08-mtp-spec-v2.md` — MTP running, regresses to 7.80 tps.
- `bench/09-lookup-tuning.md` — draft-max sweep, lookup saturated ~31 tps.
- `bench/10-mtp-cascade.md` — iter-18 K-sweep proving MTP is structurally K=1.
- `bench/11-canonical-run.md` — iter-20 full-mode reproducibility verification
  (30.21 tps).
- `bench/12-server-validation.md` — iter-21 llama-server path validates
  iter-11 patch end-to-end.
- `bench/14-server-tuning.md` — iter-24 server-side ngram-cache config sweep.
- `bench/15-workload-diversity.md` — iter-26 4-workload tps range (21--32 tps, mean ~27).
- `bench/run-best.sh` — reproducible one-shot bench script.
- `patches/llamacpp-qwen36-spec-decode.patch` — iter-11 can_seq_rm relax.
- `patches/upstream-pr-draft/` — polished upstream PR materials
  (format-patch + PR body + README). Ready for Ivan to review and
  file; applies cleanly to master as of `86db42e`.
- `patches/llamacpp-pr20700-applied.patch` — PR #20700 snapshot on 0d0764d.
- `patches/llamacpp-mrope-spec-decode-fix.patch` — iter-7 diagnostic patch.
- `patches/inject_mtp.py` — merge MTP safetensors into Q2_K_XL GGUF.
- `patches/strix-halo-builder.Dockerfile` — custom builder image.
- `prompts/prompt_code.txt` — the 1766-token code-review prompt.
- `prompts/prompt_codegen.txt` — 1080-token code-generation-from-spec prompt (iter-26).
- `prompts/prompt_chat.txt` — 954-token chat-with-history prompt (iter-26).
- `prompts/prompt_nl.txt` — 553-token translation/summary prompt (iter-26).
- `build-artifacts/qwen36-27b-mtp-merged.gguf` — 11.83 GiB MTP-merged
  target model (gitignored).
- `build-artifacts/lookup-cache-static.bin` — 15 MB static n-gram cache
  from 10 MB code corpus (gitignored, helps only on matched prompts).

## Known issues / hazards

- **MTP on PR #20700 is structurally single-token (iter-18 ruling).**
  `--draft-max N` is a no-op on the MTP path. The cap is NOT the earlier
  alleged hardcode at `tools/server/server.cpp:1309` (which is a
  boolean gate, not a clamp); it is in `common/speculative.cpp:603-649`
  where `common_speculative_state_mtp::draft()` argmaxes one
  vocab-sized vector from `llama_get_mtp_logits()` and pushes exactly
  one token per step, with `params.n_max` marked `GGML_UNUSED`. The
  MTP graph produces a single-row logits tensor
  (`src/llama-context.cpp:1819-1835`). K>1 would need ~400-600 LoC
  **and** a retrained shift-k head. See
  `patches/llamacpp-unlock-mtp-k.patch` (empty-by-design) for
  references.
- **Env-relink during iter-16 rebuild.** Building `llama-lookup-create`
  with ninja during iter-16 relinked `libllama-common.so.0` to the
  PR #20700 artifact. iter-13's 30.05 tps was measured against the
  pre-PR-20700 lib; the cleanest equivalent on today's symlinked lib is
  iter-16's 29.02 tps (dm=5 dynamic, greedy). Numbers are within noise.
- **MTP-merged GGUF needs F32 norm tensors** (`patches/inject_mtp.py`
  correctly handles this). F16 norms trip a
  `GGML_ASSERT(nb10 % sizeof(src1_t) == 0)` in `ggml-cuda/binbcast.cu`
  because the backbone's norms are F32 and the multiply is mixed-dtype.
- **Harbor compose template is broken for rocm images.** The Harbor
  `.env` has `EXTRA_ARGS="llama-server --no-mmap ..."` which produces
  `argv = ["-m","<path>","llama-server",...]` -> argv[0]="-m" and fails
  container start. The ROCm image has no entrypoint. Not regressed by
  this run; pre-existing. Restored original `.env` at end. Direct
  `docker run` invocation is the workaround.
- **Unsloth GGUF convert drops MTP tensors**. Qwen3.6's root arch is
  `Qwen3_5ForConditionalGeneration` (VLM wrapper); Unsloth's text-only
  convert path skips the `mtp.*` tensors. The source safetensors DO
  contain all 15 MTP tensors. `patches/inject_mtp.py` pulls those 2 shards
  directly and injects them.
- **90+ M-RoPE "decode failed" log lines per 256-token lookup run are
  NORMAL.** They are swallowed by checkpoint-restore-then-retry in the
  speculative loop and do not affect output. Do not disable the
  speculative path to silence them.
