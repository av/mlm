# Qwen3.6-27B fast-decode on Strix Halo (gfx1151)

Night run: 2026-04-23 00:20 - 04:30 CEST. Orchestrator log:
`/tmp/timeboxed-qwen36-fast-1776896422.md`. Iteration notes in `notes/`,
benchmark numbers + raw logs in `bench/`, local patches in `patches/`.

## TL;DR

- Baseline Q4_K_M: **10.87 tps** (iter-5, bench/01-baseline-q4km.md).
- Baseline UD-Q2_K_XL: **13.82 tps** (iter-8, bench/03-low-bit-quants.md).
- Best achieved tonight: **30.05 tps** via lookup speculative decoding on
  UD-Q2_K_XL with `--draft-max 4`, α=65%, coherent output (iter-13,
  bench/06-patched-lookup.md). **2.17x** speedup over Q2_K_XL baseline,
  **2.76x** over the Q4_K_M baseline.
- User target: **40 tps** — **NOT reached**. Lookup saturates at
  ~31 tps clean on this hardware (iter-16, bench/09-lookup-tuning.md).
- MTP head (PR #20700) works end-to-end after a local dtype fix but
  regresses to 7.80 tps on Strix Halo (iter-15,
  bench/08-mtp-spec-v2.md) — the PR is tuned for >1 TB/s CUDA
  datacenter GPUs, not our 256 GB/s bandwidth-bound APU.
- Reproduce the 30 tps result: `./bench/run-best.sh`
  (see "Reproduce" below; ~40-60 s wall on 256 tokens).

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

1. **MTP via PR #20700 regresses on Strix Halo.** At K=1 with alpha=1.00,
   decode drops from 11.91 -> 7.80 tps (-35%). Per-step cost breakdown:
   plain step 83 ms; MTP K=1 step 128 ms. The extra cost is the 65th
   MTP transformer block (attention + 5120x17408 FFN + vocab matmul on
   32768 trimmed vocab) + a 2-token verify ubatch on the 27B backbone.
   On CUDA datacenter parts (>1 TB/s HBM) this overhead hides behind
   memory stalls and alpha=1.00 wins. On 256 GB/s LPDDR5x it is
   purely additive: we are already at 64-70% of BW, there is no idle
   bandwidth for the MTP layer to hide behind. **PR #20700 also
   hardcodes `n_max=1` at `tools/server/server.cpp:1309`**, so `--draft-max>=2`
   silently caps; true K>=2 cascade would need
   `build_mtp_head` recursion plumbing that the PR does not ship.
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

1. **Fix GDN rollback for high draft-max** — at dm=5 we get 29-31 tps
   clean. The dm=8+ sweep shows 36-52 tps wall-clock is physically
   present but degenerate. Fixing `llama_memory_recurrent::seq_rm`
   checkpoint-ring sizing + verify-batch bounds for dm>=6 should unlock
   another ~20-30% cleanly. Estimate: 200-400 LoC in
   `src/llama-memory-recurrent.cpp`, 3-5 days incl. regression tests
   across ROCm/CUDA/Metal. Projected: ~36-40 tps.
2. **EAGLE-3 port (PR #21437)** — isolated draft head, runs once in
   parallel with the target verify batch. Structurally better-suited to
   bandwidth-bound hardware than MTP (smaller, detached from target
   forward). Estimate: 1-2 weeks C++ port work + drafter head training.
   Projected: alpha ~= 0.80 -> ~40+ tps.
3. **Lightweight MTP** — the current MTP head is ~380M params (a full
   transformer block on a 27B backbone is ~1.4% of total, but it runs
   every step and adds ~45 ms). Prune to <10% of backbone cost
   (half the FFN width or single-head attention) -> K=1 savings
   outweigh overhead. Requires training, not in-tree. Projected:
   12-15 tps at K=1, could stack with lookup.
4. **Prompt-specific static lookup caches** — if the workload is
   repeat-heavy (e.g. editing a specific codebase, doc rewriting,
   code review over a fixed corpus), pre-built static caches matched
   to the workload should recover the dynamic-cache alpha (0.85-0.92)
   without warm-up cost. Our generic code-corpus static cache did not
   overlap the benchmark prompt and hurt. Per-project caches are a few
   hours of glue. Projected: ~33-36 tps on matched workloads.
5. **vLLM-ROCm migration** — vLLM has the DFlash draft stack in
   mainline (Qwen3.6-35B-A3B drafter published) and handles hybrid
   rollback more cleanly than llama.cpp. Different inference stack
   entirely; weeks of porting / ops work to match Harbor integration.

## Files

- `README.md` — this file.
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
- `bench/run-best.sh` — reproducible one-shot bench script.
- `patches/llamacpp-qwen36-spec-decode.patch` — iter-11 can_seq_rm relax.
- `patches/llamacpp-pr20700-applied.patch` — PR #20700 snapshot on 0d0764d.
- `patches/llamacpp-mrope-spec-decode-fix.patch` — iter-7 diagnostic patch.
- `patches/inject_mtp.py` — merge MTP safetensors into Q2_K_XL GGUF.
- `patches/strix-halo-builder.Dockerfile` — custom builder image.
- `prompts/prompt_code.txt` — the 1766-token code-review prompt.
- `build-artifacts/qwen36-27b-mtp-merged.gguf` — 11.83 GiB MTP-merged
  target model (gitignored).
- `build-artifacts/lookup-cache-static.bin` — 15 MB static n-gram cache
  from 10 MB code corpus (gitignored, helps only on matched prompts).

## Known issues / hazards

- **PR #20700 hardcodes `n_max=1` in `tools/server/server.cpp:1309`.**
  Any MTP K>=2 is silently capped to K=1. Not a bug in our code, but
  important: don't trust `--draft-max N` on the MTP path above 1.
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
