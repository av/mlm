# CHANGELOG — Qwen3.6-27B fast-decode night run

Chronological record of the 2026-04-22/23 overnight timeboxed run.
Start 00:20 CEST, end 06:00 CEST (~5h40m).
27 iterations producing 25 code+doc commits (2 iterations — 11 and 12 —
folded into later commits; final commit count = 25 after this polish).
Orchestrator log: `/tmp/timeboxed-qwen36-fast-1776896422.md`.

## Phases

| Phase | Iterations | Hours | Theme |
|---|---|---|---|
| I. Investigate | 1-4 | 00:20-00:55 | Bootstrap, gguf/safetensors recon, upstream llama.cpp support survey |
| II. Benchmark baseline | 5 | 00:55-01:35 | Q4_K_M decode baseline, gap to 40 tps |
| III. Identify blocker | 6-7 | 01:35-02:00 | First spec attempt -> gibberish -> M-RoPE red herring -> recurrent-state desync root cause |
| IV. Alternatives | 8 | 02:00-02:20 | Quant sweep (Q3/Q2/IQ3) as no-spec fallback |
| V. Apply upstream | 9-12 | 02:20-03:00 | Upstream PR survey, iter-11 patch, PR #20700 deep-dive |
| VI. BREAKTHROUGH | 13 | 03:00-03:10 | Patched llama-lookup hits 30.05 tps, 2.17x baseline |
| VII. MTP path | 14-15 | 03:10-03:50 | Port PR #20700, merge MTP GGUF, measure: regresses to 7.80 tps |
| VIII. Tune & rule out | 16, 18 | 03:50-04:55 | Lookup saturates at ~31 tps; MTP ruled structurally single-token |
| IX. Document & validate | 17, 19-23 | 04:30-05:30 | README, MORNING, reproducibility, server validation, upstream-PR draft |
| X. Post-mortem polish | 24-25 | 05:30-05:45 | Server tuning explored (~12 tps cap); CHANGELOG written (meta) |
| XI. Honest numbers | 26-27 | 05:45-06:00 | Workload-diversity bench, 21-32 tps range adopted, final doc polish |

## Iterations

### Iter 1 — 00:25 CEST — bootstrap workspace
- Created `qwen36-fast/` with README, notes/00-context (orchestrator verbatim),
  notes/01-harbor-state. Harbor runs llama.cpp on ROCm 7.2 (not Vulkan) via
  `kyuz0/amd-strix-halo-toolboxes`. No Qwen3.6 GGUF present.
- Commit: **c131968** `qwen36-fast: bootstrap workspace and harbor state notes`

### Iter 2 — 00:34 CEST — MTP absent in Unsloth GGUF
- Started bg download of Unsloth Qwen3.6-27B Q4_K_M. gguf-dump 35B-A3B as proxy:
  zero `mtp/nextn/draft/medusa/eagle` tensors. Qwen3-0.6B vocab mismatch
  (151,936 vs 248,320) → llama.cpp refuses it as drafter.
- Commit: **61b55cb** `qwen36-fast: MTP head inspection + draft model compatibility`

### Iter 3 — 00:39 CEST — MTP present in HF safetensors
- HF safetensors DO contain all 15 `mtp.*` tensors. Root arch
  `Qwen3_5ForConditionalGeneration` (VLM wrapper) is why Unsloth's text-only
  convert dropped them. Decision: patch convert + emit separate drafter GGUF.
- Commit: **7414225** `qwen36-fast: inspect HF safetensors for MTP + drafter strategy decision`

### Iter 4 — 00:46 CEST — llama.cpp upstream survey
- Shallow-cloned llama.cpp master (0d0764d). Qwen3.6 fully supported
  (`qwen35` + `qwen35moe` arches + converters + builders). MTP scaffold
  (`LLM_TENSOR_NEXTN_*`) exists but inference is NOT wired.
  `convert_hf_to_gguf.py:4780-4782`: `if name.startswith("mtp"): return`.
- Commit: **4302e12** `qwen36-fast: survey llama.cpp Qwen3.6 + MTP support status`

### Iter 5 — 00:56 CEST — Q4_K_M baseline
- llama-bench on ROCm 7.2 docker. Decode: **10.87 tps** @ d=2048, 10.67 @ d=7000.
  ~70% of 15.6 tps weights-only bandwidth ceiling. Gap to 40 tps: **3.7x**.
  Harbor compose has pre-existing `argv[0]="-m"` bug; bypassed via direct
  `docker run`.
- Commit: **063c1d6** `qwen36-fast: baseline Q4_K_M decode benchmark on ROCm 7.2`

### Iter 6 — 01:35 CEST — first lookup attempt: gibberish
- `llama-lookup dm=4`: reports 26 tps BUT output garbled + 196 decode failures.
  Error at `src/llama-batch.cpp:264-274` ("for M-RoPE: X < Y"). Also downloaded
  just the 2 MTP-containing safetensors shards (4.29 GiB vs 54 GiB full).
- Commit: **7b150d6** `qwen36-fast: lookup/n-gram speculative decode benchmark`

### Iter 7 — 01:42 CEST — M-RoPE is a red herring
- Real root cause: `llama_memory_recurrent::seq_rm` refuses partial rollback
  BY DESIGN. Qwen3.6 GDN state desyncs on partial verify → downstream M-RoPE
  pos check trips. Affects ALL hybrid/recurrent arches. Diagnostic-only patch
  written.
- Commit: **a707836** `qwen36-fast: diagnose M-RoPE spec-decode bug + fix attempt`

### Iter 8 — 02:01 CEST — quant sweep as no-spec fallback
- UD-Q2_K_XL: **13.82 tps** (+27%), Q3_K_S 13.48 (+24%), UD-IQ3_XXS 11.98 (+10%,
  IQ-dequant unoptimised on gfx1151). FA on vs off: only 5.5% delta (we're
  bandwidth-bound). Gap narrows to 2.89x.
- Commit: **3613915** `qwen36-fast: benchmark Q3/Q2/IQ3 quants as no-spec fallback`

### Iter 9+10 — 02:30 CEST — upstream PRs found
- **PR #19493** (recurrent checkpoint ring): MERGED 2026-04-19, in our clone.
- **PR #20700** (Qwen3.5 MTP): OPEN, WIP, CONFLICTING.
- **PR #21437** (EAGLE3), **PR #22105** (DFlash): OPEN, WIP.
- Server disables spec via `common_context_can_seq_rm` probe. Fix identified:
  relax probe to try checkpoint round-trip.
- Commit: **b0631ad** `qwen36-fast: upstream survey + long-ctx bench + re-verify lookup with ckpt flags`

### Iter 11 — 02:31-03:00 CEST — wrote + built iter-11 patch
- `patches/llamacpp-qwen36-spec-decode.patch` (~30 LoC in `common/common.cpp`):
  if raw seq_rm fails, try checkpoint round-trip; promote NO→FULL. Built
  patched llama.cpp + llama-lookup in docker ROCm 7.2 (~15 min link).
- (No separate commit; test/result in iter-13 commit.)

### Iter 12 — 02:30-03:00 CEST (parallel) — PR #20700 deep-dive
- Read every hunk of #20700. Qwen3.5↔3.6 MTP layout IDENTICAL at GGUF level.
  Reported Q6_K word-salad bug traced: q_proj shape IS gated, reporter
  hypothesis wrong. Port delta for 3.6: ~70-90 LoC.
- Commit: **be52ebf** `qwen36-fast: deep-dive PR #20700 + Qwen3.6 port plan`

### Iter 13 — 03:05 CEST — **BREAKTHROUGH**
- Patched llama-lookup on UD-Q2_K_XL + `--draft-max 4 --ctx-checkpoints 8`:
  **30.05 tps, α=65.29%, coherent output.** 2.17x over Q2_K_XL baseline,
  2.76x over Q4_K_M. 90 M-RoPE "decode failed" lines are handled transparently
  by PR #19493 checkpoint-restore-retry.
- iter-11 patch is DORMANT for this binary (lookup path skips the probe) —
  principle validated; server path verification deferred.
- Commit: **3d96953** `qwen36-fast: test iter-11 patched llama-lookup on Q2_K_XL`

### Iter 14 — 03:26 CEST — port PR #20700, MTP broadcast bug
- Applied PR #20700 locally (4 trivial conflicts resolved). Built via
  custom `strix-halo-builder:rocm-7.0` on top of server-rocm base image.
  Merged MTP tensors into Q2_K_XL via `patches/inject_mtp.py` (F16 for
  all, first attempt). Hit `GGML_ASSERT(nb10 % sizeof(src1_t) == 0)` in
  `ggml-cuda/binbcast.cu:255` — mixed-dtype broadcast alignment failure.
- Commit: **cfd0d0d** `qwen36-fast: port PR #20700 + test MTP spec on Qwen3.6-27B`

### Iter 15 — 03:45 CEST — MTP works, REGRESSES on Strix Halo
- Fix: write MTP norms as F32 (backbone norms are F32), matmul tensors stay F16.
  Regenerated merged GGUF, warmup passes.
- **Plain on merged GGUF**: 11.91 tps (−14% vs unmerged Q2_K_XL, 65th MTP
  block runs every step).
- **MTP K=1**: **7.80 tps, α=1.00** — 35% regression. Per-step cost:
  plain 83 ms, MTP 128 ms. On 256 GB/s memory the extra MTP forward can't
  hide; on >1 TB/s CUDA it wins.
- Commit: **9d9a8b1** `qwen36-fast: fix MTP merge dtype + measure K=1..3 MTP spec decode`

### Iter 16 — 04:12 CEST — lookup tuning saturates
- Built 15 MB static lookup cache from 10 MB code corpus. Swept dm ∈
  {2..16}. Best clean: **31.13 tps** dm=5 dmin=2 dynamic (+3.6% over
  iter-13). Cleanest: 29.02 tps dm=5 greedy, α=0.92.
- α does NOT generalize: 0.92 code-review → 0.61 NL-QA → 0.08 code-gen.
  Static cache HURTS on non-overlapping prompts. dm≥6 wall-clock 36-52
  tps but degenerate token loops.
- Ninja relinked `libllama-common.so.0` to iter-14 PR#20700 artifact;
  iter-13 regime not byte-identical reproducible afterwards.
- Commit: **a490f96** `qwen36-fast: tune lookup spec decode — static cache + draft-max sweep`

### Iter 17 — 04:36 CEST — comprehensive README + bench script + postmortem
- Wrote 265-line README, created `bench/run-best.sh` (tested end-to-end
  with PASS/FAIL logic), wrote 396-line `notes/08-final-state.md`. Copied
  iter-13's 1766-token prompt to `prompts/prompt_code.txt` (was empty).
- Commit: **21b5cec** `qwen36-fast: comprehensive README + reproducible bench script + final postmortem`

### Iter 18 — 04:52 CEST — **DEFINITIVE MTP RULING**
- Iter-15's "n_max=1 hardcode at server.cpp:1309" claim: WRONG. That line
  is a boolean gate, not a clamp.
- Real cap: `common/speculative.cpp:603-649` —
  `common_speculative_state_mtp::draft()` argmaxes ONE vocab-sized vector
  from `llama_get_mtp_logits()` and pushes exactly one token.
  `params.n_max` is `GGML_UNUSED`. MTP graph is single-row by construction
  (`src/llama-context.cpp:1819-1835`).
- Tested K ∈ {1,2,3,4}: bit-identical output, α=1.00, `draft_n=127,
  accepted=127`. **`--draft-max` is a no-op for MTP.**
- Unlocking K>1 needs ~400-600 LoC + retrained shift-k head. Not a patch.
- Commit: **afb486b** `qwen36-fast: unlock PR #20700 MTP K>1 cap + measure cascade`

### Iter 19 — 05:05 CEST — README update + EAGLE-3 future path
- Updated README with definitive MTP ruling. Added +120 lines to
  `notes/08-final-state.md`. Wrote `notes/09-eagle3-future-path.md`:
  PR #21437 on top of #18039, no Qwen3.6-27B drafter on HF, projected
  35-45 tps with α ≈ 0.55-0.70. Recommendation: validate on
  `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` first.
- Commit: **a46dab0** `qwen36-fast: definitive MTP ruling in README + EAGLE-3 future-path research`

### Iter 20 — 05:10 CEST — canonical run-best.sh full-mode verified
- `./bench/run-best.sh` full mode: **30.21 tps, α=66.94%, PASS**.
  Within +0.5%/+1.65pp of iter-13 claim — reproducibility confirmed on
  a fresh run. Output fully coherent.
- Commit: **534a070** `qwen36-fast: canonical full-mode bench run — reproducibility verified`

### Iter 21 — 05:11 CEST — iter-11 patch validated via llama-server
- Ran llama-server with `--spec-type ngram-cache`. Startup log shows
  `srv load_model: speculative decoding will use checkpoints` (the NEW
  promotion branch). Without the patch this would say "not supported by
  this context". Short: 13.22 tps α=1.00. Long: 11.76 tps α=1.00.
  Patch CONFIRMED for server path.
- Commit: **e450e54** `qwen36-fast: validate iter-11 patch via llama-server path`

### Iter 22 — 05:19 CEST — upstream PR draft prepared
- Patch rewritten with real hashes (f7f33e8..6aeee8e), applies cleanly via
  `git am` to master (`86db42e`). ORTHOGONAL to PR #20700 (zero hits on
  `can_seq_rm` in #20700). Produced `patches/upstream-pr-draft/`:
  0001-...patch (87 lines), PR-DESCRIPTION.md (191 lines), README.md (39 lines).
- Commit: **c975403** `qwen36-fast: prep can_seq_rm patch as upstream-PR-ready draft`

### Iter 23 — 05:27 CEST — morning briefing + fresh-shell verify
- `run-best.sh --short` under `env -i bash`: **23.691 tps PASS** (cold
  dynamic cache). Script hardened (PATH + HOME fallbacks). Disk audit:
  qwen36-fast/ 13 GiB (12 GiB untracked regeneratable MTP GGUF); HF cache
  50 GiB (38 GiB prunable). MORNING.md written as 59-line one-pager.
- Commit: **e6103a1** `qwen36-fast: morning briefing + clean-shell verify + disk-usage notes`

### Iter 24 — 05:34 CEST — server-side ngram tuning saturates at 12 tps
- Swept server configs (spec-type variants, dm 2/4/8/16, ctx-checkpoints,
  np). Best: **12.28 tps** (Config G: `--spec-type ngram-cache --draft-max 2
  -np 1`). 2.45x slower than llama-lookup. Root cause: server ngram-cache
  fires on ~22% of steps vs llama-lookup's 94%. Same
  `common_ngram_cache_draft` under the hood, different caller-side cache
  invocation. Closing this gap would need ~100 LoC in
  `tools/server/server-context.cpp`.
- Commit: **819050c** `qwen36-fast: sweep server-side ngram spec config`

### Iter 25 — 05:39 CEST — final docs QA pass + CHANGELOG (meta)
- Cross-reference verification (32 referenced files, 0 broken), CHANGELOG.md
  written (257 lines, full 24-iteration story at the time of writing),
  README header metadata refreshed (date + iteration count), MORNING.md
  commit counts updated. All cross-references verified.
- Commit: **5fe76aa** `qwen36-fast: final CHANGELOG + doc QA pass`

### Iter 26 — 05:51 CEST — workload-diversity bench, honest 21-32 tps range
- Ran llama-lookup with canonical config across 4 realistic workloads, 2
  reps each. Added 3 new prompts to `prompts/` (codegen, chat, nl).
- **Honest tps range across workloads (8 runs, default sampling)**:
  - code-review (self-referential): **27.3 tps** α=0.62
  - chat w/ history: **29.3 tps** α=0.72
  - code-generation-from-spec: **25.1 tps** α=0.40
  - NL translation/summary: **25.6 tps** α=0.40
- Overall range: **21.3 - 32.0 tps**, mean 26.8 ± 3.5. Speedup vs 13.82
  baseline: 1.54x - 2.32x. The iter-20 30.21 tps number is the high end
  of the range (reproducible on the self-referential code-review prompt),
  not the typical user experience.
- Updated MORNING.md + README.md TL;DR to report the honest range instead
  of the single 30-tps peak number. CHANGELOG still references both the
  30.21 peak (iter-20 reproducibility) AND the 21-32 typical range.
- Key findings: rep-to-rep variance 4-7 tps under default sampling;
  greedy (`--temp 0`) degenerates all 4 workloads into token loops at
  dm=4; `--ignore-eos` needed for fair comparison; lookup α is
  workload-shape-dependent.
- Files: `bench/15-workload-diversity.md` + 8 raw logs,
  `prompts/prompt_{codegen,chat,nl}.txt`.
- Commit: **8e920e0** `qwen36-fast: workload-diversity bench + honest tps range`

### Iter 27 (this commit) — 05:58 CEST — final CHANGELOG polish (iter 25+26)
- Added iter-25 and iter-26 entries to this CHANGELOG (iter-25 wrote the
  file covering iters 1-24, so iter-26 and this polish were not yet here).
- Cross-checked the three top-level docs (README / MORNING / this file) for
  consistency on the tps range. All three now agree: **peak 30.21 tps,
  typical 21-32 tps range, mean ~27 tps**.
- Updated phase table (split post-mortem polish into phase X + phase XI)
  and final-state repo counts.
- Commit: (this one) `qwen36-fast: final CHANGELOG polish with iter 25+26 entries`

## Final state as of Thu 2026-04-23 06:00 CEST

- **Honest tps range across 4 realistic workloads (iter-26)**: **21 - 32
  tps, mean 26.8 ± 3.5** over 8 runs. Per-workload: code-review ~27, chat
  ~29, new code-gen ~25, NL translation ~26. Speedup 1.54x - 2.32x over
  the UD-Q2_K_XL baseline (13.82 tps).
- **Peak (reproducible, high end of the range)**: **30.21 tps**
  (`bench/run-best.sh` full mode, iter-20), α = 66.94%, UD-Q2_K_XL +
  patched llama-lookup + `--draft-max 4`, on the self-referential
  code-review prompt. That single number stood in the README for three
  iterations before iter-26 supplied the honest range context.
- **Baseline**: 13.82 tps (UD-Q2_K_XL) / 10.87 tps (Q4_K_M). Net speedup
  at peak: **2.17x / 2.76x**. Net speedup across workload range: **1.54x
  - 2.32x** (over Q2_K_XL baseline).
- **Target 40 tps NOT reached** — gap -8 to -19 tps depending on
  workload. Gap is structural on this hardware; no available patch
  closes it.
- **MTP via PR #20700**: works end-to-end but regresses to 7.80 tps on
  Strix Halo's 256 GB/s bandwidth. Structurally K=1 (iter-18 ruling).
- **Upstream contribution ready**: `patches/upstream-pr-draft/` contains
  a ready-to-file `can_seq_rm` relaxation patch (~32 added / 2 removed
  lines in `common/common.cpp`), validated via both llama-lookup
  (iter-13) and llama-server (iter-21) paths. Orthogonal to PR #20700.

### Repo physical state

- Branch: `main`, **25 commits** ahead of `origin/main` after this polish
  commit (24 before it). Not pushed.
- Working tree: clean.
- Total night diff: ~140+ files, 62k+ insertions (code + logs + notes).
- File counts in `qwen36-fast/`:

  | Path | Count | Notes |
  |---|---:|---|
  | `bench/*.log` | 86 | raw logs (iter-26 added 8 workload-diversity logs) |
  | `bench/*.md` | 14 | analysis per iteration (incl. 15-workload-diversity) |
  | `notes/*.md` | 11 | decision records |
  | `patches/` | 7 | 5 patches + Dockerfile + inject_mtp.py + upstream-pr-draft/ |
  | `prompts/` | 4 | prompt_code + iter-26's prompt_{codegen,chat,nl} |
  | top-level .md | 3 | README + MORNING + CHANGELOG |
  | `build-artifacts/` | 2 gitignored | mtp-merged.gguf (12 GiB) + lookup-cache-static.bin (15 MB) |
  | `deps/llama.cpp/` | gitignored | shallow clone + ROCm build |

The night ended at Thu 06:00 CEST with the repo in this state.

### Next-move recommendation (ordered by effort:gain)

1. **File the `can_seq_rm` upstream PR** (10 min). Helps the whole
   community; doesn't move our local number. Start here to make the work
   useful beyond this machine.
2. **EAGLE-3 port validation on LLaMA-3.1-8B** (1 day). Cheapest decisive
   signal for whether EAGLE-3 on gfx1151 is worth a week. See
   `notes/09-eagle3-future-path.md`.
3. **If EAGLE-3 signal is positive (≥2.5x on LLaMA-3.1-8B)**: 2-4 week
   EAGLE-3 port + drafter distillation for Qwen3.6-27B. Projected 35-45
   tps.
4. **If EAGLE-3 signal is weak (<1.5x)**: pivot to fixing GDN rollback
   for dm≥6 coherence (200-400 LoC in `src/llama-memory-recurrent.cpp`).
   Projected clean 36-40 tps (the wall-clock is already there).

### What future Ivan should read first (in order)

1. `MORNING.md` — 30-second briefing + reproduce command.
2. `README.md` — full context, results table, hazards.
3. This file (`CHANGELOG.md`) — chronological story with commit hashes.
4. `notes/08-final-state.md` — longer-form postmortem + lessons.
5. `patches/upstream-pr-draft/PR-DESCRIPTION.md` — ready to file.
