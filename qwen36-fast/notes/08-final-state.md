# Final state & postmortem — Qwen3.6-27B fast-decode on Strix Halo

Run: 2026-04-23 00:20 - 04:30 CEST (~4h real, 16 iterations).
Orchestrator log: `/tmp/timeboxed-qwen36-fast-1776896422.md`.

## 1. One-sentence answer

Night ended at **30.05 tps** (2.17x over Q2_K_XL baseline) via n-gram
lookup speculative decoding; the 40 tps target was **not** reached, and
on 256 GB/s LPDDR5x hardware it is unlikely to be reached without real
engineering work (hundreds of LoC in `llama_memory_recurrent`, or an
EAGLE-3 port, or a trained lightweight drafter — none of which is a
one-day task).

## 2. Timeline vs phased plan

The original plan had three phases (orchestrator snapshot in
`notes/00-context.md`). What actually happened:

| Planned phase | Planned work | What happened |
|---|---|---|
| Phase 1 (0-2h): baseline + vanilla spec | pull 27B Q4, baseline, draft-model spec | Baseline done (iter-5). Draft-model spec blocked (no vocab-matching drafter). Pivoted to lookup. |
| Phase 2 (2-5h): Gated DeltaNet kernel work | reference impl + Vulkan shaders | **Skipped entirely.** Pivoted to the spec-decode rollback bug after iter-6 showed gibberish output. Way higher ROI than kernel work. |
| Phase 3 (5-6h): drafter training / DFlash plan | MTP wiring or EAGLE-3 sketch | MTP wiring partially done via PR #20700 port; DFlash plan deferred as not useful for this hardware. |

The pivot away from Vulkan/GDN kernel work was the right call. The
bottleneck was never the kernels (we hit 64-72% of bandwidth ceiling
across all quants — that's within range for a well-tuned llama.cpp
build on ROCm 7.2). The bottleneck was the spec-decode plumbing
refusing to engage on hybrid-recurrent architectures.

## 3. What we tried and what happened

### Iter 1-4: Setup + reconnaissance

- Bootstrapped `/home/everlier/code/mlm/qwen36-fast/`.
- Downloaded Unsloth Qwen3.6-27B-GGUF (Q4_K_M, Q3_K_S, UD-IQ3_XXS,
  UD-Q2_K_XL). Q4_K_M = 16.8 GiB, Q2_K_XL = 11 GiB.
- **Found**: Unsloth's GGUF convert does NOT preserve the `mtp.*`
  tensors from the HF safetensors. Qwen3.6's root arch
  `Qwen3_5ForConditionalGeneration` wraps the LM under
  `model.language_model.*` which the text-only convert path ignores for
  anything outside the standard block list.
- **Found**: HF safetensors DO contain all 15 `mtp.*` tensors
  (`mtp.fc`, `mtp.layers.0.*` full attention block, `mtp.norm`, two
  pre-fc norm tensors). Config: `mtp_num_hidden_layers: 1`,
  `mtp_use_dedicated_embeddings: false` (shares embed + lm_head with
  backbone).
- **Found**: llama.cpp has full Qwen3.6 support
  (archs `qwen35`, `qwen35moe`; converters + model builders upstream).
  An `LLM_TENSOR_NEXTN_*` scaffold exists (for DeepSeek/GLM4/etc.) but
  no `SPECULATIVE_TYPE_MTP` in `common/speculative.cpp`, and
  `convert_hf_to_gguf.py` explicitly does
  `if name.startswith("mtp"): return  # ignore MTP layers for now`.
- No <=1B Qwen3.6 sibling exists; Qwen3-0.6B has a 151,936 vocab
  (Qwen3.6 is 248,320) and a different arch - llama.cpp refuses to
  load it as a draft.

Lessons: always gguf-dump before assuming a drafter tensor survived
conversion; Qwen VL wrappers confuse text-only converters.

### Iter 5-6: Baseline + first spec attempt

- Q4_K_M baseline: 10.87 tps @ d=2048 (10.67 @ d=7000). ~70% of
  weights-only bandwidth ceiling. Decode barely degrades with context
  (FA on).
- Gap analysis: need 3.7x over Q4_K_M for 40 tps, or 2.9x over Q2_K_XL.
- First lookup-spec attempt (iter-6): **gibberish output** despite
  reported 13-50 tps wall-clock. 58-196 "decode failed" errors. Error
  string pointed at M-RoPE `X < Y` strict ordering in
  `src/llama-batch.cpp`.

Lessons: ROCm 7.2 HIP on gfx1151 is fine - 70% of BW ceiling is normal.
Quant choice matters more than kernels here.

### Iter 7: Diagnose

- Pinpointed the real cause: **NOT** the M-RoPE check directly, but
  `llama_memory_recurrent::seq_rm` refusing partial rollback by design.
- Qwen3.6 is hybrid (Gated DeltaNet 3:1 Gated Attention). GDN layers
  have recurrent state. During spec-verify of K draft tokens, the
  recurrent state advances through all K. On partial reject, `seq_rm`
  silently fails -> memory desyncs -> next ubatch trips the downstream
  M-RoPE min/max_pos check.
- Wrote a diagnostic-only patch
  (`patches/llamacpp-mrope-spec-decode-fix.patch`) that disambiguates
  the error message. Does NOT enable spec decode.

Lessons: an assertion failing at site X does not mean X is the bug.
Trace the state machine backwards to where the invariant was violated.
The M-RoPE check was a downstream indicator of recurrent desync.

### Iter 8: Quant sweep

- Q3_K_S: 13.48 tps (+24%).
- UD-IQ3_XXS: 11.98 tps (+10%). Worse than Q3_K_S of similar size =>
  dequant kernel on gfx1151/ROCm 7.2 is unoptimised for IQ formats.
- UD-Q2_K_XL: 13.82 tps (+27%). Best standalone.
- UD-Q2_K_XL with FA off: 13.06 tps (-5.5%). We are more bandwidth-bound
  than compute-bound, so FA's benefit is small.
- Gap narrows: at Q2_K_XL need 2.89x for 40 tps (was 3.68x).

Lessons: UD-quants are worth testing; IQ quants are NOT universally
better than K quants on ROCm (backend-dependent). FA is a 5% win here,
not the "40% speedup" marketing quote from CUDA attention-bound regimes.

### Iter 9-10: Upstream survey

- **PR #19493** ("server : speculative checkpointing", +420/-179) was
  merged upstream 2026-04-19. Our 0d0764d clone already has it. This
  adds a checkpoint ring to `llama_memory_recurrent` for prompt-cache
  reuse and is orthogonal-but-related to spec-decode rollback.
- **PR #20700** ("feat: MTP support for dense Qwen 3.5 with FastMTP") is
  OPEN, WIP, CONFLICTING as of 2026-04-21. Author: itigges22 (first-time
  contributor). 890 LoC across 20 files. Targets Qwen3.5, not 3.6.
  Author claims 82% alpha on 9B; community reports 27B word-salad.
- **PR #21437** (EAGLE3 stacked on #18039) is actively iterated, not
  merged.
- **PR #22105** (DFlash) is experimental, vLLM-only still on
  llama.cpp side.
- Long-ctx Q2_K_XL: 13.82 (2k) -> 13.01 (16k) tps = -5.9%. Hybrid GDN
  keeps ctx cost nearly flat.
- **Identified**: `common_context_can_seq_rm` in `common/common.cpp`
  probes `seq_rm(0, 1, -1)` after 2-token warmup. Qwen3.6 GDN fails
  this probe -> server defensively disables spec decode entirely
  ("speculative decoding not supported by this context"). Fix: relax
  the probe to also try checkpoint round-trip; promote NO -> FULL when
  checkpoint works.

Lessons: always survey the PR board for your problem BEFORE patching -
we nearly wrote 300 LoC that upstream had merged four days prior.

### Iter 11: Patch + build

- Wrote `patches/llamacpp-qwen36-spec-decode.patch` (~30 LoC in
  `common/common.cpp`): if raw seq_rm fails, try a checkpoint
  round-trip; promote NO -> FULL when checkpoint works.
- Built on top of 0d0764d in Docker (`kyuz0/amd-strix-halo-toolboxes:rocm-7.2`,
  `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`). Build succeeded ~15 min.

Lessons: build in the same image you will run in. Our first attempt
used `ghcr.io/ggml-org/llama.cpp:server-rocm` which lacked
`amdclang++`. The kyuz0 toolbox has the full ROCm toolchain.

### Iter 12 (parallel): PR #20700 deep-dive

- Read every hunk of PR #20700 (`notes/07-pr20700-port-plan.md`).
- Qwen3.5 <-> Qwen3.6 MTP layout is **identical** at GGUF level: same
  arch name, same 15 `mtp.*` tensor names, same shapes (64 layers,
  5120 hidden, 248320 vocab). Converter should work as-is.
- Known bug report in PR: AImindPalace sees word-salad on Qwen3.5-27B
  Q6_K on Metal. Hypothesized: `q_proj` shape mismatch. Verified on our
  shards: shape is `[12288, 5120]` which IS the gated form
  (24 heads * 256 dim * 2). Hypothesis disproved.
- Port delta for Qwen3.6: minimal, ~70-90 LoC at most.

Lessons: when reading someone else's WIP, verify the stated bugs
against source-of-truth data before replicating their "fix". The
reporter's hypothesis was wrong.

### Iter 13 (BREAKTHROUGH): patched llama-lookup works

- Ran `llama-lookup -m Qwen3.6-27B-UD-Q2_K_XL.gguf -ngl 99 -fa on -f
  prompt_code.txt -n 256 --draft-max 4`.
- **30.05 tps, alpha=65.29%, coherent output.**
- 90 M-RoPE "decode failed" log lines, but PR #19493's
  checkpoint-restore-retry handles each one transparently. User never
  sees the errors in the output.
- Key observation: our iter-11 patch is COMPILED IN but DORMANT for the
  llama-lookup binary (which doesn't call `common_context_can_seq_rm`).
  The patch was right but the path we validated was different from the
  one the patch targets. The underlying hypothesis - that
  checkpoint-based rollback works on hybrid GDN at `--draft-max 4` - is
  confirmed.

Lessons: don't assume your patch is what made something work. Trace
the actual call graph. A binary that links a patched lib may skip the
patched function entirely.

### Iter 14-15: MTP path (PR #20700)

- Applied PR #20700 to local tree. 4 merge conflicts, all trivial.
- Built via custom builder image (`patches/strix-halo-builder.Dockerfile`)
  on top of `ghcr.io/ggml-org/llama.cpp:server-rocm` since it has
  amdclang++ pre-installed.
- Merged MTP tensors into Q2_K_XL GGUF via `patches/inject_mtp.py`.
  First attempt used F16 for all MTP tensors -> hit
  `GGML_ASSERT(nb10 % sizeof(src1_t) == 0)` in `binbcast.cu` during
  warmup decode. Root cause: backbone norms are F32; F32 x F16 mixed
  mul tripped the assertion. Fixed by writing norm tensors as F32,
  matmul tensors as F16. Final GGUF: 11.83 GiB, 866 tensors
  (865 = 64*13+standard + 1 extra MTP block).
- **MTP K=1 result: 7.80 tps, alpha=1.00.** A 35% regression vs the
  11.91 tps plain decode on the merged GGUF (which itself is 14% lower
  than the non-MTP Q2_K_XL baseline because the 65th MTP block runs
  every step).
- Economics: plain step 83 ms; K=1 step 128 ms. Extra 45 ms = MTP
  forward + 2-token verify ubatch on the 27B backbone. On 256 GB/s
  memory we're at 64-70% of BW, no idle bandwidth for MTP to hide
  behind. On CUDA datacenter GPUs with >1 TB/s this flips positive;
  the PR author reports 82% alpha wins on RTX 5060 Ti.
- MTP K>=2 is silently capped by `params_base.speculative.n_max = 1`
  hardcoded at `tools/server/server.cpp:1309`. Not a config we can
  override via CLI.

Lessons: dtype mismatch between injected tensors and base model is an
under-tested failure mode. When in doubt, match the backbone's dtype
exactly (`ggml-dump` the source GGUF first). MTP / EAGLE / any
per-step drafter pays its overhead every token; if the base decode
is already bandwidth-saturated, K=1 savings cannot outpace the
overhead. This is an architectural truth, not a bug.

### Iter 18: Definitive MTP structural limitation

After iter-15 we hypothesised that the MTP K>1 regression could be
unlocked by removing an alleged `n_max=1` hardcode at
`tools/server/server.cpp:1309`. Iter-18 investigated this claim and
**ruled it out**:

- The actual line `backend_sampling &= !(slot.can_speculate() &&
  task.params.speculative.n_max > 0)` is a **boolean gate** that
  disables backend sampling when spec is active, NOT a clamp on
  draft length.
- The real structural cap is in `common/speculative.cpp:603-649`:
  `common_speculative_state_mtp::draft()` calls
  `llama_get_mtp_logits()`, argmaxes **one** vocab-sized row, and
  pushes exactly one token. `params.n_max` is declared
  `GGML_UNUSED`. The function body is not a `for (k < n_max)` loop.
- The MTP logits tensor itself is shaped as one vocab row by
  construction at `src/llama-context.cpp:1819-1835`. One forward
  pass returns one candidate. There is no K-dim.
- Verified empirically on the PR #20700 build: `--draft-max ∈
  {1,2,3,4}` all produce bit-identical output, tps ≈ 7.6,
  alpha=1.00, `draft_n=127, accepted=127`. **`--draft-max` is a
  no-op for MTP.**

**Definitive conclusion**: MTP on Qwen3.6 in llama.cpp as implemented
by PR #20700 is **structurally single-token**. This is not a
hardcode or a one-line bug. Unlocking K>1 would require:

1. ~400-600 LoC across `src/models/qwen35.cpp`, `src/llama-graph.h`,
   `src/llama-context.cpp`, `common/speculative.cpp`,
   `include/llama.h` to produce and consume a K-row logits tensor.
2. **A retrained MTP head for shift-k lookahead.** The released head
   was trained to predict token at position t+1 given hidden state
   at t. There is no training signal for predicting t+2..t+K from
   the same hidden state; you'd need either a different head per
   shift (K heads), or a single head trained with shift-k targets,
   or an autoregressive cascade that uses the head's own t+1
   prediction as input for t+2 (which re-introduces per-step
   overhead and defeats the bandwidth economics).

**On bandwidth-bound hardware (256 GB/s), even if K=4 MTP cascaded
cleanly, the per-step MTP forward cost would eat most of the savings.
The released MTP head cannot beat the iter-13 lookup result (30 tps)
on this APU without also being re-architected lighter. EAGLE-3 (PR
#21437) is structurally better-suited because its draft head is
detached from the backbone forward and can be quantized/shrunk
independently.**

Patch `patches/llamacpp-unlock-mtp-k.patch` is empty-by-design — it
documents the structural reason no patch was written.

### Iter 16: Lookup tuning sweep

- Built a 15 MB static lookup cache from 10 MB code corpus (lifeos +
  harbor + llama.cpp source) via `llama-lookup-create`.
- Swept `--draft-max` in {2, 3, 4, 5, 6, 7, 8, 10, 12, 16} * static /
  dynamic / both * temp=0.8 / greedy T=0. 40+ configs.
- **Best clean: 31.13 tps** @ dm=5 `--draft-min 2` dynamic. +3.6% over
  iter-13. **Marginal.**
- **Cleanest: 29.02 tps** @ dm=5 greedy dynamic, alpha=0.92.
- dm >= 8 reaches 36-52 tps wall-clock but output is
  degenerate token loops (" wants wants", "111111"). Unusable.
- **Static cache consistently UNDERPERFORMS** when corpus doesn't
  overlap the prompt (alpha drops 0.92 -> 0.15-0.50). Our generic
  code corpus had no overlap with the reviewed Python module.
- **Alpha does NOT generalize**: 0.92 on code-review (self-referential
  prompt), 0.61 on NL-QA, 0.08 on cold-start code-gen. Lookup is a
  workload amplifier, not a universal speedup.
- **Environmental note**: building `llama-lookup-create` during this
  iteration relinked `libllama-common.so.0` to the PR #20700
  artifact (different lib than iter-13). iter-13's binary is dormant
  for MTP plumbing but the relinked lib introduces `[MTP-FINDSLOT]` /
  `[MTP-SEQRM]` paths that add diagnostic noise and a new failure
  mode where seq_rm rollback without checkpoint produces token loops
  at high dm.

Lessons: ninja incremental builds silently relink shared deps; keep
baseline binary + its lib preserved. Never trust a "marginal gain"
sweep that also changes the lib under the hood. The iter-13 regime
may not be byte-identical re-reproducible today; the 29-31 tps range
is what holds.

## 4. Hard numbers table

| Metric | Value | Source |
|---|---:|---|
| Q4_K_M baseline d=2048 | 10.87 tps | iter-5 |
| Q3_K_S baseline | 13.48 tps | iter-8 |
| UD-Q2_K_XL baseline | 13.82 tps | iter-8 |
| UD-Q2_K_XL + lookup dm=4 (iter-13) | 30.05 tps | iter-13 |
| UD-Q2_K_XL + lookup dm=5 dyn greedy | 29.02 tps | iter-16 |
| UD-Q2_K_XL + lookup dm=5 dmin=2 | 31.13 tps | iter-16 |
| MTP K=1 on merged GGUF | 7.80 tps | iter-15 |
| Plain decode on merged GGUF | 11.91 tps | iter-15 |
| **Target** | **40 tps** | orchestrator |
| **Gap** | **-10 tps (-25%)** | |
| Speedup achieved (clean) | 2.17x over Q2_K_XL baseline | iter-13 |
| Speedup achieved (clean) | 2.76x over Q4_K_M baseline | iter-13 |

## 5. Ruled out (with evidence)

- **MTP K=1 via PR #20700 on Strix Halo: NET REGRESSION.** 7.80 tps vs
  11.91 tps plain on same GGUF. Per-step cost arithmetic is
  unfavourable on 256 GB/s memory. (iter-15)
- **IQ-quants are not worth it here**: UD-IQ3_XXS at 11.98 tps is
  slower than Q3_K_S at 13.48 tps despite similar size. ROCm 7.2 HIP
  dequant for IQ3 is unoptimised on gfx1151. (iter-8)
- **dm >= 8 in lookup is not a speed knob**: wall-clock rises to
  50 tps but output is token loops. Do not use. (iter-16)
- **Draft-model spec decode**: no vocab-matching <=1B Qwen3.6 exists.
  Qwen3-0.6B rejected by llama.cpp vocab compat check. (iter-2)
- **Static lookup cache from a generic corpus**: hurts when prompt
  doesn't match the corpus. Only helps for workloads with verbatim
  overlap against the cached corpus. (iter-16)
- **Harbor for 27B running**: the compose template has argv mishandling
  for ROCm images; direct `docker run` is the workaround. (iter-5)

## 6. Still open / plausible future paths

Ordered by effort:gain tradeoff (revised after iter-18 ruling):

1. **EAGLE-3 port (PR #21437 on top of #18039)** — NOW the top path.
   PR #18039 adds base EAGLE3 infra + LLM_ARCH_EAGLE3 + encoder/decoder
   graph (`src/models/eagle3.cpp`); PR #21437 extends it with hybrid
   recurrent-state support and `qwen35` / `qwen35moe` integration
   (2889-line diff, 34 files). Drafter head is a 1-layer transformer,
   detached from the backbone forward → per-step cost is a fraction of
   the 27B backbone read. On bandwidth-bound hardware this is
   structurally the right shape. **No Qwen3.6-27B EAGLE3 drafter
   exists on HF** — training one is part of the work. See
   `notes/09-eagle3-future-path.md` for a sequential recipe.
   Projected: alpha ~= 0.55-0.70 → 35-45 tps on Q2_K_XL.
   Effort: 2-4 weeks (port + drafter training + hybrid tuning).

2. **Fix GDN rollback for dm>=6 coherence.** 200-400 LoC in
   `src/llama-memory-recurrent.cpp`. Projected gain: clean 36-40 tps
   (the wall-clock is already there, just degenerate). 3-5 days incl.
   multi-backend regression.

3. **Prompt-specific static lookup caches** for repeat-heavy workloads
   (editing one codebase, doc rewriting, code review over a fixed
   corpus). Glue work, hours. Projected: 33-36 tps on matched prompts.

4. **MTP-head retraining for shift-k lookahead** — **the only
   MTP-path forward** after iter-18. Not a llama.cpp patch; a
   research project:
   - 400-600 LoC in llama.cpp to produce and consume K-row logits
     from one MTP forward (graph + `llama_get_mtp_logits` + speculative).
   - Retrain the MTP head with shift-k targets (K separate heads, or
     a single shift-conditioned head). Requires access to Qwen's
     training corpus or a good distillation proxy.
   - Per-step MTP overhead on this APU (~45 ms at K=1) likely keeps
     wins below EAGLE-3. Documented for completeness.
   Projected: 12-18 tps at best. Effort: months.

5. **Lightweight custom MTP.** Same burden as above minus the shift-k
   head research, plus training a small-footprint drafter. Worse
   upside than EAGLE-3 for similar effort. Projected: 12-15 tps.

6. **vLLM-ROCm migration**. Different inference stack entirely. DFlash
   (PR #22105) in llama.cpp is WIP / conflicting and, per the PR's
   own benchmarks, hybrid target speedup is intrinsically limited by
   recurrent-state rollback overhead. z-lab publishes
   `z-lab/Qwen3.5-27B-DFlash` and `z-lab/Qwen3.6-35B-A3B-DFlash` but
   **not** `z-lab/Qwen3.6-27B-DFlash` — same drafter-weights gap as
   EAGLE-3. Weeks of porting work + Harbor integration rework.

7. **Port iter-11 server-side test.** Our can_seq_rm patch was never
   exercised — llama-lookup skips the probe. Running
   `llama-server --spec-type ngram-cache` against the patched binary
   would validate the patch fires. Low-risk, hours of work. Doesn't
   improve tps but closes the verification loop.

8. **Wait for PR #20700 to mature** — **DEMOTED after iter-18.**
   Even if the PR merges cleanly, MTP is structurally single-token in
   llama.cpp's current shape. Merging PR #20700 as-is does not unlock
   K>1 and does not unlock tps on this hardware. Treat #20700 as a
   correctness milestone for CUDA/MTP users, not a speedup for us.

## 7. Lessons for Ivan's future LLM work on Strix Halo

### Hardware / stack lessons

- **Bandwidth is the ceiling, not compute.** Q4_K_M decode hits
  ~10.9 tps on Q4_K_M = ~70% of the 15.6 tps weights-only ceiling at
  256 GB/s. This is a mature-kernel number; you will not meaningfully
  beat it with better kernels alone. Speedup must come from fewer
  weight reads per output token (quant reduction) or parallel
  verification (spec decode).
- **Unified-memory APU != discrete-GPU.** CUDA/datacenter wisdom
  ("MTP K=1 wins", "FA gives 40%", "IQ-quants are smaller") maps
  poorly here. Verify every performance claim on this hardware.
- **gfx1151 ROCm 7.2 kernels are ~70% of ceiling for K-quants,
  ~56% for IQ-quants.** Prefer K-quants (Unsloth UD series) until
  someone writes optimised HIP dequant for IQ3/IQ4.
- **FA gives 5% on this hardware, not 40%.** Still on by default,
  but don't expect the advertised CUDA gains.

### Spec-decode lessons

- **Hybrid Gated DeltaNet / recurrent state + spec decode is a
  known-hard problem upstream.** PR #19493 merged the necessary
  checkpoint infra 2026-04-19, PR #20700 is the MTP consumer (WIP).
  Any hybrid arch (qwen35, qwen35moe, qwen3next, glm4moe+mrope,
  lfm2, mamba-base, plamo2, kimi-linear) shares this trap.
- **Lookup / n-gram spec works** as of 0d0764d (with #19493 in tree)
  on hybrid arches, up to `--draft-max 5` clean. dm>=6 degenerates.
- **alpha is workload-dependent for lookup.** Code-review with
  verbatim repetition hits 0.92; cold-start code-gen hits 0.08. Do
  not quote a single alpha number; quote a prompt type.
- **MTP K=1 on bandwidth-bound HW is net-negative** even at
  alpha=1.00. Needs K>=3 cascade or a lighter drafter head.
- **Dtype mismatch in merged GGUFs** (F32 backbone norms + F16 MTP
  norms) trips CUDA binbcast alignment. Always match backbone dtype.

### Engineering lessons

- **Survey upstream PRs first.** PR #19493 saved us from writing 300
  LoC that had been merged 4 days earlier. One hour of `gh pr list
  --search "memory recurrent rollback"` is worth a day of coding.
- **Trace state backwards.** M-RoPE assertion at site X was a symptom
  of recurrent desync at site Y three call frames earlier. Read the
  call graph, don't patch the assertion.
- **Never trust "incremental ninja build is harmless".** Relinking
  libllama-common.so.0 silently changed the regime between iter-13 and
  iter-16. If you ran a canonical benchmark, freeze the binary.
- **Test with the binary you actually run.** Our patch compiled in but
  didn't fire - the CLI binary skipped the probe. Validating the patch
  needed the server binary.
- **Don't assume drafters/tensors survive format conversion.** Unsloth
  GGUF dropped Qwen3.6's `mtp.*` tensors because of VLM wrapper
  arch name. `gguf-dump` every converted artifact.
- **Know your bandwidth economics.** Plain decode 83 ms/token.
  Adding 45 ms for MTP head = 128 ms/token. At alpha=1.00 you save
  1 real token per draft: 128/2 = 64 ms/accepted-token. Baseline
  was 83 ms -> 64 ms is technically a win in CPU-time-per-accepted but
  LOSES in wall-clock because baseline isn't bandwidth-idle. Work
  out the arithmetic on paper before running the merge.

### Process lessons

- **Pivot early when Phase 2 is the wrong problem.** GDN kernel work
  would have been a week of effort for maybe 5% gain; spec-decode
  unblock was 2 hours for 120% gain.
- **Write the numbers down as you go.** Each iteration notes file
  made the final writeup tractable. Without those, reconstructing the
  run at 04:00 CEST would be impossible.
- **Separate patches from experiments.** `patches/` contains only the
  source-diff artifacts (applicable via `git apply`); `bench/` is
  logs; `notes/` is decision records. Future Ivan can reproduce.
- **Preserve the prompts with the repo.** iter-13's 30 tps requires the
  exact 1766-token code-review prompt. If it only lives in `/tmp`,
  the result isn't reproducible. Copied to `prompts/prompt_code.txt`.

## 8. For the morning review

Start here:
1. `README.md` - 1-minute summary + results table.
2. `./bench/run-best.sh --short` - verify the pipeline still works
   (~20 s, expect 15-25 tps due to short-decode cache warmup).
3. `./bench/run-best.sh` - full reproduction (~50 s, expect 28-31 tps).
4. This file - full postmortem if you want the whole story.

If you want to continue this work, the single highest-ROI next step
is **fix GDN rollback for high draft-max coherence** (path 1 in
Section 6). It unlocks tps that's physically present but currently
wasted to output-quality degradation.
