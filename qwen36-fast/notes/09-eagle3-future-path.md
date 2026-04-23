# EAGLE-3 future-path research (Qwen3.6-27B on Strix Halo)

Research-only note. Written 2026-04-22 in iter-19 after the definitive
MTP ruling (iter-18). No code applied; no benchmarks run. This is a
planning document for a future dedicated timebox focused on EAGLE-3,
which iter-18 identified as the single most promising remaining path
to 40 tps on Qwen3.6-27B on Strix Halo / gfx1151.

## 1. Primary candidate: PR #21437 on top of PR #18039

### PR #21437 — `eagle3: add qwen3.5 4B 9B 35B-A3B support`

- URL: https://github.com/ggml-org/llama.cpp/pull/21437
- State: **OPEN**, `mergeable=UNKNOWN`, `reviewDecision=REVIEW_REQUIRED`
- Author: `36330` (first-time-ish contributor; not a recognised
  maintainer). Zero maintainer reviews, two non-maintainer questions.
- Size: **+1764 / −96** across **34 files**.
- Branch: `pr/eagle3-more`.
- Created: 2026-04-04. Last update: 2026-04-04 (**stale**, author
  pushed once, never returned).
- Top reviewer signal: `ngxson` (contributor) commented "You're
  pushing to wrong branch, I suppose this is not yet ready." Author's
  response: "I successfully merged the latest version." — no follow-up.
- Automated `ggml-gh-bot` flag: "Large PR — maintainers may not be
  able to review this PR as-is. Consider splitting." (Same flag
  fires on PR #18039.)
- **Merge ETA: unclear, but likely months behind #18039, which is
  still itself open.** Treat both as patches to apply locally.

### PR #18039 — base `[Speculative decoding] feat: add EAGLE3 speculative decoding support`

- URL: https://github.com/ggml-org/llama.cpp/pull/18039
- State: **OPEN**, `mergeable=CONFLICTING`, `reviewDecision=REVIEW_REQUIRED`
- Author: `ruixiang63` (Ruixiang Wang; same author as DFlash PR #22105,
  plausibly the current SOTA spec-decode contributor to llama.cpp).
- Size: **+1137 / −53**.
- Created: 2025-12-14. Actively iterated through 2026-04-20.
- This PR is **the** EAGLE3 base in llama.cpp. It:
  - Adds `LLM_ARCH_EAGLE3` to the architecture enum.
  - Adds `src/models/eagle3.cpp` with `llm_build_eagle3_encode` +
    `llm_build_eagle3_decode` (1-layer transformer decoder).
  - Adds feature extraction from target layers via a new callback.
  - Adds `d2t` (draft→target vocab) mapping tensor.
  - Adds `--eagle3` CLI flag in `common/arg.cpp`.
  - Wires EAGLE3 into `examples/speculative-simple/` only (not
    `llama-server`).
- Qwen3 (original, not hybrid) support present. LLaMA 3.1 / 3.3 /
  gpt-oss-20b/120b support present. **Qwen3.5 explicitly NOT
  supported**, which is exactly why #21437 exists.
- Claimed numbers (author's, RTX A6000 48 GB):
  - LLaMA 3.1-8B BF16: 44.5 → 146.2 t/s, alpha=0.81, 3.28× speedup.
  - Qwen3-8B BF16: 43.6 → 94.8 t/s, alpha=0.70, 2.17×.
  - Qwen3-14B BF16: 24.4 → 35.7 t/s, alpha=0.40, 1.46×.
  - Qwen3-32B Q4_K_M: 32.0 → 39.7 t/s, alpha=0.40, 1.24×.
  - Qwen3-30B-A3B BF16 on DGX Spark: 31.1 → 43.3 t/s, alpha=0.64,
    1.39×. **This is the closest analogue to our target machine**
    (DGX Spark is also an integrated-memory system at 256 GB/s).

### PR #21437 mechanism (what #21437 adds on top of #18039)

EAGLE3 fundamentals (both PRs):

- **Draft head = 1-layer transformer** plus an FC fusion layer and
  an output projection to a compact draft vocab (typically 32k-64k,
  remapped via `d2t` to the full target vocab).
- Target model forward emits features from **3 specific layers**
  (`eagle3_extract_0/1/2` callbacks in PR #21437's `src/models/qwen35.cpp`
  diff). These are the encoder inputs to the draft head.
- Draft head runs **autoregressively for K steps**, each step
  conditioned on accumulated target features + its own previous
  draft tokens — classic EAGLE recurrence, not single-shot like
  DFlash and not single-token-locked like MTP.
- Target verifies the K-token candidate block in one batched decode.
  Accepted prefix is committed; rejected suffix triggers rollback.

What #21437 adds specifically for Qwen3.5 / hybrid architectures:

- `eagle3_round_cells` + `has_eagle3_round` plumbing in
  `llama_memory_recurrent` / `llama_memory_hybrid`: a per-token
  mapping of verify-round cells so the recurrent state of
  Gated-DeltaNet layers can be committed cell-by-cell, then rolled
  back to the accepted prefix **without replaying the whole block**.
- Per-head conv-state writes into `conv_states_all` view-by-view,
  gated by `has_eagle3_round`, in `src/models/qwen35.cpp` and
  `qwen35moe.cpp`.
- "Recurrent round-state APIs" on `include/llama.h` and
  `src/llama-memory.h` — this is where the PR does the real work;
  not trivial to extract from.

So #21437 is the **hybrid-recurrent-correct** EAGLE3 integration.
Without it, EAGLE3 on Qwen3.5/3.6 would hit the same rollback
desync we saw with lookup at dm>=6.

### K-lookahead story (comparison)

| Method | How many draft tokens / step | Why |
|---|---|---|
| MTP (PR #20700, current) | **1** (structural) | `common_speculative_state_mtp::draft()` argmaxes one logits row; MTP graph emits single-row logits (`src/llama-context.cpp:1819-1835`). |
| EAGLE-3 (#18039 / #21437) | **K configurable** (`--draft 8` typical) | Draft head is a full autoregressive 1-layer transformer; runs K times per verify. |
| DFlash (#22105) | **block_size (up to 16)** in one shot | Masked-token block diffusion; multi-layer draft in a single forward. |
| Lookup (upstream) | up to `--draft-max` | N-gram suffix match; no model. |

EAGLE-3's K is a real knob. That is the single most important
difference from the MTP we ruled out in iter-18.

## 2. Qwen3.5 vs Qwen3.6 support

- **Qwen3.5 and Qwen3.6 share the `qwen35` arch in llama.cpp.**
  The stock master already loads Qwen3.6-27B and Qwen3.6-35B-A3B via
  the same converters and model builders (iter-4 finding).
- **Dimensions are identical at the LM level:**
  - Qwen3.6-27B `text_config`: hidden=5120, intermediate=17408,
    layers=64, vocab=248320.
  - Qwen3.5-27B (and `z-lab/Qwen3.5-27B-DFlash` drafter config):
    hidden=5120, intermediate=17408, layers=64, vocab=248320.
  - Qwen3.6-27B's root arch is `Qwen3_5ForConditionalGeneration` (VLM
    wrapper), but the language tower under `model.language_model.*`
    is bit-for-bit a Qwen3.5 tower.
- **PR #21437's `src/models/qwen35.cpp` edits apply directly** to
  Qwen3.6-27B decode — same graph, same recurrent layers.
- **Port delta for Qwen3.6 = near-zero in llama.cpp itself.** The
  real delta is the drafter weights (see §3).

## 3. Drafter weights availability — **THE blocker**

Searched HF namespaces: `z-lab`, `hf-eagle`, `sglang-project`,
`AngelSlim`, `Tengyunw`, `yuhuili`, `lmsys`, `RedHatAI`,
`Zjcxy-SmartAI`, plus general searches for
`Qwen3.6+eagle` / `Qwen3.6+drafter` / `Qwen3.6+speculator` /
`Qwen3.6+EAGLE`.

Findings:

- **No EAGLE-3 drafter for Qwen3.6-27B exists on HF** as of 2026-04-22.
- **No EAGLE-3 drafter for Qwen3.5-27B exists either.** AngelSlim's
  Qwen3 EAGLE3 collection covers 1.7B / 4B / 8B / 14B / 32B /
  30B-A3B — not the 27B dense shape. Zjcxy-SmartAI has 4B / 8B / 32B
  Chinese variants. `Tengyunw` has qwen3_8b and qwen3_30b_moe.
- Other adjacent drafters:
  - `lmsys/Qwen3-235B-A22B-EAGLE3` — wrong target size.
  - `BLR2/Qwen3.5-9B-Eagle3-ShareGPT` and `BLR2/Eagle3-Qwen3.5-9B` —
    9B, not 27B.
  - `jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge` — 35B MoE, wrong shape.

**Conclusion:** a Qwen3.6-27B EAGLE-3 drafter has to be **trained**.
Options:

1. **Train from scratch using SpecForge (SGLang's drafter-training
   fork).** SGLang has published multiple Qwen3 EAGLE3 drafters
   (`lmsys/SGLang-EAGLE3-Qwen3-*-SpecForge-*`). Pipeline: collect
   feature traces from Qwen3.6-27B on a diverse corpus, train a
   1-layer-transformer drafter head against them. Compute budget:
   ~hundreds of GPU-hours on H100 per published recipe. Walltime:
   days-to-weeks.
2. **Distill from the released Qwen3.6 MTP head.** We already have
   the 15 `mtp.*` tensors. They are a 1-layer full-attention block
   trained on shifted-next-token — not the EAGLE3 feature-fusion
   head, but potentially a good warm start. Novel, untested.
3. **Port `z-lab/Qwen3.5-27B-DFlash` drafter weights** as an EAGLE-3
   initialisation. Same shape (hidden=5120, intermediate=17408,
   5-layer draft, attached to layers [1,16,31,46,61]). DFlash draft
   architecture differs (block-diffusion, not autoregressive), but
   some of the projection layers + input-fusion parts might
   transfer with fine-tuning. Speculative; not documented anywhere.
4. **Train a Qwen3.5-27B EAGLE3 drafter and use it on Qwen3.6-27B.**
   Risk: logits/hidden-state distribution shift between 3.5 and 3.6
   is modest but nonzero. Would need measurement; likely produces
   lower alpha than a 3.6-native drafter but may still win.

## 4. Expected speedup on Strix Halo

Base: UD-Q2_K_XL decode = 13.82 tps (iter-8), target = 40 tps.

EAGLE-3 per-step cost model on this hw:

- Target verify batch of K+1 tokens on 27B Q2_K_XL backbone.
  At K=8, the batch is 9 tokens; decode cost scales roughly linearly
  in batch but bandwidth-bound decode has a small prefill component
  → approx 1.3-1.5× the 72 ms/token baseline = ~100 ms for a
  9-token verify on Q2_K_XL.
- Draft head cost: 1-layer transformer, 5120 hidden, ~0.3-0.5 GB
  weights (F16, quantised). At 256 GB/s: ~2 ms/forward. K=8 forwards
  = ~15-20 ms.
- Rollback overhead on hybrid GDN: with #21437's per-cell state
  management, this should be amortised ~1ms/verify; without it
  (which is the whole point of #21437) it'd be another full target
  forward, blowing the budget.
- Total per verify-step wall: ~120 ms for K=8.
- Acceptance on Qwen3-14B BF16 (closest dense Qwen3 EAGLE3 data
  point in #18039): alpha=0.40, on DGX Spark for Qwen3-30B-A3B:
  alpha=0.64. Project alpha=0.55 for Qwen3.6-27B Q2_K_XL (dense,
  Q2 quant hurts draft agreement, matches our 65% lookup alpha
  ballpark).
- Effective tokens/verify: 1 + 8×0.55 = 5.4.
- Effective tps: 5.4 tokens / 0.120 s = **45 tps**.

This is the upper-end-of-plausible. Realistic range **35-45 tps**
depending on drafter quality + rollback overhead. **Target of 40 tps
is inside this band**, which is why EAGLE-3 is the top remaining
path.

Risks that cut the projection:

- Drafter alpha <0.50 on Q2_K_XL backbone (quant hurts): lands
  at ~30-33 tps, no improvement over lookup.
- Hybrid rollback overhead (PR #21437 might not be correct on
  first apply; recurrent-state APIs are new): lands at 20-25 tps.
- Q2_K_XL drafter verify not numerically stable: might need Q4 or
  BF16 drafter, which increases the 2 ms → 6 ms per forward, still
  fine.

## 5. Concrete next-steps recipe

For a future dedicated timebox focused on EAGLE-3. Sequential, with
effort estimates in working-hours of a skilled ML+C++ engineer.

**Step 1 — Apply both PRs locally (3-5h).**
Clone master. Apply #18039 first (`gh pr diff 18039 | git apply`).
Resolve conflicts (current master has moved since Dec 2025; expect
4-8 conflicts mostly in `common/speculative.cpp`, `common/arg.cpp`,
`src/llama-arch.*`). Then apply #21437's diff. `--3way` merge or
manual. Build in the kyuz0 toolbox image.

**Step 2 — Validate on LLaMA-3.1-8B with existing drafter (2-4h).**
Download `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`. Run the PR's exact
speculative-simple invocation on LLaMA-3.1-8B Q4_K_M. Expect
something like 2× speedup vs baseline LLaMA 3.1-8B Q4 on this
hardware. **Goal: confirm the PR's code path works at all on
gfx1151/ROCm 7.2.** If it doesn't, the rest is moot.

**Step 3 — Validate on Qwen3-8B with existing drafter (2-4h).**
Download `Tengyunw/qwen3_8b_eagle3`. Smoke-test on Qwen3-8B
Q4_K_M. Expect 1.5-2× speedup. **Goal: confirm Qwen3 EAGLE3 path
works on this hardware.** If not, #18039 itself has a ROCm gap.

**Step 4 — Apply iter-11 can_seq_rm relaxation on top (1h).**
The EAGLE3 `speculative-simple` code path will re-use
`common_context_can_seq_rm`. Our existing patch
(`patches/llamacpp-qwen36-spec-decode.patch`) makes hybrid-arch
pass the compat check. Needed for step 6.

**Step 5 — Port PR #21437's Qwen3.5 hybrid hooks review (4-6h).**
Read `notes/07-pr20700-port-plan.md` style review of #21437's
`qwen35.cpp` / `qwen35moe.cpp` / recurrent-state changes. Confirm
they apply cleanly to Qwen3.6-27B (same arch name, same layer
count, same recurrent-layer indices). **Goal: one-paragraph
confirmation that no Qwen3.6-specific code needs to be added.**

**Step 6 — Train a Qwen3.6-27B EAGLE3 drafter (1-3 weeks).**
Use SpecForge pipeline. Feature-extract from Qwen3.6-27B BF16 on
a 1-10 GB diverse corpus (Wikipedia, code, chat). Train 1-layer
transformer EAGLE3 head on an H100 for ~24-72h. Expected output:
a `qwen36-27b-eagle3-drafter.safetensors` of ~500 MB-1 GB.
**This is the critical-path item and the longest.** Cheaper
alternative: reuse `z-lab/Qwen3.5-27B-DFlash` weights by
re-projecting its 5-layer block-diffusion head to EAGLE3's
1-layer autoregressive head (~1 day of adapter training).

**Step 7 — Convert drafter to GGUF + quantise (2-4h).**
Run `convert_hf_to_gguf.py --target-model-dir Qwen3.6-27B` on the
drafter. Quantise to Q4_K_M or keep at BF16 depending on bench in
step 2. Merge/inject into UD-Q2_K_XL backbone if needed (reuse
`patches/inject_mtp.py` pattern — likely a parallel `inject_eagle3.py`).

**Step 8 — End-to-end bench + tune (1 day).**
Run `llama-speculative-simple --eagle3` on Qwen3.6-27B Q2_K_XL + our
drafter. Sweep `--draft {4,6,8,10}`. Measure alpha on
prompt_code.txt + NL-QA + code-gen. Expected:
- Best case: 40-45 tps, alpha ~= 0.60.
- Realistic: 35-40 tps, alpha ~= 0.55.
- Bad case: 22-28 tps (drafter too lossy at Q2). Then try BF16
  drafter + Q4_K_M backbone — loses some tps from backbone but
  keeps drafter accurate.

**Step 9 — Server integration (2-3 days).**
PR #18039 wires only `speculative-simple`. `llama-server` needs
`--eagle3` plumbing + drafter-loading path + spec-decode server
integration parallel to MTP's. Reference: PR #20700's
server-context.cpp pattern. This is table stakes for Harbor.

**Total realistic timebox: 3-5 weeks of one engineer**, with the
drafter training being the single biggest variance (can compress
to 3 days with 8×H100 rented, or stretch to 3 weeks on a single
smaller GPU).

## 6. Secondary option: DFlash / PR #22105

### PR #22105 — `[Speculative decoding] feat: add DFlash support`

- URL: https://github.com/ggml-org/llama.cpp/pull/22105
- State: **OPEN**, `mergeable=CONFLICTING`, `reviewDecision=REVIEW_REQUIRED`
- Author: `ruixiang63` (same as PR #18039). Actively iterated.
- Size: **+1970 / −54**. Built on top of PR #18039 (shares all
  EAGLE3 commits; those will drop from the diff once #18039 merges).
- Created: 2026-04-19. Last update: 2026-04-20. Very fresh.
- Mechanism: block-diffusion drafter. Produces an entire block of
  candidate tokens in ONE draft forward, vs EAGLE-3's K autoregressive
  forwards. Drafter is multi-layer (4-5 transformer layers) vs
  EAGLE-3's single layer — larger weights, but single forward per
  verify step.

### Qwen3.6 applicability

The PR **explicitly calls out hybrid target performance limits**.
Quoting the PR body:

> Speedup is intrinsically limited on hybrid target models...
> Pure-attention target models can drop rejected suffixes with seq_rm;
> hybrid targets cannot, because recurrent state is not decomposable
> by token position.

The PR's workaround is snapshot-before-verify + replay-accepted-prefix,
which costs **one extra target forward per rejection step**. On
Qwen3.5-4B (hybrid, identical-arch to 3.6-27B) the PR reports
`thinking off` alpha=85%/3.36× on quicksort but alpha=9.3%/0.63× on
NL-QA. Highly prompt-dependent, **much more so than EAGLE-3**.

### Drafter availability for Qwen3.6-27B

Searched `z-lab` namespace on HF:

- `z-lab/Qwen3.5-27B-DFlash` **exists** (bf16, 5120/17408/64, same
  dims as Qwen3.6-27B). Shape-compatible with Qwen3.6-27B. May work
  as-is via the Qwen3.5→Qwen3.6 hidden-distribution similarity;
  would need verification.
- `z-lab/Qwen3.6-35B-A3B-DFlash` exists (wrong target size — MoE
  35B, not dense 27B).
- **`z-lab/Qwen3.6-27B-DFlash` does NOT exist.**

So: for DFlash, there's a hope that the Qwen3.5-27B drafter is
close enough to Qwen3.6-27B to work without retraining. This is
an untested hypothesis.

### Expected on Strix Halo

Given the PR's own hybrid-target numbers (Qwen3.5-4B: 0.63× to 3.36×
across prompts, `thinking on`: 0.86× to 1.33×), projecting to our
27B Q2_K_XL: realistic range **8-25 tps** depending on prompt.
This is **worse than our current 30 tps lookup** on most prompts.

Recommendation: **DFlash is a secondary option** to track but not to
prioritise over EAGLE-3. Its hybrid-target speedup is not competitive
with EAGLE-3 + hybrid-rollback (#21437) on per-prompt-worst-case.

## 7. Risks

1. **PR #18039 / #21437 both open and unmerged.** Building on both
   means tracking two stale branches. If maintainers request
   restructure, our local port becomes throwaway.
2. **ROCm 7.2 / gfx1151 may have EAGLE3 kernel gaps.** PR #18039
   uses `GGML_TENSOR_FLAG_SYNC` and `ggml_set_sync()` for a new
   split-barrier pattern. This is CUDA-tested. AMD HIP backend may
   behave differently; needs step-2 smoke test.
3. **Drafter training is expensive and high-variance.** Alpha on
   the first training run may be <0.4, requiring hyperparam tuning
   and more data. Budget 2-3 training attempts, not 1.
4. **Qwen3.5→3.6 drafter transfer is unvalidated.** If steps 6's
   cheap option (reuse 3.5 drafter, fine-tune on 3.6 features)
   works, days saved; if not, weeks of pure training.
5. **Q2_K_XL backbone may be too lossy for the drafter's feature
   extraction to work.** Upstream EAGLE3 numbers are BF16 or Q4_K_M,
   never Q2. Might need Q4_K_M as the floor and sacrifice ~3 tps
   of baseline speed.
6. **Server plumbing (step 9) is real work.** EAGLE3 in the PRs is
   `speculative-simple` only. Harbor uses `llama-server`.
7. **Timebox creep.** A 3-week estimate for one engineer is an
   optimistic lower bound given we are building on two open PRs with
   no maintainer review; more realistic for a from-scratch 27B
   drafter-trained run is 4-6 weeks.

## 8. Single-next-best-move

If Ivan gets one week: **do Step 1 + Step 2 + Step 3**. Apply both
PRs, validate `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` and
`Tengyunw/qwen3_8b_eagle3` work on gfx1151/ROCm 7.2, measure the
actual speedup curves on this hardware. That is the make-or-break
data point: if EAGLE3 gets 2.5×+ on LLaMA 3.1 8B here, then the rest
of the recipe is drafter-training work — which is a known-shape
problem. If it gets <1.5× on LLaMA 3.1 8B on this hw, the whole
EAGLE3 path needs reconsideration and we should reopen the GDN
rollback fix as the top option.

**Cheapest decisive signal: one week, two existing drafters, one
new benchmark script, no training compute. Do that first.**
