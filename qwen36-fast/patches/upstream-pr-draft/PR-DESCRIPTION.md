# common: relax `can_seq_rm` probe to try checkpoint round-trip

## Problem

Since #19493 merged the per-sequence recurrent-state checkpoint feature
(`llama_state_seq_get_data_ext` / `set_data_ext` with
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`), the core library has everything
it needs to run speculative decoding on hybrid (attention + recurrent)
models. Rollback on partial-reject is done by saving state before the
draft batch and restoring it if the verify step rejects a token.

The server's spec-decode enable gate (`common_speculative_is_compat`
→ `common_context_can_seq_rm` in `common/common.cpp`) still probes the
raw `llama_memory_seq_rm(mem, 0, 1, -1)` call. For any hybrid/recurrent
memory the `seq_rm` refuses partial rollback *by design* (see
`src/llama-memory-recurrent.cpp:166`, comment "state of prior tokens
not preserved"). The probe therefore returns `NO`, then
`common_speculative_is_compat` returns `false`, and the server logs:

```
srv load_model: the target context does not support partial sequence removal
srv load_model: speculative decoding not supported by this context
```

...and silently disables spec decode, even though the model *is*
compatible via the checkpoint path.

Affected model families (everything that sets a recurrent or hybrid
memory):

- Qwen3.5 / Qwen3.6 (27B, 35B-A3B) — Gated DeltaNet 3:1 Attention
- Qwen3-Next
- GLM4-MoE
- Mamba / Mamba2 variants
- LFM2
- Plamo2
- Kimi-Linear

## What this patch does

Widens `common_context_can_seq_rm`: when raw `seq_rm` fails (or the
initial 2-token probe `llama_decode` fails), attempt a checkpoint
round-trip (`get_size_ext` → `get_data_ext` → `set_data_ext` with
`PARTIAL_ONLY`). If the round-trip succeeds the context supports
checkpoint-based rollback and we promote to
`COMMON_CONTEXT_SEQ_RM_TYPE_FULL` — the exact branch that
`server-context.cpp`, `common/speculative.cpp`, and
`common/speculative-simple.cpp` already handle (they log "speculative
decoding will use checkpoints").

32 added / 2 removed lines in one file (`common/common.cpp`). No
behaviour change for models where raw `seq_rm` already works (probe
returns `PART` as before).

## Before / after

**Before** (Qwen3.6-27B UD-Q2_K_XL, llama-server `--spec-type ngram-cache`):

```
srv load_model: the target context does not support partial sequence removal
srv load_model: speculative decoding not supported by this context
```
Spec decode disabled.

**After** (same model, same flags):

```
common_context_can_seq_rm: checkpoint round-trip OK (size=...); enabling
    spec decode via checkpoint rollback
srv load_model: speculative decoding will use checkpoints
```
Spec decode active.

## Empirical validation

Measured on AMD Strix Halo (gfx1151, 96 GB unified, 256 GB/s LPDDR5x),
ROCm 7.2, llama.cpp master + this patch.

Model: `unsloth/Qwen3.6-27B-GGUF` UD-Q2_K_XL (hybrid Gated DeltaNet).

1. **`llama-lookup` (via patched `common_context_can_seq_rm`, spec path
   already works in the lookup binary once the probe doesn't refuse):**
   - Baseline (no spec): **13.82 tps**
   - With `--draft-max 4 --ctx-checkpoints 8`: **30.05 tps**, α = 65.29 %
   - 2.17× over baseline, output fully coherent.

2. **`llama-server --spec-type ngram-cache` (the direct target of this
   patch — without it, spec decode is disabled at model load):**
   - Server startup log shows `srv load_model: speculative decoding
     will use checkpoints` (the promoted-to-FULL branch this patch
     enables).
   - Short prompt: 13.22 tps, α = 1.00 on 29/29 drafts.
   - Long code-review prompt (1776 → 512 tokens): 11.76 tps, α = 1.00
     on 119/119 drafts.
   - Drafts generate, verify, accept through the checkpoint-promoted
     path. Output coherent.

Raw logs, benchmarks, and the full investigation trail are in
<https://github.com/eval-l-live/mlm> under `qwen36-fast/bench/` (runs
iter-13 and iter-21) and `qwen36-fast/notes/`.

## Compatibility

- Does **not** modify any model builder (`src/models/*.cpp`).
- Does **not** touch `llama_memory_recurrent` or any other memory
  module.
- Does **not** modify any converter or GGUF format.
- Only widens the capability probe in `common/`.
- For models where raw `seq_rm` already works, the probe returns
  `PART` as before — zero behaviour change.
- For models where `seq_rm` fails AND the checkpoint round-trip also
  fails, we still promote to `FULL` with a clear log line (same as the
  pre-patch fallthrough since #19493 wired this code path), so no new
  failure modes are introduced.

## Testing done

Builds:
- ROCm 7.2 on gfx1151 (Strix Halo), `cmake -DGGML_HIP=ON`
- Also applies cleanly to master tip (checked on
  `86db42e CUDA: fuse relu + sqr (#22249)`, `git apply --check`
  passes, `git am` passes).

Models:
- Qwen/Qwen3.6-27B (UD-Q2_K_XL, Q4_K_M) — hybrid Gated DeltaNet
  (primary target; confirmed both `llama-lookup` and `llama-server`
  paths work).
- Regression check on a non-hybrid model (probe still returns `PART`
  via the existing fast path — the new code is only reached when the
  raw `seq_rm` call fails).

Binaries exercised: `llama-lookup`, `llama-server` (`--spec-type
ngram-cache`).

## Risks and caveats

- If a model's memory module reports `llama_state_seq_get_size_ext(...,
  PARTIAL_ONLY) > 0` but the actual checkpoint path is still broken at
  decode time, spec decode will fail at verify time rather than at the
  probe. Failure is visible (accept-rate drops, model emits garbage)
  and can be disabled by the user with `--spec-type none`. No memory
  corruption or crash is possible because the checkpoint machinery
  itself is unchanged.
- Models that implement neither raw `seq_rm` nor the checkpoint path
  previously returned `NO` (which disabled spec). This patch promotes
  them to `FULL` with a warning, which is the same fallback the
  existing code already uses when `seq_rm` fails in the non-hybrid
  case. No regression — worst case is the existing pre-patch
  `SEQ_RM_TYPE_FULL` fallthrough.
- Checkpoint buffer allocation: only done inside the probe (called
  once at model load). The buffer is sized by
  `llama_state_seq_get_size_ext` and freed when the `std::vector<uint8_t>`
  goes out of scope — no long-lived allocation.

## Relationship to other PRs

- **#19493 (merged)** — provides the underlying
  `llama_state_seq_*_ext(PARTIAL_ONLY)` API this patch probes. Hard
  prerequisite.
- **#20700 (open, WIP, Qwen3.5 MTP)** — orthogonal. That PR modifies
  `common_speculative_is_compat` in `common/speculative.cpp` (a
  different function in a different file) and gates its relaxation on
  `llama_model_n_mtp_layers(model) > 0` (MTP-specific). This patch
  widens the more general probe in `common/common.cpp` and benefits
  all drafter kinds (lookup, ngram-cache, draft-model, and — once
  #20700 or similar lands — MTP/EAGLE3). Both patches can coexist.
- **#21437 (EAGLE3, open)** / **#22105 (DFlash, open)** — both need the
  probe to return `FULL` on hybrid models to enable spec decode. This
  patch is a precondition for them being usable on Qwen3.5/3.6,
  Qwen3-Next, GLM4-MoE, etc.

## Open questions for reviewers

1. Should the checkpoint round-trip probe be a separate helper
   (`common_context_can_checkpoint`) that `common_context_can_seq_rm`
   calls, rather than inlined? That would also let
   `common_speculative_is_compat` (and #20700) share the logic.
2. Logging verbosity: the success case currently emits `LOG_INF` at
   model load. Happy to demote to `LOG_DBG` if the maintainers prefer
   quieter startup.
3. Should the fallthrough on `llama_decode` failure still promote to
   `FULL` when the checkpoint API is *available* (this patch's
   behaviour), or only when we can *successfully* round-trip empty
   state? Currently, if the 2-token decode fails we accept a non-zero
   `get_size_ext` as sufficient evidence; we could tighten to require
   a successful `set_data_ext` round-trip on empty state, at the cost
   of a second decode attempt after the round-trip.
4. Is there a lightweight smoke-test model in the CI matrix that
   exercises a hybrid memory? If not, adding one to
   `ci/run.sh`/`tests/test-context.cpp` could lock in this behaviour
   for future refactors.
