# 05 — M-RoPE spec-decode bug: diagnosis + fix deferred

Iteration 7 of the qwen36-fast timeboxed run. Date: 2026-04-23.

Follow-up to iteration 6 (bench/02-lookup-spec.md), which identified that
llama-lookup on Qwen3.6-27B Q4_K_M produces gibberish + `decode: failed to
decode, ret = -1` on every verify batch, and blamed "M-RoPE X < Y check".

This iteration's goal: precisely localize the check, attempt a minimal fix,
and re-run lookup to verify.

## TL;DR

- **The M-RoPE message is a red herring.** The check that fires is real and
  at a real line, but the root cause is NOT M-RoPE positional encoding.
- **Root cause**: Qwen3.6 is a *hybrid-recurrent* model (Gated DeltaNet 3:1
  Gated Attention). The recurrent (SSM-like) state refuses partial rollback
  by design. When lookup submits a k-token verify batch and the target
  model rejects any drafts, lookup tries to truncate the memory with
  `seq_rm(seq=0, p0=n_past, p1=-1)`. The recurrent cache refuses (partial
  intersection invalid), the hybrid seq_rm returns false without touching
  the attn cache, the next batch's min position is ≤ memory's stored max,
  and the M-RoPE check in `llama_batch_allocr::init` fires as the visible
  error.
- **Scope**: this blocks llama-lookup, llama-cli `--draft-*`, llama-server
  `-md`, and any future MTP/EAGLE path on Qwen3.6, Qwen3-Next, GLM4-MoE
  (with mrope), and any other hybrid-recurrent arch in llama.cpp
  (lfm2, mamba-base, plamo2, kimi-linear, qwen35moe).
- **Fix**: requires implementing SSM/GDN state checkpoint+restore in
  `llama_memory_recurrent` (~300-500 LoC, multi-day). Out of scope for
  one timebox iteration.
- **A small diagnostic patch** was written (see patches/) that adds a
  dedicated error message for this case so the next person hits the real
  cause immediately instead of chasing M-RoPE. Not built/benchmarked — the
  patch is diagnostic only.

## Exact file + line of the failing check

**`deps/llama.cpp/src/llama-batch.cpp:264-274`** (in the M-RoPE branch of
`llama_batch_allocr::init`).

```cpp
if (n_pos_per_embd > 1) {
    // M-RoPE case ...
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        ...
        const llama_pos p0 = memory ? memory->seq_pos_max(s) : -1;
        if (batch.token) {
            if (p0 >= 0 && p0 >= seq_pos_min(s)) {   // <-- THIS
                LLAMA_LOG_ERROR(
                    "%s: the tokens of sequence %d ... "
                    " for M-RoPE, it is required that the position satisfies: X < Y\n",
                    __func__, s, s, p0, s, seq_pos_min(s));
                return false;
            }
        }
        ...
    }
}
```

`n_pos_per_embd() == 4` for rope_type `LLAMA_ROPE_TYPE_MROPE` or
`LLAMA_ROPE_TYPE_IMROPE` (`src/llama-hparams.cpp:205`). Qwen35/Qwen35MoE
use `LLAMA_ROPE_TYPE_IMROPE` (`src/llama-model.cpp:9329-9331`), so Qwen3.6
takes this branch.

**Symptom in log** (from iteration-6's `/tmp/qwen36-lookup-bench/10-lookup-dyn-rep1.log`):

```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 1779
 - the tokens for sequence 0 in the input batch have a starting position of Y = 1777
 for M-RoPE, it is required that the position satisfies: X < Y
decode: failed to initialize batch
llama_decode: failed to decode, ret = -1
```

Y (=1777) is exactly 2-3 below X (=1779). That's the arithmetic signature
of "last verify batch had 4 drafts (advanced state to n_past+3), only 0-2
accepted, lookup now re-submitting starting at n_past+k for small k".

## Why spec-decode violates the check

### The real root cause (not M-RoPE)

Qwen3.6 is hybrid: 64 layers, 3 Gated DeltaNet (recurrent) per 1 Gated
Attention (full KV). See `src/models/qwen35.cpp:35-41` — `is_recurrent(il)`
dispatch. The recurrent layers use a Mamba-like SSM state stored in
`llama_memory_recurrent` (a separate memory module). The hybrid is
`llama_memory_hybrid` which wraps both.

Sequence of events for lookup dm=4 on Qwen3.6:

1. After prompt prefill, `n_past = 1778` (say). Memory state: attn cache
   has entries 0..1778; recurrent cache has been evolved through 0..1778.
2. Lookup drafts 4 tokens, calls `common_batch_add` 4 times with positions
   `(n_past, n_past+1, n_past+2, n_past+3)` = 1778..1781, all seq=0.
3. `llama_decode` runs this as a single ubatch. Because the batch has
   1 unique sequence (seq=0) with consecutive positions, the hybrid
   memory calls `balloc.split_equal(n_ubatch, sequential=true)`
   (`src/llama-memory-hybrid.cpp:78`), which returns a ubatch with
   `n_seqs=1, n_seq_tokens=4, equal_seqs=true`. The GDN state is then
   evolved through all 4 positions during forward pass — this *does not
   produce valid probabilities* for the target-verify step because the
   state advancement commits the rejected-draft input to the SSM state.
4. Target samples. Say it accepts draft[0] (=1778) but rejects
   draft[1] at 1779 (because the target's distribution at 1779 disagrees).
5. Lookup sets `n_past = 1779` and calls
   `llama_memory_seq_rm(mem, 0, 1779, -1)`.
6. **Here's the failure**:
   * `llama_memory_hybrid::seq_rm` first asks recurrent:
     `mem_recr->seq_rm(0, 1779, -1)` (`src/llama-memory-hybrid.cpp:135`).
   * Inside `llama_memory_recurrent::seq_rm`, at
     `src/llama-memory-recurrent.cpp:166`:
     ```cpp
     if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
         // partial intersection is invalid if it includes the final pos
         return false;
     }
     ```
     `p0=1779`, `cell.pos=1781` (the tail cell, last batch's last pos),
     `p1=INT_MAX`. Satisfies all three → returns false.
   * Hybrid `seq_rm` early-returns false (never touches attn cache).
   * **Both caches still have seq_pos_max=1781, but user thinks n_past=1779.**
7. Next verify batch: lookup adds new drafts starting at `n_past=1779`.
   `llama_batch_allocr::init` sees memory->seq_pos_max(0)=1781,
   batch-min-pos=1779, applies M-RoPE branch's `p0 >= seq_pos_min(s)`
   check → TRUE → returns false. "failed to initialize batch" → -1.
8. Lookup continues looping: draft rejected silently, sampler re-samples
   the SAME token (because target context unchanged), infinite-loop
   gibberish output until n_predict hits.

### Why the even simpler case (full reject) ALSO fails

If the target accepts zero drafts: `n_past` stays where it was before
drafting, but recurrent state advanced anyway. `seq_rm(0, n_past, -1)` has
the same `p0 < cell.pos` signature → refused. Same failure.

### Why non-Qwen3.6 non-hybrid models work

A plain Qwen3 or LLaMA decode with spec works because:
- attn KV cache supports partial truncation cleanly
- no recurrent component
- `seq_rm` after partial-accept succeeds, memory rolls back, next batch
  starts at correct position

## Why the proposed "M-RoPE section fix" from the orchestrator brief was wrong

The iteration-6 brief suggested the bug is M-RoPE-specific 3D position
computation. It isn't. Proof:

1. The Qwen3.6 tokens being decoded are pure text, not image+text, so
   M-RoPE's 4-axis [t,h,w,time] reduces to just the temporal axis — equivalent
   to scalar position.
2. The non-M-RoPE branch of the same `init` code (lines 289-321) applies
   a STRICTER check: `seq_pos_min(s) != p0 + 1` — which would ALSO fire
   for any hybrid-recurrent spec-decode attempt.
3. Switching rope_type away from IMROPE would not fix the underlying
   partial-rollback refusal of `llama_memory_recurrent`.
4. The orchestrator's suggested fix candidates (relax the check, compute
   3D M-RoPE position per token) do not address the recurrent state
   corruption — they would let the decode proceed with a poisoned GDN
   state, producing garbage output faster but still garbage.

## The fix — what's actually needed

To enable speculative decoding on hybrid-recurrent models, add state
checkpoint/restore to `llama_memory_recurrent`:

1. In `llama_memory_recurrent::init_batch`, before `find_slot` /
   `prepare`, snapshot the current SSM state tensors (`ssm_states_all`)
   and conv state tensors (`conv_states_all`) for each affected sequence.
   Store in a per-context checkpoint buffer.
2. Add a new API `llama_memory_recurrent::seq_rollback(seq_id, to_pos)`
   that restores the checkpoint if `to_pos` is within the last batch's
   position range, and decrements the tail cell's pos.
3. Call `seq_rollback` from `llama_memory_hybrid::seq_rm` when partial
   intersection is detected, before falling through to the current
   return-false logic.
4. For spec-decode consumers (lookup/speculative/server), after partial
   accept: call `seq_rm` as today; the hybrid now succeeds; next decode
   works.

Cost estimate: ~300-500 LoC. Core work is the tensor checkpoint
(allocation, backend-aware copy via `ggml_backend_tensor_copy_async`,
teardown on success). Multi-day to land, test across backends (ROCm,
CUDA, Metal, Vulkan, CPU).

**An alternative "cheap" fix** (not implemented here): constrain lookup
and common/speculative.cpp to submit 1-token verify batches when
`llama_model_has_recurrent(model) && cparams.kv_unified`. This removes
the benefit of spec decode on these models (you can still draft, but you
verify one token at a time = 0 speedup). Not worth the wire.

## The patch

`patches/llamacpp-mrope-spec-decode-fix.patch` — 21-line diagnostic-only
patch against `src/llama-batch.cpp`. Adds a dedicated error message so
the next person chasing this bug sees the root cause (recurrent-state
rollback refusal) rather than a misleading M-RoPE hint. Does NOT change
behavior — spec decode still fails, just with a clearer message.

## Build outcome

Not attempted. The patch is diagnostic-only; rebuilding llama.cpp inside
ROCm 7.2 docker for ~15 min to get a better error message is not worth
the compute. The iteration-6 numbers already told us spec-decode α=65%
is unreachable with the current stack.

## Benchmark outcome

Not applicable — no behavioral change patch was built.

## Decision for Phase 1

**Spec-decode via llama.cpp is dead for Qwen3.6** until one of:
1. Someone implements recurrent-state checkpointing in llama.cpp
   (upstream PR, multi-day).
2. We port Qwen3.6 to a framework that already handles this
   (vLLM/SGLang do it for Mamba/Mamba2; MLX likewise). Note: none have
   ROCm/gfx1151 kernels for the GDN layer today — that's Phase 2's work.
3. We switch to a non-hybrid Qwen variant (there is no 3.6 dense
   non-hybrid at usable size; the 27B IS the target).

Realistic path to the 40 tps target on Strix Halo:
- (A) Phase 2 Vulkan GDN kernel + run on llama.cpp backend that avoids
  the recurrent-cache entirely (not possible — the memory module IS the
  arch).
- (B) Implement the recurrent-state checkpoint in llama.cpp, then wire
  MTP head as drafter. This is the original Phase 3 plan but now
  bottlenecked on (B) first.
- (C) Abandon llama.cpp, port to vLLM-ROCm or SGLang-ROCm. Heavy but
  unblocks Phase 2+3 in one move.

## Next iteration recommendation

Ranked:
1. **Scope the recurrent-state checkpoint patch for llama.cpp**. Enumerate
   exactly the tensors + backend APIs needed. This is the single biggest
   unblocker and may be ~200-300 LoC if done right.
2. **Try draft-model spec with kv_unified=false + parallel=1 on non-recurrent
   layers only**. Extremely unlikely to work (the hybrid memory splits
   affect both), but takes 10 minutes to disprove.
3. **Fallback bench: Q2_K 27B on ROCm** for raw bandwidth headroom
   (~36 tps ceiling → realistic ~25 tps with overhead). No spec needed.
   Quality loss unclear; worth measuring.
4. **Investigate vLLM-ROCm support for Qwen3.6.** If it exists, we skip
   the llama.cpp C++ swamp entirely.

## Files touched

- `patches/llamacpp-mrope-spec-decode-fix.patch` (new, diagnostic-only)
- `notes/05-mrope-spec-fix.md` (this file)
