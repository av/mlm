# Iter-24: Server-side ngram-cache tuning sweep

Date: 2026-04-23 CEST
Binary: `deps/llama.cpp/build-rocm/bin/llama-server` (build b8892-0d0764dfd, iter-11 patched libllama-common)
Model: `Qwen3.6-27B-UD-Q2_K_XL.gguf` (Unsloth)
Image: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`
Prompt: `prompts/prompt_code.txt` (1776 tokens, self-referential code review)
Request: POST /v1/chat/completions, `max_tokens=512`, `temperature=0`

## Motivation

Iter-21 confirmed the iter-11 patch fires on llama-server path (`speculative
decoding will use checkpoints`) but the end-to-end throughput was only 11.76
tps — far below the 30.05 tps that iter-13 reached via llama-lookup on the
same model + same prompt + same `--draft-max 4`. This iteration sweeps every
relevant server-side spec-decode knob to either (a) close the gap or (b)
document the structural reason it cannot close on the server path.

## Configs swept

Seven configs total. A–D are the requested baseline; E–G are additional
finer-grain dm-sweep around A to find the sweet spot.

| Tag | Flags |
|---|---|
| A | `-c 4096 -np 1 --spec-type ngram-cache --draft-max 4  --ctx-checkpoints 8` |
| B | `-c 8192 -np 1 --spec-type ngram-cache --draft-max 16 --draft-min 2 --ctx-checkpoints 8` |
| C | `-c 8192 -np 1 --spec-type ngram-simple --draft-max 16 --draft-min 2 --spec-ngram-size-n 2 --spec-ngram-size-m 16 --ctx-checkpoints 8` |
| D | `-c 8192 -np 1 --spec-type ngram-map-k --draft-max 16 --draft-min 2 --spec-ngram-size-n 2 --spec-ngram-size-m 16 --spec-ngram-min-hits 1 --ctx-checkpoints 8` |
| E | `-c 4096 -np 1 --spec-type ngram-cache --draft-max 8  --ctx-checkpoints 8` |
| F | `-c 4096 -np 1 --spec-type ngram-cache --draft-max 4  --ctx-checkpoints 16 --checkpoint-every-n-tokens 32` |
| G | `-c 4096 -np 1 --spec-type ngram-cache --draft-max 2  --ctx-checkpoints 8` |

Default `--spec-type ngram-cache` is with `--draft-max 16`, `--draft-min 0`.
Default `--spec-ngram-size-n 12`, `--spec-ngram-size-m 48`, `--spec-ngram-min-hits 1`
(source: `common/common.h:306-315`).

## Sweep results

All 7 configs completed. Prompt token count 1776, completion exactly 512 (all
hit `finish_reason=length`), temp=0. All outputs are coherent `<think>`-based
reasoning — no word salad.

Raw data: `14-server-tuning-{A..G}.{log,json,server.log}`.

| Cfg | spec-type | draft-max | decode tps | prompt tps | draft_n | accepted | α (fired) | draft-coverage (draft_n / 512) | Output sanity |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | ngram-cache  | 4  | **12.11** | 312 | 119 | 119 | 1.00 | 23.2% | coherent |
| B | ngram-cache  | 16 | 10.96 | 307 | 131 | 131 | 1.00 | 25.6% | coherent |
| C | ngram-simple | 16 |  8.67 | 306 | 213 | 213 | 1.00 | 41.6% | coherent |
| D | ngram-map-k  | 16 |  9.34 | 306 | 185 | 185 | 1.00 | 36.1% | coherent |
| E | ngram-cache  | 8  | 10.92 | 310 | 137 | 137 | 1.00 | 26.8% | coherent |
| F | ngram-cache  | 4 + `-cpent 32` | 11.96 | 308 | 119 | 119 | 1.00 | 23.2% | coherent |
| **G** | **ngram-cache**  | **2**  | **12.28** | 306 | 103 | 103 | 1.00 | 20.1% | **coherent (best)** |

Reference: **iter-13 llama-lookup at `--draft-max 4` on this exact prompt: 30.05 tps, α=65.29%** (bench/06-patched-lookup.md).

## Observations

1. **α=1.00 is misleading on this prompt.** The draft counter `draft_n` on
   server only counts drafts that **fired and were fully accepted** (since
   temp=0 + ngram verbatim match = deterministic). When no ngram hit exists,
   no draft is produced and no counter increments. Effective draft coverage
   is 20–42% of output tokens — the other 58–80% decode at plain target-model
   speed (~13.8 tps).

2. **Monotonic: smaller `--draft-max` = higher tps on this hardware.** G
   (dm=2) > A (dm=4) > E (dm=8) ≈ B (dm=16). Each doubling of draft-max
   costs ~10% tps even though α stays at 1.00. Larger verify batches take
   longer per decode step than the draft savings recoup.

3. **`ngram-simple` and `ngram-map-k` fire more drafts but lose overall.**
   C and D have 1.7–1.8× higher draft coverage than A (213, 185 vs 119) but
   run 20–28% slower because each proposed draft is part of a bigger
   verify batch, and the `ngram-size-n=2, ngram-size-m=16` M-gram expansion
   produces longer chains (on the repetitive prompt) that take more
   per-batch time.

4. **`--ctx-checkpoints` and `--checkpoint-every-n-tokens` are prefill-side
   knobs.** Config F (ctx-ckpts=16, cpent=32) is indistinguishable from A
   (12.11 vs 11.96 tps, within noise). On a 512-token decode that hits only
   2 checkpoints during prefill, raising ctx-ckpts doesn't help. These are
   prompt-cache eviction policy knobs, not decode-loop knobs.

5. **`-np 1` vs `-np 4` doesn't matter for single-request throughput.** All
   our configs use `-np 1`; iter-21 used implicit `-np 1` also (verified in
   startup log — 4 slots initialized are spec-decode contexts, NOT task
   slots).

## Best server config found

**Config G: `--spec-type ngram-cache --draft-max 2 --ctx-checkpoints 8 -c 4096 -np 1`**

- **12.28 tps decode** (+1.4% over iter-21's 11.76, +4.4% over same-param
  config A's 12.11 tps — but within run-to-run noise).
- α = 1.00 on drafts that fired; draft coverage 20% of output.
- Output fully coherent (1828 chars `<think>` reasoning, same shape as iter-21).

## Gap to iter-13 llama-lookup 30.05 tps: NOT closed

**Best server-side tps 12.28 is 41% of llama-lookup's 30.05.** The ~18 tps
gap does not close on any of the 7 configs tried.

### Ruled out: workload mismatch

First hypothesis was that iter-13's `n_predict=256` measurement avoided the
long `<think>` tail and caught more of the verbatim-repeat phase. To test,
re-ran Config G (the new best) with `max_tokens=256` via the same chat
endpoint (data: `14-server-tuning-G-n256.json`):

| Config G n_predict | decode tps | draft_n | draft coverage |
|---|---:|---:|---:|
| 256 | 12.59 | 61 | 23.8% |
| 512 | 12.28 | 103 | 20.1% |

Essentially identical. **Workload composition is NOT the dominant gap
source.** The server's ngram-draft mechanism simply fires ~3× less often
than llama-lookup's.

### Remaining reasons (in likely decreasing order)

### Reason #1 — ngram fire rate: server ≈ 22%, lookup ≈ 94%

iter-13 llama-lookup on the same prompt: `n_drafted=242` on 256 decoded =
**94% draft coverage**.
iter-24 server Config G at n=256: `draft_n=61` on 256 decoded = **24% draft
coverage**.

Same `common_ngram_cache_draft()` function (`common/ngram-cache.cpp:146`) —
so the ngram-cache logic is identical. The difference is how the caller
prepares the cache and invokes draft. llama-lookup builds a context cache
from the full input in a single pass before decode starts and keeps it
populated across all steps. Server rebuilds / updates the ngram context
cache per-decode-step and appears to skip draft emission when it can't
satisfy strict-mode thresholds.

### Reason #2 — checkpoint save overhead per spec step (server-specific)

In `tools/server/server-context.cpp:369`, when
`ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL` (which is exactly the
branch our iter-11 patch enables on Qwen3.6), the server calls
`server_get_checkpoint(ctx, this->id, n_tokens)` **on every spec-decode step**.
The server log shows each checkpoint is **149.626 MiB** (full GDN recurrent
state). At ROCm D→D of ~50 GB/s that's ~3 ms/step. Over ~103 spec steps
(Config G) that's ~0.3 s overhead on a 42 s decode — ~0.7% of wall clock.
Measurable but not dominant.

llama-lookup does NOT take checkpoints — it calls `llama_memory_seq_rm`
directly and relies on the decoder's internal rollback. That's why
llama-lookup sees M-RoPE `decode: failed` log spam but the output is still
coherent (PR #19493 machinery handles it). Server pre-empts that by
saving the checkpoint up-front, paying the save cost always rather than
the rollback cost sometimes.

### Reason #2 — checkpoint save overhead per spec step (server-specific)

In `tools/server/server-context.cpp:369`, when
`ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL` (which is exactly the
branch our iter-11 patch enables on Qwen3.6), the server calls
`server_get_checkpoint(ctx, this->id, n_tokens)` **on every spec-decode step**.
The server log shows each checkpoint is **149.626 MiB** (full GDN recurrent
state). At ROCm D→D of ~50 GB/s that's ~3 ms/step. Over ~200 spec steps
that's ~0.6 s overhead on a 42 s decode. Not dominant but measurable.

llama-lookup does NOT take checkpoints — it calls `llama_memory_seq_rm`
directly and relies on the decoder's internal rollback. That's why
llama-lookup sees M-RoPE `decode: failed` log spam but the output is still
coherent (PR #19493 machinery handles it). Server pre-empts that by
saving the checkpoint up-front, paying the save cost always rather than
the rollback cost sometimes.

### Reason #3 — chat-template + reasoning-mode overhead

Small (<1%): the `/v1/chat/completions` endpoint wraps the user message in
Qwen3.6's chat template + forces `<think>` start, doing JSON parse + sampler
init per request. llama-lookup's `-f prompt.txt` is raw-prompt mode. Not a
factor at this magnitude.

## Verdict

- **Sweep outcome:** Best server config for Qwen3.6-27B Q2_K_XL is
  **`--spec-type ngram-cache --draft-max 2 --ctx-checkpoints 8 -c 4096
  -np 1`** at **12.28 tps** decode on this prompt. No single-flag change
  gets within 2× of llama-lookup's 30.05 tps on the same model.

- **Biggest single-flag impact:** `--draft-max` (monotonic, smaller
  is better on this hardware — 2 > 4 > 8 > 16). Second biggest:
  `--spec-type` (ngram-cache beats ngram-simple/ngram-map-k on dm=16).
  `--ctx-checkpoints` and `--checkpoint-every-n-tokens` are noise for
  short decodes. `--spec-ngram-size-{n,m}` + `--spec-ngram-min-hits`
  only affect `ngram-simple` / `ngram-map-*` which already lose.

- **Gap closed?** NO. Best server 12.28 tps vs llama-lookup 30.05 tps is a
  2.45× ratio. The iter-13 n_predict=256 workload is NOT the cause (verified:
  Config G at n_predict=256 on server gives 12.59 tps, not 25-ish).
  The dominant cause is **ngram draft fire rate**: llama-lookup proposes
  drafts on 94% of decode steps, server on only 20-24%. Same
  `common_ngram_cache_draft` function; different caller-side cache management.
  Checkpoint-save-per-step is a secondary cost (<5% of budget).

- **Recommendation for README:** server is viable for the persistent-
  serving + streaming + multi-user case at ~12 tps on Qwen3.6 with
  `--draft-max 2`. For raw single-request throughput, llama-lookup at 30
  tps remains the winner. Do not change the "Reproduce" default.

## What remains

- Dig into `tools/server/server-context.cpp` draft-generation path to find
  why ngram-cache is invoked only on ~22% of decode steps vs 94% in
  llama-lookup. Candidates: early-out when `spec_draft` is non-empty (reusing
  partial drafts), or strict-mode thresholds kicking in. A 100-LoC change
  there would likely close most of the remaining gap.
- Make `llama_state_seq_get_data_ext PARTIAL_ONLY` lazy (only save when
  draft is non-trivial) — ~50 LoC patch to server-context.cpp. Secondary
  effect.
- Server-side prompt-cache warmup (via `/slots/save`) to amortize the
  ngram-cache ramp across requests — may help the "cold-cache" tps on
  subsequent requests.
