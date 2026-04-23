# Re-verify upstream #19493 checkpoint-based spec decode on Qwen3.6-27B

Run date: 2026-04-23 CEST (iter-10)

## TL;DR — **upstream PR #19493 does NOT fix Qwen3.6 spec decode** (this iter)

Against iter-9's optimistic read of the upstream survey, **the merged
checkpoint infrastructure is not sufficient to enable speculative
decoding on Qwen3.6**. The `llama-server` process refuses to even
initialise the speculative path on a Qwen3.6-27B Q2_K_XL target:

```
common_speculative_is_compat: the target context does not support partial sequence removal
srv    load_model: speculative decoding not supported by this context
```

This happens *before* any draft attempt; no n-gram map / draft / accept
events are emitted. Effective decode tps equals the no-spec baseline
(**13.84 tps** measured with `--spec-type ngram-cache` vs **13.82 tps**
bench baseline — within noise).

### Why #19493 doesn't help Qwen3.6

The upstream `common_context_can_seq_rm` probe decodes 2 dummy tokens,
then tries `llama_memory_seq_rm(mem, 0, 1, -1)`. The result maps to:

- `PART` — full rollback works ⇒ vanilla spec decode
- `FULL` — rollback of the whole sequence works but partial does not ⇒
  checkpoint-based spec decode (the #19493 code path)
- `NO` — even a full rollback fails after 2 decode steps ⇒ spec decode
  refused entirely

Qwen3.6's hybrid Gated-DeltaNet 3:1 Gated-Attention (visible in server
logs as `sched_reserve: fused Gated Delta Net (autoregressive/chunked)
enabled` + `llama_memory_recurrent: ... 598.50 MiB RS buffer`) hits
the **`NO`** branch — the recurrent state layers refuse the seq_rm
probe even after 2 tokens. #19493 only wires the `FULL` path; there is
no fallback into the `NO` regime.

### Important flag-name correction (vs iter-9 notes)

The `--spec-ckpt-num-tries` flag **does not exist** in the current
upstream. The real knobs are:

- `--ctx-checkpoints N` (alias `-ctxcp`, `--swa-checkpoints`) —
  controls the server's context checkpointing (saves partial sequence
  state so a cache hit can short-circuit prompt reprocessing). **Not
  spec-decode specific.** Default 32.
- `--spec-type [none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]` —
  picks the speculator. Only honoured if `common_speculative_is_compat`
  returned `PART` or `FULL`.
- `--draft-max N` — max draft lookahead (default 16).
- `--draft-min N`, `--spec-ngram-size-n`, `--spec-ngram-size-m`,
  `--spec-ngram-min-hits` — tuning knobs.

Iter-9's `06-upstream-survey.md` cited `--spec-ckpt-num-tries` based
on the PR description text; that flag was **not** merged — either it
was a working-title in the PR description or renamed before merge.

## Binary / build

- Image: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2` (`llama-server`
  timestamp 2026-04-09, 12.13 MiB)
- Reported: `version: 8738 (d6f303004)`, `build_info: b8738-d6f303004`
- Our local `deps/llama.cpp` HEAD: same `0d0764d` commit (build 8738)
- Checkpoint infrastructure present in both: `common/speculative.cpp`
  defines `common_speculative_checkpoint`, `draft_create_checkpoint`,
  `draft_restore_checkpoint`. `tools/server/server-context.cpp` defines
  `server_prompt_checkpoint`, `server_get_checkpoint`, wiring at line
  369-371 (context checkpoint) and line 879 ("speculative decoding
  will use checkpoints" — this message did not fire in our runs).

## Measurement

Command (run twice — once with `ngram-map-k`, once with `ngram-cache`;
identical result):

```
docker run -d --rm --name qwen36-spec-server \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 18080:8080 \
  kyuz0/amd-strix-halo-toolboxes:rocm-7.2 \
  llama-server -m .../Qwen3.6-27B-UD-Q2_K_XL.gguf \
    --host 0.0.0.0 --port 8080 -fa 1 -c 4096 -ngl 99 \
    --spec-type ngram-cache --draft-max 4 --ctx-checkpoints 8 \
    --no-warmup
```

Request: greedy chat completion, `temperature=0.0 top_k=1 n_predict=256`,
prompt = iter-6 `/tmp/qwen36-lookup-bench/prompt_code.txt`
(1778 tokens after chat-template wrapping).

### Comparison table

| run                                             | decode tps | accept α | clean? | comment                                           |
|-------------------------------------------------|-----------:|:---------|:------:|---------------------------------------------------|
| iter-5 `llama-bench` baseline (d=2048)          |      10.87 | n/a      |    Y   | Q4_K_M                                            |
| iter-6 `llama-lookup` dm=0 (Q4_K_M)             |      10.83 | n/a      |    Y   | identical to baseline                             |
| iter-6 `llama-lookup` dm=4 (Q4_K_M)             |     ~26.10 | 65.1%    |  **N** | M-RoPE error, gibberish, 64 decode failures       |
| iter-8 `llama-bench` baseline Q2_K_XL (d=2048)  |      13.82 | n/a      |    Y   | best no-spec                                      |
| **iter-10** `llama-server` spec ngram-map-k     |      13.78 | **n/a**  |    Y   | `speculative decoding not supported by this context` → spec path off |
| **iter-10** `llama-server` spec ngram-cache     |      13.84 | **n/a**  |    Y   | same; within noise of baseline                    |

Context checkpoints (the orthogonal feature) DID fire — two per task at
~150 MiB each — that is the seq-cache prompt-reuse system, not the
spec-decode one. Easy to confuse because both features share the
`--ctx-checkpoints` flag wiring and both log "created context
checkpoint N of 8". The log line *not* seen — `"speculative decoding
will use checkpoints"` — is the discriminator.

### Speedup vs baseline 13.82 tps

**×0.998 (negligible)**. Spec path does not engage.

### Acceptance rate α

Unavailable — no draft tokens were ever proposed. Iter-6's 65.1 % α on
Q4_K_M + lookup at dm=4 still stands as the best estimate of what we
*would* see if the rollback path worked end-to-end.

## VERDICT

**Upstream PR #19493 is a *necessary but not sufficient* fix for Qwen3.6
speculative decoding.** It solves the `seq_rm(…partial…)` refusal path
by saving/restoring recurrent state around a partial draft accept, but
only when the backend supports the `FULL` rollback tier. Qwen3.6's
Gated DeltaNet layers are currently stricter (the stock HIP/ROCm
`llama_memory_recurrent` implementation rejects even the `seq_rm(0,1,-1)`
probe after a 2-token decode) so the compat probe returns `NO` and the
whole spec path is disabled defensively.

### What remains

The high-impact unit of work, in priority order:

1. **Relax the `seq_rm NO → speculative off` refusal** (or teach the
   compat probe to accept checkpoint-restorable backends). About 30-80
   LoC in `common/common.cpp:common_context_can_seq_rm` +
   `tools/server/server-context.cpp:874`. Probe `seq_rm` after a
   state-checkpoint save, and if restore round-trips successfully,
   return `FULL`. This is the minimal viable change to turn on
   ngram-map / ngram-cache spec decode on Qwen3.6 with today's
   checkpoint code path. **Do first** — it unblocks everything.

2. **Port PR #20700 (Qwen3.5 MTP + FastMTP) to Qwen3.6.** The two
   models share the `qwen35` arch in llama.cpp and the `mtp.*` tensor
   layout in HF safetensors (confirmed in iter-3). Re-uses:
   - our iter-3 observation that Qwen3.6 MTP already sits in HF
     safetensors (15 tensors, 4.29 GiB, `mtp.layers.0.*` full attention
     block, not GDN — easier port),
   - upstream's MTP attention graph work,
   - `--two-phase-decode` option for safer hybrid rollback,
   - FastMTP vocab trimming (248K → 32K) for ~3.7× drafter throughput.
   Author reports 82 % α on 9B Q4_K_M.

3. **Alternative: train an EAGLE3 head** (PR #21437 already does it for
   Qwen3.5 4B/9B/35B-A3B; extend to Qwen3.6-27B). Heavier lift, but
   independent of the hybrid rollback problem because EAGLE3 runs
   drafter separately.

4. **DFlash port** (PR #22105, 8× speedup claimed) stays out of reach
   while hybrid rollback is broken — DFlash writes recurrent state
   ahead of accept and needs the same restore.

## Raw log

See `05-lookup-with-checkpoint.log` (full server log + JSON response
body).

## Cross-reference

- iter-6 lookup gibberish: `bench/02-lookup-spec.md` — cause was the
  same `seq_rm NO` trap, just surfacing as M-RoPE position errors
  because `llama-lookup` (the old binary) didn't check compat and
  proceeded to decode anyway.
- iter-7 rollback diagnosis: `notes/05-mrope-spec-fix.md` — correct
  root-cause analysis, suboptimal estimate of effort needed
  (we now think it's 30-80 LoC not 300-500 thanks to #19493 laying
  the state-dump plumbing).
- iter-9 PR survey: `notes/06-upstream-survey.md` — still correct on
  what PRs exist; wrong about the fix being out-of-the-box ready for
  Qwen3.6, and wrong about the `--spec-ckpt-num-tries` flag name.
