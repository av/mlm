# Cold vs warm dynamic-cache characterization

Iter-29 (final). 2026-04-23 ~05:40 CEST. **Answers a user-facing question**: will
Ivan see different tps on the first message vs follow-up messages in a real workflow?

**Short answer**: **Yes, but the gap is the OPPOSITE of what you'd expect**. The cold
first message is the FASTEST. Warm follow-ups are slower because the persisted
dynamic cache accumulates n-grams from one stochastic trajectory that don't
reappear in the next, so α (acceptance) collapses and rollback-failure rate rises.

## Setup

- Model: Qwen3.6-27B UD-Q2_K_XL on ROCm 7.2 / gfx1151.
- Binary: `deps/llama.cpp/build-rocm/bin/llama-lookup` (iter-11-patched).
- Prompt: canonical `prompts/prompt_code.txt` (1766 tokens, code review).
- Decode: `-n 256 --draft-max 4`, default sampling (temp=0.8, non-greedy).
- Dynamic cache: `--lookup-cache-dynamic /cache/warm.bin` (persists across invocations).
- Each docker run is a cold model-load (KV cache not shared across processes).

## Results — 3 sequential invocations, same prompt, persisted cache

| Run | Cache state             | tps     | α       | n_drafted | n_accept | rollback events |
| --- | ----------------------- | ------- | ------- | --------- | -------- | --------------- |
| 1   | **cold** (empty cache)  | **26.44** | **53.28%** | 244       | 130      | 150             |
| 2   | warm from #1            | 23.32   | 36.10%  | 241       | 87       | 291             |
| 3   | warm from #1+#2         | 24.77   | 9.67%   | 517       | 50       | 450             |

Cache file grew 156 KB -> 168 KB -> 178 KB across the three runs (n-grams accumulating).

Files: `17-run1-cold.log`, `17-run2-warm.log`, `17-run3-warm2.log`.

### Observation

- **α monotonically collapses**: 53.3% -> 36.1% -> 9.7%.
- **tps does NOT fully collapse** because run 3 drafts far MORE (517 vs 244) — the
  warm cache fires on almost every step, just with wrong guesses. Many tiny
  rejects still squeeze out a small win vs pure autoregressive decode.
- **Rollback failures scale with α collapse**: 150 -> 291 -> 450 `MTP-SEQRM NO
  checkpoint found -> seq_rm FAILED` events. Each rollback failure costs a full
  target forward, which is why run 2/3 are slower wall-clock than run 1 despite
  firing more drafts.

### Root cause (why α collapses)

Default sampling with `temp=0.8` means each run produces a DIFFERENT continuation.
Run 1's dynamic cache learns n-grams from Run 1's random output trajectory. Run 2
generates a *different* random trajectory, so most of Run 1's learned n-grams miss.
Run 3 sees cache contents from two incompatible trajectories and mostly misses.

This is fundamentally different from "the dynamic cache warms up during a single
generation" (which is beneficial — prompt-internal repetition gets learned as
generation proceeds). It's "the cache remembers things that never happen again".

## 768-token single-invocation (in-run warmup curve) — NOT MEASURED

Attempted. Crashed at ~token 700 with `GGML_ASSERT(logits != nullptr)` after
`find_slot: non-consecutive token position 2465 after 2463` — Qwen3.6 Gated
DeltaNet recurrent rollback bug under `--ignore-eos` on long generation. Log:
`17-long768.log`. This is a known hybrid-model issue (iter-7, iter-16); cannot
characterize intra-run warmup on this stack without the GDN-rollback kernel fix.

## Prompt-switch behavior

| Run | Prompt | Cache state            | tps   | α      |
| --- | ------ | ---------------------- | ----- | ------ |
| 5   | NL     | cold (fresh)           | 24.71 | 12.62% |
| 4   | NL     | warm from code-review  | 25.45 | 8.33%  (only 7 tokens, EOS'd) |

The warm-from-code-review cache gave NL α=8.3% (vs 12.6% from a fresh cold cache
on the same prompt). So cache contents from a different domain actively HURT.
This was already suspected from iter-16's static-cache sweep; here it's confirmed
for the dynamic-cache persistence path too.

**α does NOT persist usefully across context changes** — it actively degrades.

## Interpretation — what Ivan sees in a real workflow

1. **First message is the fastest**: ~26 tps on coding prompts, cold dynamic
   cache that fills in-run from prompt + generated output.
2. **Second+ messages are slower, not faster**: ~23-25 tps if the model backend
   reloads the dynamic cache from disk between requests (which persistence is
   designed to do).
3. **The longer the cache accumulates across stochastic sessions, the worse it
   gets**: run 3 hit α=9.7% and saw 450 rollback failures.
4. **Mitigation**: EITHER do not persist dynamic cache across requests (cheapest;
   restart each session cold and let it warm in-prompt) OR use greedy decoding
   (temp=0) so persisted n-grams actually match the next run's trajectory.
   Greedy has its own problem — iter-16 showed it loops at dm=4.

**Recommendation for a production serving setup**: **disable cross-session dynamic-
cache persistence**. The in-prompt warmup (prompt tokens + first few decoded
tokens) gives most of the lookup-spec win on the first response already. Keeping
a stale dynamic cache actively costs tps.

## Why tps doesn't warm UP over time (the question in the task prompt)

The hypothesis was: dynamic cache fills during generation, so tps should rise
later in a long run. **We couldn't measure this directly** because Qwen3.6
crashes past ~700 tokens of uninterrupted generation. For the 256-token window
that does work, the cache is mostly filled from the **prompt** (1766 tokens) —
that's what gives Run 1 its α=53% out of the gate. Further warmup from
additional decoded tokens is marginal because the prompt is ~7x larger than the
output window.

## Raw files

- `17-run1-cold.log` — 257 tokens, 26.44 tps, α=53.28%
- `17-run2-warm.log` — 257 tokens, 23.32 tps, α=36.10%
- `17-run3-warm2.log` — 257 tokens, 24.77 tps, α=9.67%
- `17-run4-promptswitch.log` — 7 tokens (EOS), 25.45 tps, α=8.33%
- `17-run5-nl-cold.log` — 257 tokens, 24.71 tps, α=12.62% (cold NL baseline)
- `17-long768.log` — crashed at ~token 700 (GDN rollback bug)
