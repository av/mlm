# iter-26 Workload-diversity benchmark

**Date**: 2026-04-23

## Canonical config

Same flags as `bench/run-best.sh` full-mode, plus `--ignore-eos` so every
workload runs the full 256-token budget (otherwise codegen/nl hit EOS after
a handful of tokens and the tps reading collapses):

```
llama-lookup \
    -m Qwen3.6-27B-UD-Q2_K_XL.gguf \
    -ngl 99 -fa on \
    -f prompts/prompt_<workload>.txt \
    -n 256 --draft-max 4 --ignore-eos
```

Default sampling (temp=0.8, top-k/top-p defaults) — same as run-best.sh, same
as how Ivan will actually use the model. Dynamic n-gram cache only (no static
cache). Iter-11 patched binary at `deps/llama.cpp/build-rocm/bin/llama-lookup`.

## Per-workload results

| Workload | prompt tokens | tps rep1 | tps rep2 | alpha rep1 | alpha rep2 | Output sanity |
|---|---:|---:|---:|---:|---:|---|
| code-review | 1766 | 29.21 | 25.35 | 0.74 | 0.49 | OK |
| code-generation | 1080 | 23.49 | 26.63 | 0.16 | 0.64 | OK |
| chat | 954 | 26.66 | 32.03 | 0.58 | 0.87 | OK |
| nl-summary | 553 | 21.30 | 29.81 | 0.22 | 0.58 | OK |

## Aggregate statistics

- N = 8 runs across 4 workloads x 2 reps
- Mean tps:    **26.81**
- Stddev tps:   3.50
- Min tps:      21.30
- Max tps:      32.03
- **Range: 21.3 -- 32.0 tps**
- Mean +/- 1 stddev: **23.3 -- 30.3 tps**

## Per-workload mean tps (rep1 and rep2 averaged)

| Workload | mean tps | mean alpha | baseline (Q2_K_XL no-spec) | speedup |
|---|---:|---:|---:|---:|
| code-review | 27.28 | 0.62 | 13.82 | 1.97x |
| code-generation | 25.06 | 0.40 | 13.82 | 1.81x |
| chat | 29.34 | 0.72 | 13.82 | 2.12x |
| nl-summary | 25.56 | 0.40 | 13.82 | 1.85x |

## Ivan-facing interpretation

- **Honest range: 21--32 tps depending on workload**.
- Typical code-review (self-referential, repetitive): **~27 tps** (iter-13 canonical regime).
- Conversational chat with history: **~29 tps** (benefits from repeated keywords in the history).
- Code generation from a spec (no source to quote): **~25 tps** (lookup fires less, alpha 0.15--0.65).
- Natural-language translation / summarization: **~26 tps** (lowest alpha, shortest prompt).

Mean across all 4 workload types: **~27 tps**.

## Caveats

1. **Rep-to-rep variance is real**: sampling at temp=0.8 wanders into different
   regions of output space, which changes how often the dynamic n-gram cache
   fires. For code-review we saw 29.21 and 25.35 tps in back-to-back runs; for
   chat we saw 26.66 and 32.03 tps. This is not noise in the decoder; it is the
   real distribution a user observes. Expect ~4--7 tps swing run-over-run on
   the same prompt.
2. **Baseline (no spec decode) on Q2_K_XL is 13.82 tps** (iter-8,
   `bench/03-low-bit-quants.md`). Every workload here beats baseline, but the
   worst case (NL-summary rep1, 21.30 tps) is only 1.54x over baseline. Do not
   tell users 'always 30 tps'; the realistic expectation is **1.5x--2.3x**
   speedup depending on workload repetition structure.
3. **Greedy decoding (`--temp 0`) degenerates output at `--draft-max 4`** on
   all four workloads (see the aborted greedy-mode run earlier in this
   iteration; all tails became 'wants wants' / 'rationale rationale' style
   token loops). Greedy is fine for measuring alpha but not for usable output.
   **Use default sampling in production.** run-best.sh already does.
4. **M-RoPE `decode: failed` + `[MTP-SEQRM] NO checkpoint found` log lines are
   normal** — PR #19493 checkpoint-restore handles them silently. They scale
   with draft-max; at dm=4 expect 60--120 per 256-token decode.
5. **Prompt length matters indirectly**: our NL prompt is 553 tokens (much
   smaller than the other three at 954--1766) which means a smaller dynamic
   cache — and thus lower alpha. In practice, longer prompts of the same
   workload type would cluster slightly higher.
6. **`--ignore-eos` was required** to force all 4 workloads to decode the full
   256 tokens for an apples-to-apples measurement. Without it, codegen and NL
   emit `<|im_end|>` within the first few tokens under default sampling and
   we cannot measure steady-state throughput. Production callers do not use
   `--ignore-eos`; this is a benchmarking necessity.

## Recommendation for the MORNING / README honesty update

Replace "~30 tps" with a range: **"21--32 tps depending on workload (coding: ~27, chat: ~29, new code: ~25, NL: ~26)"**.

Baseline (no speculative decoding on UD-Q2_K_XL): **13.82 tps**. Speedup
range across the 4 workloads: **1.54x -- 2.32x**.
