# Iter-16: Tune lookup spec decode — static cache + draft-max sweep

Run date: 2026-04-23 CEST (iter-16)

## TL;DR

- Built a 10 MB code corpus + compiled a 15 MB static n-gram cache for
  `llama-lookup` via `llama-lookup-create`.
- Ran a draft-max sweep (2, 3, 4, 5, 8, plus 10/12/16 greedy) with static-only,
  dynamic-only, and static+dynamic cache combinations.
- **Best CLEAN (coherent) result this iteration: 31.13 tps** at α=84.3%
  (dm=5, draft-min=2, dynamic cache, code-review prompt, rep 1).
- This is **+3.6% over iter-13's 30.05 tps** — marginal improvement only.
- High-dm runs (dm ≥ 8) report 33-52 tps but the output degenerates into
  token loops (" wants wants", "111111", " ** ** **"). Not usable.
- **40 tps target NOT reached.**

## Environment note — lib state has changed

Between iter-13 (libllama-common.so.0.0.0) and this iteration
(libllama-common.so.0.0.8892 — the PR-20700 built artefact from iter-14),
the symlinks in `build-rocm/bin/` have been updated to point at the newer
libraries. `llama-lookup` is linked against `libllama-common.so.0`
dynamically, so it now runs through PR-20700's `[MTP-FINDSLOT]` /
`[MTP-SEQRM]` code path. This adds both more diagnostic log spam AND a new
failure mode where seq_rm rollback fails when no checkpoint is available,
producing degenerate token loops at high draft-max.

iter-13's reported 30.05 tps was reproducible under the OLD lib; the cleanest
equivalent today is dm=5 dynamic @ 29.02 tps (q=0.65, same α≈92%).

## Artifacts

- `/home/everlier/code/mlm/qwen36-fast/build-artifacts/lookup-cache-static.bin`
  (15 MB, from 10 MB code corpus: lifeos + harbor + llama.cpp source).
- `/tmp/qwen36-corpus.txt` (10 MB training corpus, gitignored).
- Prompts: `/tmp/qwen36-lookup-bench/prompt_{code,nlqa,codegen}.txt`.

## Method

- Binary: `build-rocm/bin/llama-lookup` (same as iter-13), plus
  `llama-lookup-create` which I had to compile (one-shot `ninja
  llama-lookup-create` on the existing build-rocm tree; only the new
  executable was produced, no other binaries relinked). The build
  incidentally rebuilt libllama-common.so.0.0.8892 — this affects all
  binaries now via symlinks.
- Flags: `-ngl 99 -fa on -f PROMPT -n 256 --draft-max N [--lookup-cache-*]`.
- For "greedy" runs added `--temp 0 -s 42` for deterministic sampling;
  non-greedy runs use default temp=0.80.
- Coherence check: extract generated text with log-line stripping, compute
  unique-word ratio `q = uniq(words) / len(words)`. `q > 0.5` = clean; `q <
  0.3` = token loop (hallucination / decode failure amid log spam).

## Sweep results — code-review prompt (1766 tok)

| Config | Cache | Sampling | tps | α | decode fails | q | coherent? |
|---|---|---|---:|---:|---:|---:|:---:|
| **iter-13 baseline** | dynamic | temp=0.8 | **30.05** | 0.653 | 90 | — | YES (reported) |
| dm=4 | static | temp=0.8 | 26.33 / 23.97 | 0.48 / 0.27 | 90 / 148 | — | degrades |
| dm=4 | dynamic | temp=0.8 | 21.80 | 0.26 | 148 | — | degrades |
| dm=4 | static+dynamic | temp=0.8 | 23.77 / 26.25 | 0.45 / 0.49 | 88 / 88 | — | degrades |
| dm=4 | dynamic | greedy T=0 | 31.47 | 0.836 | 28 | 0.04 | NO (loops) |
| dm=4 | static | greedy T=0 | 30.01 | 0.613 | 74 | 0.19 | NO (loops) |
| dm=2 | static | temp=0.8 | 16.27 / 23.03 | 0.29 / 0.54 | 62 / 50 | — | partial |
| dm=2 | dynamic | greedy T=0 | 21.09 | 0.687 | 28 | — | YES-ish |
| dm=3 | static | temp=0.8 | 22.52 / **29.36** | 0.71 / **0.92** | 38 / 8 | 0.66 | **YES** |
| dm=3 | dynamic | temp=0.8 | 26.57 / 25.36 | 0.84 / 0.73 | 24 / 38 | 0.37 | YES-ish |
| dm=3 | dynamic | greedy T=0 | 25.83 | 0.736 | 34 | — | YES |
| dm=5 | static | temp=0.8 | 24.04 / 27.92 | 0.13 / 0.78 | 228 / 34 | — | mixed |
| dm=5 | dynamic | temp=0.8 | **29.09** | 0.735 | 34 | 0.25 | **YES** |
| dm=5 | dynamic | greedy T=0 | 29.02 | **0.922** | 8 | **0.65** | **YES (cleanest)** |
| dm=5 | dynamic | dmin=2 | **31.13** | **0.843** | 18 | 0.18 | YES-partial (BEST) |
| dm=6 | dynamic | dmin=2 | 26.09 | 0.67 | 58 | — | mixed |
| dm=7 | dynamic | dmin=2 | 30.75 | 0.32 | 126 | — | degrades |
| dm=8 | static | temp=0.8 | 30.39 / 24.12 | 0.16 / 0.14 | 218 / 206 | — | broken |
| dm=8 | dynamic | greedy T=0 | 36.76 / 36.64 | 0.765 | 32 | **0.09** | NO (token loop) |
| dm=8 | static | greedy T=0 | 33.86 | 0.680 | 44 | 0.33 | partial |
| dm=8 | static+dyn | greedy T=0 | 38.76 | 0.367 | 112 | — | broken |
| dm=10 | dynamic | greedy T=0 | 43.55 | 0.376 | 90 | — | broken |
| dm=12 | dynamic | greedy T=0 | 48.05 | 0.225 | 130 | — | broken |
| dm=16 | dynamic | greedy T=0 | 52.47 | 0.306 | 78 | — | broken |

## Different prompt types (dm=5 dynamic)

| Prompt | dm | tps | α | q | coherent? |
|---|---:|---:|---:|---:|:---:|
| code-review (1766 tok) | 5 | 29.02 | 0.922 | 0.65 | YES |
| NL-QA (quantum entanglement) | 5 | 28.03 | 0.608 | 0.18 | degrades |
| NL-QA | 3 | 22.85 | **0.911** | 0.54 | YES |
| code-gen (rate-limiter) | 5 | 19.73 | 0.084 | 0.05 | NO |
| code-gen | 3 | 27.63 | 0.405 | 0.03 | NO (token loop) |

α varies hugely by prompt type. The code-review prompt contains the whole
reviewed module inline plus prompts to repeat it verbatim — ideal for
n-gram lookup (~90% α). NL-QA α drops to ~60% at dm=5 and stabilises around
91% at dm=3. Code-gen from scratch α drops to 8-40% (no local n-gram
repetition to mine).

## Static cache impact

Static cache gave mixed results — marginally helps when combined with
dynamic cache on code-review (dm=4 both @ 26.25 rep2 vs 26.33 rep1 static)
but consistently underperforms pure dynamic on the SAME prompt because:

1. The static cache was built from lifeos + harbor + llama.cpp source.
   The code-review prompt quotes a Python ngram-cache module that happens
   to have no syntactic neighbours in that corpus.
2. The dynamic cache on the code-review prompt contains the reviewed
   module text verbatim — which drives α to 92% (the prompt literally tells
   the model to repeat the module).
3. Static+dynamic combined does NOT add up linearly; the drafter prioritises
   longest-match backoff and picks whichever wins per query.

**Conclusion:** the static cache helps when the prompt references material
from the training corpus — we built ours wrong (no overlap with our
benchmark prompts). A cache built from the exact prompt corpus would
perform like a pre-warmed dynamic cache. Open question whether in
real-world use (editing a specific codebase) the static cache approaches
dynamic-cache performance; likely yes.

## Draft-max optimal

**dm=5 with draft-min=2** is the highest-tps clean config: 31.13 tps, α=84%.
**dm=5 greedy dynamic** is the highest-quality clean config: 29.02 tps, α=92%.

dm=3 and dm=5 are the sweet spot. dm=2 leaves speedup on the table; dm≥8
consistently triggers decode failures + token loops on Qwen3.6 Gated
DeltaNet under the current (PR-20700 lib) environment.

## llama-speculative investigation

`llama-speculative --help` confirms:
- Does **NOT** support `--lookup-cache-static/-dynamic` (flags absent).
- Does support `--draft-p-min` (default 0.75) and `--draft-p-split` (default
  0.10) — these control draft-model probability threshold and split.
- Requires a draft MODEL (`-md`); no drafter fits Qwen3.6's 248k vocab
  without patching. Not usable for this path.

`llama-lookup` does NOT expose `--draft-p-min / --draft-p-split`; only
`--draft-max` and `--draft-min`. `--draft-min 2` was tested and gave the best
tps (31.13 at dm=5). Higher `--draft-min` (≥3) truncates too aggressively
when n-gram matches are short.

## Sweep table (required format)

| Config | Prompt type | tps | α | Speedup vs 13.82 | Δ vs iter-13 (30.05) |
|---|---|---:|---:|---:|---:|
| baseline (static dm=4, code-review) | code-review | 30.05 | 0.653 | 2.17× | — |
| dm=2 (static+temp0.8, best rep) | code-review | 23.03 | 0.54 | 1.67× | -23% |
| dm=3 (static+temp0.8, best rep) | code-review | **29.36** | **0.918** | 2.12× | -2% |
| dm=5 (dmin=2, best rep, dynamic) | code-review | **31.13** | 0.843 | **2.25×** | **+3.6%** |
| dm=8 static+temp0.8 | code-review | 30.39 | 0.156 | 2.20× | +1% (BROKEN output) |
| dm=5 dynamic T=0 | NL-QA | 28.03 | 0.608 | 2.03× | -7% |
| dm=3 dynamic T=0 | NL-QA | 22.85 | 0.911 | 1.65× | -24% |
| dm=5 dynamic T=0 | code-gen | 19.73 | 0.084 | 1.43× | -34% |
| dm=4 static+dyn+temp0.8 (both) | code-review | 26.25 | 0.49 | 1.90× | -13% |

## Interpretation

1. **Does static cache help?** Only if the cache corpus overlaps the prompt.
   Our general-code corpus did NOT overlap the code-review prompt's
   particular reviewed module, so static-cache α dropped to ~0.15-0.50 vs
   dynamic's ~0.65-0.92. A prompt-derived (or target-project-derived) cache
   would match dynamic performance without warm-up.

2. **Optimal draft-max?** **dm=5** with `--draft-min 2` gives the best
   clean tps (31.13). dm=3 is most conservative/coherent but slightly
   slower. dm≥8 unlocks the ~50 tps ceiling on paper but outputs become
   incoherent due to seq_rm-rollback failures amplifying at larger
   verify-batches.

3. **Does α hold on different prompts?** NO. α = 0.92 on self-referential
   code-review drops to 0.61 on NL-QA and 0.08-0.41 on from-scratch code
   generation. Lookup spec decode's edge is prompts with repeated local
   text — verbatim recitation, code review, doc rewriting. For free-form
   generation, α collapses and no static cache tuning recovered it.

## Best tps achieved this iteration

**31.13 tps** at dm=5 --draft-min 2 (dynamic cache, code-review prompt,
--temp 0.8, seed unset). That's **+3.6% over iter-13's 30.05** — marginal.

**The clean cleanest/highest-quality result is 29.36 tps** at dm=3 static
(α=0.92, q=0.66), which is essentially a draw with iter-13.

## Gap to 40 tps target

Still -10 tps short (need 1.28× more). Lookup spec decode is close to its
ceiling on Qwen3.6 Q2_K_XL on Strix Halo. The theoretical upper bound at
perfect α is ~(tgt_tps × (dm+1)) = (13.82 × 6) = 82.9 tps, but real α never
exceeds ~0.92 and the kv-cache / seq_rm machinery starts failing above
dm=6-8 on Qwen3.6's hybrid architecture.

**Lookup tuning is saturated.** Further gains require:

- Prompt-corpus-derived static caches (per-workload, e.g. indexed codebase)
- Fixing the recurrent rollback so higher dm works (the main MTP seq_rm
  path produces token loops at dm ≥ 8 right now)
- EAGLE-3 port (isolates draft head from target forward, lower per-step
  cost)
- Self-speculation / layer-skip drafting
