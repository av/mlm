# Upstream PR draft — `common: relax can_seq_rm probe`

This folder contains a ready-to-file upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The draft
was polished during the `qwen36-fast` timebox (iter-22) after the
underlying patch was validated end-to-end via `llama-server` in iter-21.

## Contents

- `0001-common-relax-can_seq_rm-probe-to-try-checkpoint-roun.patch` —
  squashed `git format-patch`-formatted commit. 32 added / 2 removed
  lines in `common/common.cpp`. Applies cleanly (`git apply --check`
  and `git am`) to llama.cpp master as of
  `86db42e CUDA: fuse relu + sqr (#22249)`.
- `PR-DESCRIPTION.md` — the GitHub PR body to paste when filing. Covers
  problem, fix, before/after log, empirical validation, compatibility,
  testing, risks, and open questions.
- `README.md` — this file.

## Before filing

1. Edit the `From:` and `Signed-off-by:` lines in the `.patch` file to
   your real name / email (currently placeholders).
2. Re-run `git apply --check 0001-...patch` against a fresh clone of
   `ggml-org/llama.cpp` master to confirm the patch still applies on
   whatever tip exists when you file — trivial maintenance.
3. `git am 0001-...patch` on a feature branch, push, then
   `gh pr create --title "common: relax can_seq_rm probe to try
   checkpoint round-trip" --body-file PR-DESCRIPTION.md`.

## Provenance

- Authored in `qwen36-fast/patches/llamacpp-qwen36-spec-decode.patch`
  (iter-11).
- Built + validated on ROCm 7.2 / gfx1151 via `llama-lookup` (iter-13,
  30.05 tps, α = 65.29 %) and `llama-server` (iter-21, α = 1.00,
  checkpoint-promotion log line confirmed).
- Full investigation trail: `qwen36-fast/notes/` and
  `qwen36-fast/bench/` in this repo.
