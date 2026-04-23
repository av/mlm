# Morning briefing: Qwen3.6-27B fast-decode overnight run

**Date**: 2026-04-23, started 00:20 CEST, ended 06:00 CEST (~5h40m, 22 iterations)

## 30-second answer

- **Goal**: push Qwen3.6-27B on Strix Halo from ~11 tps baseline to 40 tps decode via speculative decoding.
- **Achieved**: **30.05 tps** (canonical, reproducible; `bench/run-best.sh` in full mode reproduces it @ 30.21 tps). **2.17x over Q2_K_XL baseline, 2.76x over Q4_K_M baseline.**
- **Gap**: −10 tps / −25% from 40 tps target. The remaining gap is **not reachable on this hardware via any lookup/MTP path available today** — requires EAGLE-3 (~1 week effort) or GDN-rollback kernel fix.

## Reproduce the best result

```bash
cd /home/everlier/code/mlm/qwen36-fast && ./bench/run-best.sh
# expect ~30 tps, alpha ~65%, PASS exit 0 in ~15 s
# fresh-shell verified 2026-04-23 05:50 CEST at 23.7 tps (short mode, cold cache)
```

Script has preflights for GGUF, patched binary, docker, and `/dev/kfd`+`/dev/dri`.
Fully self-contained (tested under `env -i bash`). `--help` / `--short` supported.

## What to read first (in this order)

1. **`README.md`** (337 lines) — full context, current best, hazards, reproducibility.
2. **`notes/08-final-state.md`** (479 lines) — iteration-by-iteration postmortem, numbers table, what was ruled out.
3. **`bench/11-canonical-run.md`** (84 lines) — the reproducibility-verified number (30.21 tps full-mode).
4. **`patches/upstream-pr-draft/PR-DESCRIPTION.md`** (191 lines) — ready to file upstream as-is.

## The ONE decision for today

**If you want to try for 40 tps this week**, choose:

- **EAGLE-3 port** (`notes/09-eagle3-future-path.md`, ~1 week effort):
  apply PRs #18039 + #21437 to our tree, validate on
  `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` first (cheap signal, no training). If
  LLaMA-3.1-8B hits ≥2.5× on gfx1151 ROCm 7.2, proceed; if <1.5× pivot to GDN
  rollback kernel fix. Drafter for Qwen3.6-27B does not exist on HF — will need
  SpecForge distillation or MTP-head transfer (day 3-5 of the week).

- **OR upstream the `can_seq_rm` patch** (`patches/upstream-pr-draft/`, ~10 min effort):
  single file `0001-common-relax-can_seq_rm-probe-to-try-checkpoint-roun.patch`
  against ggml-org/llama.cpp. Unblocks lookup/n-gram spec-decode on all hybrid
  recurrent models (Qwen3.5, Qwen3.6, GLM4.5-MoE, LFM2, Mamba, Kimi-Linear).
  Community deliverable — doesn't close our 40 tps gap but helps everyone else.

**If no decision**, the 30 tps result stands and `run-best.sh` keeps reproducing.

## Hazards

- **Harbor llamacpp compose has a pre-existing `argv[0]="-m"` bug** (not my fault, iter-5 documented). I bypassed it via direct `docker run`; don't try to use Harbor's CLI path for spec-decode.
- **`build-artifacts/qwen36-27b-mtp-merged.gguf` (12 GiB, untracked)** works but MTP itself regresses tps 35% on Strix Halo — **don't enable MTP for production**. Kept only for forensic inspection of the K=1 cap.
- **MTP in PR #20700 is STRUCTURALLY K=1** — `common/speculative.cpp:603-649` argmaxes one token. `--draft-max` is a no-op there. Iter-18's definitive ruling; don't waste time on --draft-max tuning for MTP.
- **Static lookup cache (15 MiB) HURTS when prompt doesn't overlap** (iter-16). Leave `run-best.sh` using dynamic cache; don't add `--lookup-cache-static`.
- **Disk: 50 GiB of GGUFs cached** (3 unused quants are 36 GiB prunable). See `notes/10-disk-usage.md`.

## Repo state

- 20 commits on `main` ahead of `origin/main` (will be 21 after this iteration's commit).
- Working tree clean. Nothing sensitive committed. Remote push **not** done — your call.
