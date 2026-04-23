# Final state — 2026-04-23 06:00 CEST

## Quick facts
- Iterations: 29
- Commits on main (ahead of origin): 27
- Working tree: clean
- Elapsed: ~5h40m

## Best reproducible result
- 30.21 tps (canonical single-shot, iter-20) / 26.8 tps mean across 4 workloads (iter-26)
- Config: UD-Q2_K_XL + llama-lookup + draft-max 4/5 + dynamic cache
- Command: `./bench/run-best.sh`

## Known-good state
- Binary: deps/llama.cpp/build-rocm/bin/llama-lookup (PR #20700 + iter-11 patch applied)
- GGUF: ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/.../Qwen3.6-27B-UD-Q2_K_XL.gguf (11.85 GiB)
- Prompt: prompts/prompt_code.txt (the 30 tps canonical prompt)

## File counts
- bench/*.md: 15
- bench/*.log: 93
- notes/*.md: 11
- patches/: 7 items (5 patches + Dockerfile + inject_mtp.py + upstream-pr-draft/)
- bin/: 2 (push-to-origin.sh, file-upstream-pr.sh)

## Disk footprint
- qwen36-fast/: 13 GiB (12 GiB untracked regeneratable MTP merged GGUF)
- ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/: 50 GiB (4 quants; 38 GiB prunable)

## Docs reading order
1. MORNING.md
2. README.md
3. CHANGELOG.md
4. notes/08-final-state.md
5. notes/09-eagle3-future-path.md
6. patches/upstream-pr-draft/PR-DESCRIPTION.md

## Todo for Ivan
- [ ] Decide: push to origin? (bin/push-to-origin.sh)
- [ ] Decide: file the upstream PR? (bin/file-upstream-pr.sh prints the gh command)
- [ ] If pursuing 40 tps: start with notes/09-eagle3-future-path.md
- [ ] Prune old GGUFs? See notes/10-disk-usage.md (38 GiB prunable)

## Hazards / gotchas
- MTP regresses on Strix Halo: per-step +MTP-forward cost exceeds single-token savings on 256 GB/s bandwidth even at α=1.00 (iter-15, iter-18 definitive ruling; structural K=1 cap in PR #20700).
- Dynamic n-gram cache does NOT warm up cross-session: α collapses 53% → 10% across reruns at temp=0.8 (iter-29). Disable cross-session persistence in serving.
- Harbor ROCm compose has pre-existing argv bug (`argv[0]="-m"`); bypassed via direct docker run throughout (iter-5).
- Binary was relinked during iter-14 (PR #20700 applied on top) — iter-13 (30.05) and iter-16 (31.13) numbers not strictly comparable but within noise; iter-20 canonical (30.21) reflects current binary.
- Greedy (`--temp 0`) degenerates all 4 workloads into token loops at dm=4 (iter-26); default sampling required for clean output.
