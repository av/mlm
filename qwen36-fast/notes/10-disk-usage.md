# 10 — Disk usage snapshot (2026-04-23 morning)

Recorded for Ivan to decide what to prune after the overnight run.

## Working tree: `qwen36-fast/`

| Path | Size | Tracked? | Notes |
|---|---|---|---|
| `qwen36-fast/` (total) | 13 GiB | mixed | includes build + deps + artifacts below |
| `qwen36-fast/build-artifacts/` | 12 GiB | no (.gitignored) | merged MTP GGUF + static lookup cache |
| &nbsp;&nbsp;`qwen36-27b-mtp-merged.gguf` | 12 GiB | no | regeneratable via `patches/inject_mtp.py` |
| &nbsp;&nbsp;`lookup-cache-static.bin` | 15 MiB | no | corpus-derived, regeneratable |
| `qwen36-fast/deps/llama.cpp/` | 757 MiB | no (.gitignored) | shallow-clone + patched source |
| &nbsp;&nbsp;`deps/llama.cpp/build-rocm/` | 256 MiB | no | ROCm binaries — NEEDED for `run-best.sh` |

Source files + notes + patches + logs (the actually-tracked content) fit in
a few MiB — negligible.

## HuggingFace cache: `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/`

| File | Size | Needed for 30 tps repro? | Recommendation |
|---|---|---|---|
| `Qwen3.6-27B-UD-Q2_K_XL.gguf` | 12 GiB | **YES — THIS is the canonical model** | **KEEP** |
| `Qwen3.6-27B-Q3_K_S.gguf`     | 12 GiB | no (quant-sweep only, bench/03) | optional, can prune |
| `Qwen3.6-27B-UD-IQ3_XXS.gguf` | 12 GiB | no (quant-sweep only, bench/03) | optional, can prune |
| `Qwen3.6-27B-Q4_K_M.gguf`     | 16 GiB | no (baseline was measured here, but 30 tps is on Q2_K_XL) | optional |
| **Cache total** | **50 GiB** | | |

## Pruning plan (38 GiB recoverable)

If disk pressure materializes:

```bash
# Safe: drop 3 redundant GGUFs (keeps UD-Q2_K_XL, the 30 tps base)
rm ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/blobs/{4afb4abcf0207a484b0d7e92c0421b74e8ce1c7a7250bb9d824b79288da68f20,5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0,5d591dd11918e196a7b7c9d2f02e4390e7264960eb354c72d65e81a9331978f5}
# Recovers ~40 GiB. Re-downloadable with huggingface-cli if ever needed.

# Safe: drop regeneratable MTP merge (iter-15 ruling: path abandoned)
rm /home/everlier/code/mlm/qwen36-fast/build-artifacts/qwen36-27b-mtp-merged.gguf
# Recovers 12 GiB. If someone resumes MTP work, regenerate via patches/inject_mtp.py
# from the HF safetensors shards at /tmp/qwen36-mtp-shards/ (4.29 GiB).
```

## Do NOT delete

- `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/.../Qwen3.6-27B-UD-Q2_K_XL.gguf`
- `qwen36-fast/deps/llama.cpp/build-rocm/bin/llama-lookup` (the patched binary)
- `qwen36-fast/prompts/prompt_code.txt` (reproducibility prompt)
- `qwen36-fast/patches/*.patch` (tracked in git; deleting from worktree re-creatable via `git checkout`)

## Re-download / rebuild costs

| Item | Recovery cost |
|---|---|
| UD-Q2_K_XL GGUF | ~11 GiB network pull, ~8 min on gigabit |
| llama.cpp build-rocm | ~8-12 min full rebuild in toolbox Docker |
| MTP merged GGUF | ~30 s via `python patches/inject_mtp.py` given shards |
| Lookup static cache | ~2 min given source corpus |

Everything is reproducible from tracked sources + public HF downloads; nothing
in the untracked 13 GiB is precious.
