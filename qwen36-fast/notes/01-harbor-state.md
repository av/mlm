# Harbor + GGUF state snapshot (2026-04-23 00:2x CEST)

## Harbor running services (`harbor ps`)

```
NAME              IMAGE                                     COMMAND                SERVICE    STATUS
harbor.llamacpp   kyuz0/amd-strix-halo-toolboxes:rocm-7.2   llama-server ...       llamacpp   Up 3 hours (healthy)  :33831 -> 8080
harbor.ollama     ollama/ollama:rocm                        /bin/ollama serve      ollama     Up 3 hours (healthy)  :33821 -> 11434
harbor.webui      ghcr.io/open-webui/open-webui:main        start_webui.sh         webui      Up 3 hours (healthy)  :33801 -> 8080
```

llama.cpp image is the `kyuz0/amd-strix-halo-toolboxes:rocm-7.2` Strix Halo toolbox. ROCm 7.2 backend — NOT Vulkan as the plan assumed. Revisit Phase 1 step 3 accordingly (measure ROCm baseline; Vulkan is a separate test).

## Harbor llamacpp config

- `harbor llamacpp model` → (empty; no HF URL configured)
- `harbor llamacpp gguf` → `/root/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-UD-IQ3_XXS.gguf`
- `harbor llamacpp args` → `llama-server --no-mmap -dio -ngl 99 -c 256000 -np 4 --kv-unified`
- Build-from-source: not enabled (uses the pre-built toolbox image).

Currently-loaded model is **Qwen3-Coder-Next** IQ3_XXS, not the Qwen3.6-27B target. We will need to switch either by `harbor llamacpp gguf <path>` or by running a second llama-server instance.

Notable arg flags:
- `--no-mmap` — full VRAM load
- `-dio` — direct IO
- `-ngl 99` — all layers on GPU
- `-c 256000` — 256K ctx
- `-np 4` — 4 parallel slots
- `--kv-unified` — unified KV across slots

## llama.cpp model cache state

Harbor-native cache dir `/home/everlier/.cache/harbor/models/llama.cpp/` is **empty** (only dir stub). Harbor's llamacpp container uses the image-internal `/root/.cache/llama.cpp/` instead, which is inside the container filesystem. The actual GGUFs live in the HF cache (see below), mounted in.

## HuggingFace cache — GGUFs present

```
/home/everlier/.cache/huggingface/hub/
  models--unsloth--Qwen3-Coder-Next-GGUF/       (loaded model; IQ3_XXS)
  models--unsloth--Qwen3.5-122B-A10B-GGUF/
  models--unsloth--Qwen3.6-35B-A3B-GGUF/        21 GB blob + 861 MB mmproj
```

Files in the 35B-A3B cache:
- `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (~21 GB)
- `mmproj-BF16.gguf` (~861 MB)

## Is Qwen3.6-27B present? NO

`find ~ -iname "*qwen*3.6*"` matches only the **35B-A3B MoE**, not the 27B dense target.

No `Qwen3.6-27B` directory exists anywhere under /home/everlier.
No `Qwen3-0.6B` draft model GGUF cached either (needed for vanilla spec-decode experiments in Phase 1.5). Only `Qwen3-Embedding-0.6B` (wrong head) is present, under `/home/everlier/code/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B`.

## Implications for next iteration

1. Need to `harbor pull unsloth/Qwen3.6-27B-GGUF:Q4_K_M` (or a specific quant) — ~17 GB download.
2. Need a small draft model too: either `Qwen3-0.6B` or `Qwen3-1.7B` in Q4/Q6. Add to pull.
3. Backend is **ROCm 7.2** (hipBLAS path), not Vulkan — adjust bandwidth-ceiling estimates; ROCm on gfx1151 may be tighter or looser than Vulkan. Baseline numbers in the context snapshot assumed Vulkan.
4. Currently running `Qwen3-Coder-Next` is live on port 33831. Either stop it before loading 27B or spin a second llama-server on a different port. If we reuse the same container we lose coder-next; probably fine to reconfigure for the experiment window.
5. `--kv-unified -np 4` is incompatible with speculative decoding in stock llama.cpp (draft + target are separate model instances). Will need a different args profile for the spec-decode test.
