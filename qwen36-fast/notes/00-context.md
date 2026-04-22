# Context snapshot (from orchestrator)

- Model target: Qwen/Qwen3.6-27B (dense, hybrid Gated DeltaNet 3:1 Gated Attention, 64 layers, bf16, 256K native ctx). Apache-2.0.
- Quants: official FP8; community Q4_K_M (Unsloth ~16.8 GB), GGUFs on HF.
- Bandwidth ceiling: Q8 → ~9.5 tps peak, Q4 → ~18 tps peak, Q2 → ~36 tps peak on this machine.
- Baseline user-reported: ~5 tps Q8, ~15 tps Q2 (backend unconfirmed, verify in Phase 1).
- Target: ~40 tps → only reachable via speculative decoding (multiplier on bandwidth-bound decode).
- DFlash (arxiv 2602.06036) = the hot paper. Only vLLM/SGLang/MLX. Drafter published for 35B-A3B MoE, not 27B dense. No ROCm/Vulkan/llama.cpp port.
- Qwen3.6 was trained with MTP head — if it survived into the released checkpoint it's a cheap drafter.
- Harbor already manages llama.cpp in Docker on this machine. No `llama-cli` on PATH. rocm-clang at /usr/lib64/rocm/llvm/bin. No Vulkan SDK (glslang/shaderc) installed, available via dnf.
- Sudo pass available via env (never persist, never commit).

## Phased plan

**Phase 1 — measure + vanilla spec decoding (first 2h):**
1. Set up `/home/everlier/code/mlm/qwen36-fast/` + commit.
2. Pull Unsloth Qwen3.6-27B Q4_K_M GGUF into Harbor model cache.
3. Benchmark baseline on llama.cpp Vulkan (Harbor). tps at decode, short + long ctx.
4. Inspect GGUF tensors for MTP head presence.
5. Try vanilla draft-model speculative decoding: target 27B Q4, draft Qwen3-0.6B Q4. Report acceptance + tps.

**Phase 2 — Gated DeltaNet kernel work (middle 2-3h):**
6. Reference CPU impl of one Gated DeltaNet layer forward pass, validated against HF.
7. Vulkan compute shader for dequant-matvec.
8. Vulkan fused DeltaNet+FFN shader, validated against reference.

**Phase 3 — drafter training / DFlash port planning (last hour):**
9. If MTP head present: wire into llama.cpp spec decoding.
10. If not: sketch EAGLE-3 head architecture suitable for 27B.
11. Plan doc for DFlash block-verify port.
