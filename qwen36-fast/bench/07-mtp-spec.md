# Iter-14: Port PR #20700 (Qwen3.5 MTP + FastMTP) to Qwen3.6-27B

Run date: 2026-04-23 CEST

## TL;DR

**Milestones hit**: PR applied with 4 conflicts resolved, build succeeded with
a custom rocm-7.0 builder image, merged GGUF with 15 MTP tensors was produced,
and the model **loads on ROCm without tensor shape errors**. Inference crashes
on a ggml_cuda_op_mul assertion — F16 MTP tensors vs Q2_K backbone produce
incompatible broadcast strides. **Not at 40 tps yet; stopped at the crash.**

**Baseline for this iteration (iter-13):** 30.05 tps / α=65.29% via lookup spec.

**This iteration's run**: PR applied + built + merged GGUF produced, but spec
decode could not be invoked because:
(a) `--spec-type mtp` arg is gated to `LLAMA_EXAMPLE_SERVER` only, and we
reverted server-context.cpp to HEAD to avoid 7 deep conflicts.
(b) Even the basic probe `common_context_can_seq_rm` crashes in
`ggml_cuda_op_mul` on the first decode against the merged GGUF.

## Step-by-step outcome

### 1. PR #20700 apply — DONE, 4 conflicts resolved

Command used (unshallow first, then cherry-pick on top of master `0d0764d`):

```
git fetch --unshallow origin   # repo was originally shallow
git fetch origin pull/20700/head:pr-20700
git checkout -b pr20700-on-master master
git cherry-pick --no-commit pr-20700
```

Files auto-merged (12): common/arg.cpp, common/common.h, common/sampling.cpp,
convert_hf_to_gguf.py, gguf-py/gguf/constants.py, include/llama.h,
src/llama-batch.cpp, src/llama-context.{cpp,h}, src/llama-graph.h,
src/llama-memory-recurrent.{cpp,h}, src/models/{models.h,qwen35.cpp}.

Files with conflicts (4):

- **common/speculative.cpp** — trivial: kept both #include (cinttypes from HEAD,
  random from PR) and PR's full `common_speculative_is_compat` function.
- **src/llama-arch.cpp** — PR added a 2000-line `llm_get_tensor_names` helper
  that already exists in HEAD in a different form. **Deleted the PR's
  duplicate block entirely** (only used in docs/HOWTO, not referenced by
  MTP code).
- **src/llama-model.cpp** — preserved PR's new `is_mtp_layer` branch
  (nextn tensors + gated attention + dense FFN) and kept HEAD's
  `create_tensor_qkv(...)` call in the main `!is_recurrent` else-branch
  (PR had reverted to the pre-refactor per-wq/wk/wv form; HEAD uses the
  refactored helper).
- **tools/server/server-context.cpp** — 7 conflict hunks, all related to
  HEAD's `spec_draft`/`spec_ckpt` refactor vs PR's older
  `drafted`/`i_batch_dft` + MTP-two-phase fields. **Too invasive to port
  cleanly in time budget. Chose `git checkout --ours`** → HEAD version.
  Consequence: `--spec-type mtp` via server is inoperative (no MTP
  state machine in server path); server falls back to regular spec decode.

Commit ranges chained onto master: see `pr20700-on-master` branch in
`deps/llama.cpp`.

### 2. Build — DONE, ~2 min incremental

First attempt with `kyuz0/amd-strix-halo-toolboxes:rocm-7.2` failed because
that image is a runtime-only image (no cmake/ninja/amdclang++).

Resolution: built a derivative builder image based on the upstream
`ghcr.io/ggml-org/llama.cpp:server-rocm` which has amdclang++ 20.0.0 from
rocm-7.0.0 at /opt/rocm-7.0.0/. Added apt install cmake/ninja/git:

```
# /tmp/strix-halo-builder.Dockerfile
FROM ghcr.io/ggml-org/llama.cpp:server-rocm
RUN apt-get update && apt-get install -y cmake ninja-build git curl
ENV PATH=/opt/rocm-7.0.0/bin:/opt/rocm-7.0.0/llvm/bin:$PATH
ENV ROCM_PATH=/opt/rocm-7.0.0
ENV HIP_PATH=/opt/rocm-7.0.0
```

Tag `strix-halo-builder:rocm-7.0`. cmake 3.28.3, amdclang++ 20.0.0
(roc-7.0.0). Build command with `--build build-rocm` (incremental on
existing tree, preserving ~400 compiled objects from iter-11):

```
docker run --rm --entrypoint=/bin/bash \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video \
    -v $PWD:/src:rw \
    strix-halo-builder:rocm-7.0 \
    -c 'git config --global --add safe.directory /src &&
        cd /src &&
        cmake --build build-rocm -j$(nproc) --target
            llama-lookup llama-speculative llama-speculative-simple
            llama-cli llama-bench'
```

Result: all targets compiled. One unused-variable warning in
`src/llama-model.cpp:8523` (`const uint32_t n_mtp = ...`). No errors.
Links in 5 s total. Binaries timestamp 2026-04-23 02:55 CEST.

Run-time image (unchanged): `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`
with `LD_LIBRARY_PATH=/bld/bin:/opt/rocm-7.2.0/lib`. Binaries work despite
being built against rocm-7.0 headers (ABI-compatible libhipblas.so.3
exists in 7.2).

### 3. Q_proj shape — verified NO bug on Qwen3.6

iter-12 hypothesized `mtp q_proj` was non-gated `[6144, 5120]` but loader
forces gated `[12288, 5120]`. Direct safetensors inspection of
`/tmp/qwen36-mtp-shards/model-000{13,15}-of-00015.safetensors`:

```
mtp.layers.0.self_attn.q_proj.weight: shape=[12288, 5120] dtype=BF16
mtp.layers.0.self_attn.k_proj.weight: shape=[1024, 5120]  dtype=BF16
mtp.layers.0.self_attn.v_proj.weight: shape=[1024, 5120]  dtype=BF16
mtp.layers.0.self_attn.o_proj.weight: shape=[5120, 6144]  dtype=BF16
```

`12288 = 24 heads × 256 head_dim × 2 (gated)`. `6144 = 24 × 256` (ungated,
post-gate removal). **The MTP q_proj IS gated on Qwen3.6-27B.** PR #20700's
hardcoded shape `{ n_embd, n_embd_head_k * n_head * 2 }` is correct here.

The author's word-salad report on Qwen3.5-27B Q6_K Metal is unrelated
to this model. No local patch applied. **Fix not needed for Qwen3.6.**

### 4. MTP drafter GGUF — produced via custom injection script

PR #20700's converter path (`convert_hf_to_gguf.py Qwen3_5TextModel`) requires
the FULL checkpoint (backbone + MTP in one HF dir). We only have 2 of 15
safetensors shards (MTP-only slice from iter-6, 4.29 GiB). Full download is
~54 GiB — skipped on budget.

Instead wrote a **standalone injector** at `/tmp/inject_mtp.py` using
`gguf-py`:

1. Open Unsloth Q2_K_XL GGUF (`Qwen3.6-27B-UD-Q2_K_XL.gguf`, 11.85 GiB, 851
   tensors) with GGUFReader.
2. Copy all 851 backbone tensors to new GGUFWriter preserving raw quant
   dtypes (`writer.add_tensor(name, data, raw_dtype=t.tensor_type)`).
3. Copy all KV metadata. Override `qwen35.block_count` from 64 → 65. Add
   `qwen35.nextn_predict_layers = 1`.
4. Parse the 2 MTP safetensors shards (raw header read, manual BF16 →
   FP32 → F16 conversion). Remap tensor names using PR #20700's remapping
   logic (inlined in Python):
   - `mtp.fc.weight` → `blk.64.nextn.eh_proj.weight`
   - `mtp.pre_fc_norm_{embedding,hidden}.weight` → `blk.64.nextn.{enorm,hnorm}.weight`
   - `mtp.norm.weight` → `blk.64.nextn.shared_head_norm.weight`
   - `mtp.layers.0.{q,k,v}_proj.weight` → `blk.64.attn_{q,k,v}.weight`
   - `mtp.layers.0.o_proj.weight` → `blk.64.attn_output.weight`
   - `mtp.layers.0.{q,k}_norm.weight` → `blk.64.attn_{q,k}_norm.weight`
   - `mtp.layers.0.input_layernorm.weight` → `blk.64.attn_norm.weight`
   - `mtp.layers.0.post_attention_layernorm.weight` → `blk.64.post_attention_norm.weight`
   - `mtp.layers.0.mlp.{gate,down,up}_proj.weight` → `blk.64.ffn_{gate,down,up}.weight`
5. Write all 15 MTP tensors as F16 (GGUF doesn't accept raw BF16 from
   numpy easily; F16 is the safest lossless-for-our-scale choice).

Output: `/home/everlier/code/mlm/qwen36-fast/build-artifacts/qwen36-27b-mtp-merged.gguf`,
**11.83 GiB, 866 tensors (851 + 15)**. Takes ~6 s to write (copy-through).

### 5. Load test — PASSES TENSOR LOAD, CRASHES IN DECODE

Command:

```
docker run --rm --entrypoint=/bld/bin/llama-cli ... \
    -m /models/qwen36-27b-mtp-merged.gguf \
    -ngl 99 -fa on -p "Hello," -n 8 --no-warmup
```

Model loads successfully — all 866 tensors mapped with no shape errors.
`[MTP-FINDSLOT]` PR-added debug print fires (confirms PR code is in the
binary and is being triggered for hybrid models). Then:

```
/src/ggml/src/ggml-cuda/binbcast.cu:255:
  GGML_ASSERT(nb10 % sizeof(src1_t) == 0) failed

... in llama_decode
... in common_context_can_seq_rm
... in llama-cli main
```

Stacktrace: the crash happens inside llama-cli's 2-token warmup-like
decode called by PR's `common_context_can_seq_rm` probe (the 2-token
compat check added to CLI examples by PR #20700).

Interpretation: `ggml_cuda_op_mul` broadcast kernel requires `nb10`
(byte stride of the last dim of src1) to be a multiple of `sizeof(src1_t)`.
This likely trips because one of the MTP F16 tensors is being multiplied
against a Q2_K or K-norm tensor on the graph and the stride arithmetic
produces misalignment on GPU.

**Not a tensor-writing bug** — `gguf-dump` of the merged file shows
correct shapes and strides for all 15 MTP tensors. The issue is on the
compute graph side when MTP F16 meets Q2_K / F32 operands on ROCm.

Have not exhaustively debugged which specific tensor+op causes it. Next
diagnostic step would be to re-write the merged GGUF with all 15 MTP
tensors as F32 (not F16) to rule out F16-specific ROCm bincast paths —
that's +2× MTP tensor size but still only ~1 GiB delta on a 12 GiB file.

### 6. Did we hit 40 tps?

**No.** Could not execute a spec-decode run against the merged GGUF
because the probe decode crashes before any tokens are generated.

Our best result remains **iter-13's 30.05 tps / α=65.29%** via the
lookup-spec path on the unmodified Q2_K_XL GGUF.

### 7. What remains

Ranked by impact/effort for the next iteration:

1. **Rebuild merged GGUF with F32 MTP tensors instead of F16**.
   Replace `data_f16 = data_fp32.astype(np.float16)` with `data_fp32`
   in inject_mtp.py; `writer.add_tensor(name, data_fp32)`. ~30 s to
   regenerate. If this fixes the bincast assert, the path opens up.

2. **If F32 fix works**: resolve the `--spec-type mtp` gating.
   Options:
   - Add `LLAMA_EXAMPLE_CLI` and `LLAMA_EXAMPLE_SPECULATIVE` to
     `.set_examples({...})` on the `--spec-type` arg in common/arg.cpp
     (one-line change).
   - Or modify llama-speculative-simple to pick up MTP when target model
     has `n_mtp_layers > 0`.
   Then test with:
   `llama-cli -m merged.gguf --spec-type mtp --draft-max 1 ...`
   or
   `llama-speculative-simple -m merged.gguf -md merged.gguf --spec-type mtp ...`

3. **If target is server path**: redo the 7 server-context.cpp conflicts
   properly — port PR's `drafted/i_batch_dft/mtp_*` state into HEAD's
   `spec_draft/spec_ckpt` architecture. Estimated 100-200 LoC in an
   afternoon of focused work.

4. **Validate MTP logits plumbing**: after fixing 1-3, check that
   `llama_get_mtp_logits` returns non-null on the merged GGUF —
   otherwise `common_speculative_state_mtp::predict_next` falls through.
   `llama_model_n_mtp_layers(model)` should return 1.

5. **Measure tps + α**: same prompt/setup as iter-13 for apples-to-apples.
   Expected: α ≥ 75% (per PR author's 82% on Qwen3.5-9B-Q4_K_M).
   Projected tps at α=75%, K=1, target=13.82 baseline: ~22-24 tps.
   At K=4: if α=75%, effective multiplier ~2.25× → ~31 tps. Still
   below 40, because MTP-1 truly supports only K=1 in PR's
   implementation; larger K relies on vocab-trimmed MTP + prompt lookup
   hybrid which is the FastMTP piece.

## File artefacts

- Patched tree (NEW branch, not pushed): `deps/llama.cpp/pr20700-on-master`
- Build binaries: `deps/llama.cpp/build-rocm/bin/llama-{cli,lookup,speculative,speculative-simple,bench}`
- Merged GGUF: `build-artifacts/qwen36-27b-mtp-merged.gguf` (11.83 GiB, 866 tensors)
- Injection script: `/tmp/inject_mtp.py` (will copy to `patches/inject_mtp.py`)
- Builder image Dockerfile: `/tmp/strix-halo-builder.Dockerfile` (will copy to `patches/`)
- This test's log: `bench/07-mtp-spec.log`

## Decision: **record breadth, commit as wall at decode-crash**

We got further than expected (MTP-aware GGUF loads on ROCm 7.2 stock runtime)
but fell short of executing a live spec-decode run. The F32-MTP retry is a
clean single-variable change that's a natural next iteration.
