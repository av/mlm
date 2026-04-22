# llama.cpp Qwen3.6 + MTP support survey

Thu 2026-04-23 ~00:45 CEST. Based on fresh shallow clone of `ggml-org/llama.cpp` at
commit `0d0764d` (HEAD of `master`, 2026-04-22 17:52 UTC). Source lives at
`/home/everlier/code/mlm/qwen36-fast/deps/llama.cpp/` (gitignored).

## 1. Upstream Qwen3.6 architecture support: PRESENT

Qwen3.5/3.6 is a first-class arch upstream. Both dense (`qwen35`) and MoE (`qwen35moe`)
are wired end-to-end through converter, gguf-py, llama-arch, and model builder.

Evidence:

- **Arch enum**: `src/llama-arch.h:46-47`
  ```
  LLM_ARCH_QWEN35,
  LLM_ARCH_QWEN35MOE,
  ```
- **Arch name registration**: `src/llama-arch.cpp:42-43`
  ```
  { LLM_ARCH_QWEN35,    "qwen35"    },
  { LLM_ARCH_QWEN35MOE, "qwen35moe" },
  ```
- **Constants enum**: `gguf-py/gguf/constants.py:406-407` (`QWEN35`, `QWEN35MOE`).
- **Converter classes**: `convert_hf_to_gguf.py:5422-5429`
  ```python
  @ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
  class Qwen3_5TextModel(_LinearAttentionVReorderBase):
      model_arch = gguf.MODEL_ARCH.QWEN35

  @ModelBase.register("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
  class Qwen3_5MoeTextModel(_LinearAttentionVReorderBase):
      model_arch = gguf.MODEL_ARCH.QWEN35MOE
  ```
  MRO: `Qwen3_5TextModel -> _LinearAttentionVReorderBase -> Qwen3NextModel -> Qwen2MoeModel`.
- **Vision-model branch** (Qwen3_5 is a multimodal arch in HF): `convert_hf_to_gguf.py:4840`
  registers `Qwen3_5ForConditionalGeneration` for vision tensor extraction into an mmproj file.
  Vision converter explicitly skips `mtp.*` at `convert_hf_to_gguf.py:4891-4893`.
- **Model builders**: `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`
  (hybrid Gated DeltaNet + full attention, via `llm_build_delta_net_base`).
- **Model type aliases**: `src/llama-model.h:125` `LLM_TYPE_35B_A3B // Qwen3.5`
  plus `122B_A10B` and `397B_A17B` — these are the A3B / A10B / A17B MoE sizes.
  A 27B dense type alias will be inferred from n_layer / n_embd at load.
- **Vocab pretokenizer**: `LLAMA_VOCAB_PRE_TYPE_QWEN35 = 46` at `src/llama-vocab.h:57`,
  string "qwen35" at `src/llama-vocab.cpp:2029`.

**Implication**: The Unsloth Qwen3.6-27B-GGUF Q4_K_M we are pulling will load
on upstream `llama-server` out-of-the-box, with the caveat below.

## 2. MTP / NextN support: TENSOR NAMES EXIST, INFERENCE NOT WIRED

Upstream has a *partial* MTP scaffold. Six tensor slots and one hparam kv are defined
in GGUF, and several converter paths write them — but no model builder actually runs
the MTP forward pass at inference time.

- **Tensor slots**: `src/llama-arch.h:551-556`
  ```
  LLM_TENSOR_NEXTN_EH_PROJ,
  LLM_TENSOR_NEXTN_EMBED_TOKENS,
  LLM_TENSOR_NEXTN_ENORM,
  LLM_TENSOR_NEXTN_HNORM,
  LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,
  LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,
  ```
- **HParam**: `llama-hparams.h:93` `uint32_t nextn_predict_layers = 0;`
  read via `LLM_KV_NEXTN_PREDICT_LAYERS` (`%s.nextn_predict_layers`).
- **Inference stance — explicit skip**: `src/llama-arch.cpp:759-760`
  ```
  // NextN/MTP tensors are currently ignored (reserved for future MTP support)
  // These tensors only exist in the last layer(s) and are treated as output tensors
  ```
  and consumers: `deepseek2.cpp:51`, `glm4.cpp:31`, `glm4-moe.cpp:31`,
  `exaone-moe.cpp:22`, `bailingmoe2.cpp:21` all do
  `n_transformer_layers = n_layer - hparams.nextn_predict_layers;`
  i.e. they *drop* the MTP layer at graph-build time.
- **No spec-decode path consumes MTP**: `common/speculative.cpp` supports
  `COMMON_SPECULATIVE_TYPE_{NONE,DRAFT,EAGLE3,NGRAM_*}`. No `MTP` / `NEXTN` variant.

**Converter-side NextN/MTP handling today — ONLY for GLM4 family and similar:**

- `convert_hf_to_gguf.py:10032-10044` (GLM4) — uses
  `num_nextn_predict_layers` hparam and appends NextN layers as extra blocks;
  calls `self.gguf_writer.add_nextn_predict_layers(...)`.
- `convert_hf_to_gguf.py:10091-10093` (GLM4_MOE) — same pattern.
- `convert_hf_to_gguf.py:10169-10184` (GlmMoeDsaModel) — same pattern.
- `convert_hf_to_gguf.py:2117-2140` (tensor_mapping.py in gguf-py) — NextN tensor
  name regexes: they expect HF names `model.layers.{bid}.eh_proj`,
  `...embed_tokens`, `...enorm`, `...hnorm`, `...shared_head.head`, `...shared_head.norm`.
  This is the **DeepSeek / GLM** MTP schema where MTP is treated as one extra
  transformer block appended after `num_hidden_layers`.
- **Qwen3.6 MTP layout is DIFFERENT**: in the HF safetensors we already inspected,
  the names are under `mtp.*` at top level (`mtp.fc.weight`,
  `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`,
  `mtp.layers.0.{self_attn,mlp,input_layernorm,...}`, `mtp.norm.weight`).
  This does **not** match the NextN regex. **The existing NEXTN_* tensor slots
  will not capture it without a converter patch.**

**Qwen3.6 MTP explicit drop site in converter — THE KEY PATCH POINT:**

`convert_hf_to_gguf.py:4780-4782`:
```python
def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
    if name.startswith("mtp"):
        return  # ignore MTP layers for now
```
This method belongs to `Qwen3NextModel`, which is the ancestor of
`_LinearAttentionVReorderBase` → `Qwen3_5TextModel`. The early-return is the
surgical place to intercept MTP tensors and route them into a separate file.

## 3. Draft-model speculative decoding: FULLY SUPPORTED

CLI flags (identical in `llama-cli` and `llama-server`):

- `-md` / `--model-draft FNAME` — draft model path. `common/arg.cpp:3506-3511`,
  env `LLAMA_ARG_MODEL_DRAFT`.
- `--draft` / `--draft-n` / `--draft-max N` — tokens to draft per step (default 16).
- `--draft-min` / `--draft-n-min N` — minimum draft tokens (default 0).
- `--draft-p-min P` — min probability for greedy drafting (default 0.75).
- `--draft-p-split P` — probability split threshold.

Docs: `tools/cli/README.md:183-189`, `tools/server/README.md:234-240`.

Spec-decode variants enumerated at `common/common.h:158-168` and
`common/speculative.cpp:21-40`:

- `draft` — classic two-model spec decode (what we need)
- `eagle3` — EAGLE-3 head
- `ngram_simple` / `ngram_map_k` / `ngram_map_k4v` / `ngram_mod` / `ngram_cache`
  — n-gram / prompt-lookup variants (no second model required)

Vocab compatibility guardrails: `common/speculative.cpp:18-19`
```
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5
```
Target and draft vocabs must match to within 128 tokens and share prefix.
Since MTP head shares Qwen3.6's embed+lm_head, a **Qwen3.6 MTP-drafter GGUF
will have identical vocab to the 27B backbone** — vocab check passes.

Test reference: `tools/server/tests/unit/test_speculative.py` (drafts
tinyllamas/stories15M-q4_0 against Llama — usable as a template for integration tests).

## 4. Where to patch for MTP preservation

Goal: run `python convert_hf_to_gguf.py Qwen3.6-27B/` and get **two GGUFs**:
(a) backbone with 64 transformer blocks (today's behaviour, minus the silent `mtp`
drop), (b) drafter GGUF containing the single MTP block plus shared embed +
lm_head references.

**Option A — minimal (ship MTP tensors inside the main GGUF as NextN slots).**

1. `convert_hf_to_gguf.py:4780-4782` — stop blanket-dropping `mtp.*`. Route
   the 15 `mtp.*` tensors to the six `LLM_TENSOR_NEXTN_*` slots (block id
   `num_hidden_layers = 64` so they appear as one extra block).
2. `convert_hf_to_gguf.py` — add
   `self.gguf_writer.add_nextn_predict_layers(1)` in `Qwen3_5TextModel.set_gguf_parameters`
   (mirroring GLM4 at line 10043).
3. `gguf-py/gguf/tensor_mapping.py:2117-2140` — extend the NextN tensor regexes
   with Qwen3.6 `mtp.*` names (`mtp.fc`, `mtp.pre_fc_norm_embedding`,
   `mtp.pre_fc_norm_hidden`, `mtp.layers.0.*`, `mtp.norm`). Mapping is not
   1:1 with DeepSeek's NextN names; may need a new tensor enum for `mtp.fc`
   (token + hidden concat projection) — likely `NEXTN_EH_PROJ` semantically
   matches `mtp.fc`.
4. `src/llama-arch.cpp:8226` — add `LLM_ARCH_QWEN35` to the list of archs that
   register NEXTN_* tensors during model load.
5. Pros: single GGUF file, matches existing NextN convention; Cons: inference side
   still ignores those tensors, so we get nothing from `-md` until step 6.
6. Wire a new spec-decode path or reuse `COMMON_SPECULATIVE_TYPE_EAGLE3` machinery
   for an MTP drafter using the backbone hidden state + MTP block + shared lm_head.
   Large C++ change across `common/speculative.cpp` and one new model builder.

**Option B — pragmatic for 5h budget (emit separate drafter GGUF).**

The MTP block is a standalone transformer layer + pre/post norms + a fused
embed+hidden projection. Functionally it is a 1-layer Qwen3 model that consumes
`[embed(next_token); hidden_state]` and outputs a hidden state, then applies
shared lm_head.

1. **Converter patch only.** In
   `Qwen3NextModel.modify_tensors` (`convert_hf_to_gguf.py:4780`), intercept
   `mtp.*` tensors and stash them in a separate writer. At `prepare_tensors`
   emit a second GGUF with arch `qwen35` (or a bespoke `qwen35mtp`), 1 layer,
   block-copying the 27B's `token_embd.weight` and `output.weight` to avoid
   forcing llama.cpp to resolve cross-file shared tensors.
2. Because the MTP block uses full attention (not Gated DeltaNet per
   notes/03), the drafter GGUF can literally be declared as `qwen3` arch,
   1 hidden layer, same `hidden_size/num_attention_heads` as 27B. Upstream
   llama.cpp loads it with zero new code.
3. Catch: the first projection `mtp.fc.weight` fuses embed + hidden → hidden.
   That does not map to any existing Qwen3 tensor. Either (a) absorb it into
   the first attn input norm + linear (approximate — loses acceptance quality),
   or (b) publish a new arch `qwen35mtp` with ~20 lines of C++ in a new model
   builder under `src/models/`.
4. Option B.b is the cleanest. ~1 day of work. Aligns with "Path A" from
   iteration 3 notes.

**Recommendation**: Option B.b — patch converter to emit separate drafter GGUF
with new `qwen35mtp` arch, plus a tiny model builder in `src/models/`.
Avoids touching the NextN scaffold and the speculative.cpp machinery;
drafter GGUF is consumed as a normal `-md` model.

Concrete files to touch for the converter side:
- `convert_hf_to_gguf.py:4780` — replace blanket `return` with MTP redirect.
- `convert_hf_to_gguf.py` ~5422 — add drafter-writer state, `prepare_tensors` hook.
- `gguf-py/gguf/constants.py:406` — add `QWEN35MTP = auto()`.
- `gguf-py/gguf/constants.py:892` — add `{MODEL_ARCH.QWEN35MTP: "qwen35mtp"}`.
- `gguf-py/gguf/tensor_mapping.py` — add `mtp.*` → canonical names.
- `src/llama-arch.{h,cpp}` — register `LLM_ARCH_QWEN35MTP`.
- `src/llama-model.cpp` ~2805, 9046 — handle the new arch in param-setup + build.
- `src/models/qwen35mtp.cpp` (new) — forward pass for 1 MTP block.
- `src/models/models.h` — declare `llm_build_qwen35mtp`.

For MVP without C++ changes: pretend MTP is a 1-layer `qwen35` dense model
and map `mtp.fc` into `blk.0.attn_qkv.weight` by reshape + zero-pad. Expected
acceptance loss: significant (maybe still 30-40% acceptance vs 60-75% ideal).
Worth trying as a sanity check before the full C++ path.

## 5. Download status

- Restart: PID `2239766` (background, not the echoed-pid wrapper),
  started 2026-04-23 00:33 CEST.
- Progress at 00:45 CEST: `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF`
  at 15 GB of ~16.8 GB blob. ETA ~2 min.
- Log: `/tmp/qwen36-downloads/27b.log`.

## 6. Unsloth fork status

- `unslothai/llama.cpp` (fork of `ggml-org/llama.cpp`) last pushed 2026-04-22 17:51 UTC.
- Scanned 100 most recent commits: no Qwen3.5/3.6 or MTP-specific downstream patches.
  Latest Qwen-related commits (`mtmd: qwen3 audio support`, `TP: fix Qwen 3 Next data split`)
  are upstream merges. Unsloth's Qwen3.6-GGUF release was produced by stock
  upstream converter — which confirms the silent MTP drop (it matches what we
  already observed in gguf-dump in iteration 2).
- No alternate MTP converter to borrow.

## 7. Risks / unknowns

- `mtp.fc.weight` shape: safetensors said `fc` combines embed-next + hidden-prev.
  Need to read the exact dims (target dim × (2 × hidden_size) or hidden × embed+hidden)
  before writing the tensor mapping. Plan: dump the safetensors header via
  `safetensors.safe_open` header-only after download finishes — trivial.
- Qwen3.6 MTP block's internal attn may still rely on Qwen3_5's linear_attn / SSM
  path even though notes/03 claimed "full attention". Worth re-verifying against
  config's `mtp_*` keys and the layer's `self_attn.*` tensor names. If it is
  linear attention, drafter GGUF needs full `qwen35` arch support not a stripped
  dense variant.
- `SPEC_VOCAB_CHECK_START_TOKEN_ID = 5` — drafter vocab must agree with target
  on first 5 tokens. Shared lm_head guarantees this but worth spot-check.
- NVFP4 tensor reordering code in `_LinearAttentionVReorderBase._transform_nvfp4_weight`
  runs on Qwen3_5 tensors — if MTP block has NVFP4 weights, we need to run the same
  reorder on them.
- The upstream-present `qwen35` arch may be assuming 35B-A3B-style hparams
  (MoE). Loading a 27B dense GGUF as `qwen35` (no MoE) should hit the dense
  code path in `src/models/qwen35.cpp` — to confirm after download finishes
  by actually loading with harbor llama-server.

## 8. Bottom line

- Qwen3.6 27B conversion & inference: **works upstream today**, no patch needed
  to run the vanilla model.
- MTP drafter: **requires converter patch + ~50 LoC of C++** to land a minimum
  working end-to-end path. Largely mechanical. No upstream blocker.
- Speculative decoding plumbing (`-md`): **works today**. Needs a valid drafter
  GGUF with matching vocab. MTP drafter will satisfy this trivially.
