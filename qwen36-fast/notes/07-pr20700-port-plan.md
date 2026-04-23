# PR #20700 Deep-Dive + Qwen3.6 Port Plan

Date: 2026-04-22 (iteration ~12, pure research — no code / no build)
Source: https://github.com/ggml-org/llama.cpp/pull/20700
Diff snapshot: `/tmp/pr20700.diff` (1550 lines, +890/-113)
Head branch: `itigges22/llama.cpp:feat/qwen35-dense-mtp`
Last updated upstream: 2026-04-21T15:27:53Z

## 1. PR overview

- **Title**: "feat: MTP support for dense Qwen 3.5 with FastMTP vocabulary trimming"
- **Author**: itigges22 (Johnathon Isaac Tigges) — first-time llama.cpp contributor,
  motivated by the ATLAS project.
- **Status**: OPEN, `mergeable: CONFLICTING`, `reviewDecision: REVIEW_REQUIRED`.
- **Size**: 20 files, +890 / -113. No maintainer reviews posted, only community
  comments. The author explicitly states: *"This PR is still very much a WIP —
  I do not expect it to be merged any time soon."*
- **Base**: `master`. No dependency PRs required; it stacks directly on top of
  current master (which already has the #19493 recurrent-checkpoint work).
- **Reported on**: Qwen3.5-9B Q4_K_M + F16 MTP, RTX 5060 Ti 16GB, K3s.
  Claimed 82% acceptance temp=0 (two-phase: 92%), **27B NOT claimed to work**.
- **Known blockers from comments**:
  - `petter-b` reproduces only 63% acceptance (not 82%) with Q4_K_M MTP — variance
    traced to MTP head being quantized vs left F16.
  - **`AImindPalace` hit word-salad output on Qwen3.5-27B Q6_K (M3 Ultra / Metal)**
    with and without `--spec-type mtp`. "Validation has only been done on 9B.
    This looks like the fix needs 27B-specific handling (different MTP head shape
    or different vocabulary trim)." — critical for us, Qwen3.6-27B is the same
    shape.
  - `quivent` posted three downstream forks
    (`quivent/qwen-mtp-llamacpp`, `recurrent-rollback`, `qwen-mtp-optimizations`)
    with "adaptive chained MTP prediction" achieving ~2× TPS — these are the
    next-generation ideas to cherry-pick once PR #20700 lands.
- **Merge signal**: weak. No CI test GGUF, no Metal verification, 27B
  corruption reported, author himself says "needs refactor". Expect another
  1-3 months before merge.

## 2. File-by-file inventory

Every diff hunk, with purpose and line estimate. Paths are relative to llama.cpp root.

| File | ±LoC | Purpose |
| --- | --- | --- |
| `Dockerfile.atlas` | +17 / -0 | ATLAS-project build image. NOT needed for us. |
| `common/arg.cpp` | +5 / -2 | Adds `--spec-type` CLI values `mtp`, `ngram-*`, parses to enum. |
| `common/common.h` | +1 / -0 | New enum entry `COMMON_SPECULATIVE_TYPE_MTP`. |
| `common/sampling.cpp` | +7 / -0 | Two `fprintf(stderr, "[MTP-VERIFY] …")` debug prints in `common_sampler_sample_and_accept_n`. Pure instrumentation. |
| `common/speculative.cpp` | +104 / -3 | New `common_speculative_state_mtp` class (argmax over reduced MTP logits + cooldown on rejection), name lookup table, factory wire-up, and **`common_speculative_is_compat` relaxation** — if `llama_model_n_mtp_layers(model) > 0`, tolerate `seq_rm` failure because rollback will use checkpoint/restore. This replaces the patch we hand-rolled in iter-11. |
| `convert_hf_to_gguf.py` | +49 / -0 | `Qwen3_5TextModel.__init__` bumps `block_count += mtp_num_hidden_layers`. `set_gguf_parameters` writes `nextn.predict_layers`. `modify_tensors` remaps: `mtp.layers.0.*` → `model.layers.{n_hidden}.*` (appended block); `mtp.fc` → `nextn.eh_proj`; `mtp.pre_fc_norm_embedding` → `nextn.enorm`; `mtp.pre_fc_norm_hidden` → `nextn.hnorm`; `mtp.norm` → `nextn.shared_head.norm`. **This alone is the HF→GGUF tensor-emit patch we needed in iter-3.** |
| `gguf-py/gguf/constants.py` | +8 / -1 | Adds six `NEXTN_*` tensors to `MODEL_ARCH.QWEN35`'s tensor list (already existed for QWEN35MOE/DeepSeek). |
| `include/llama.h` | +8 / -0 | Three new C-API symbols: `llama_model_n_mtp_layers`, `llama_get_mtp_logits`, `llama_get_mtp_n_vocab`. |
| `src/llama-arch.cpp` | +14 / -8 | Adds the six `LLM_TENSOR_NEXTN_*` entries to the QWEN35 tensor-names set; changes six existing `LLM_TENSOR_NEXTN_*` layer classes from `LLM_TENSOR_LAYER_OUTPUT` to `LLM_TENSOR_LAYER_REPEATING` so they live inside the per-layer structure. Removes the "NextN/MTP tensors are currently ignored" comment. **No new `LLM_ARCH_*` or `LLM_KV_*` enums** — reuses existing `LLM_ARCH_QWEN35` + `LLM_KV_NEXTN_PREDICT_LAYERS`. |
| `src/llama-batch.cpp` | +3 / -2 | Minor batch-alloc tweak to allow unequal_seq_lens in certain MTP paths. |
| `src/llama-context.cpp` | +35 / -0 | `llama_context::get_mtp_logits()` reads `res->t_logits_mtp` from the graph schedule, copies to CPU; `decode()` populates `mtp_logits_buf` when `t_logits_mtp` is present; public wrappers `llama_get_mtp_logits` / `llama_get_mtp_n_vocab`. |
| `src/llama-context.h` | +7 / -0 | New `mtp_logits_buf` vector + `mtp_logits_valid` flag + `n_vocab_mtp` (FastMTP reduced vocab size). |
| `src/llama-graph.h` | +4 / -0 | Adds `ggml_tensor * t_embd_mtp; ggml_tensor * t_logits_mtp;` to `llm_graph_result`. |
| `src/llama-memory-recurrent.cpp` | +178 / -22 | **Biggest single file**. Fuzzy `seq_rm` — searches for best checkpoint cell ≤ p0-1 and rolls `tail_id` to it. Partial removal just rewinds `cells[i].pos = p0-1`. Fixed `seq_cp` to copy state into an empty cell (prevents shared mutable state). New `copy_cell(i_src, i_dst)` method using `ggml_backend_tensor_copy` on 1-D views of `r_l[il]`/`s_l[il]` — **this fixes a real upstream bug (the existing `ggml_view_1d` was passing byte count instead of element count)**. New `get_cell_count(seq_id)`. `find_slot` now keeps up to 8 historical cells per seq as checkpoint ring buffer (unless `used > 0.9 * size`), allocates a fresh cell and copies state from the previous tail. |
| `src/llama-memory-recurrent.h` | +4 / -0 | Declarations for `copy_cell` + `get_cell_count`. |
| `src/llama-model.cpp` | +93 / -33 | In QWEN35 branch of `load_hparams`: read `LLM_KV_NEXTN_PREDICT_LAYERS`, compute `n_main_layers = n_layer - nextn_predict_layers`, mark MTP layers as **non-recurrent** in `recurrent_layer_arr`, use `n_main_layers` (not `n_layer`) for type detection. In `load_tensors`: MTP layers get `nextn.eh_proj/enorm/hnorm/shared_head_norm/shared_head_head/embed_tokens` **plus standard attention + dense FFN tensors** from the same `layer.wq/wk/wv/wo/ffn_*` slots. In `create_memory`: unchanged rs_size (1 per seq). Adds `llama_model_n_mtp_layers` implementation. |
| `src/models/models.h` | +7 / -0 | Adds `mtp_inp_hidden` member + `build_mtp_head(...)` method declaration on `llm_build_qwen35`. |
| `src/models/qwen35.cpp` | +182 / -24 | Main-loop now stops at `n_transformer_layers = n_layer - nextn_predict_layers`. On the **last main layer** it splits into filtered (main logits) and unfiltered (saved as `mtp_inp_hidden` for MTP). New `build_mtp_head()` does: `greedy_tokens = argmax(main_logits); emb = embed(greedy); h_norm = RMS(hidden); e_norm = RMS(emb); combined = eh_proj(concat(e_norm,h_norm)); attn(combined with gated-RoPE); FFN; final_norm; lm_head (trimmed to 32K for FastMTP via `ggml_view_2d`)`. Writes `res->t_logits_mtp`. |
| `tools/server/server-context.cpp` | +164 / -18 | Per-slot state `mtp_draft_token/i_batch/pending/cooldown`. Auto-enables `COMMON_SPECULATIVE_TYPE_MTP` when `n_mtp_layers > 0` and compat check passes. Main loop: after sampling, if `mtp_pending` — compare sampled to last draft; on accept, decode draft in a **1-token second pass** (two-phase decode) then sample bonus token; on reject, just continue with clean state (no `seq_rm`). Falls back to `seq_rm` with a `SLT_WRN` rather than ABORT on hybrid models. |

## 3. Specific questions answered

**Q: Where are `mtp.*` tensors written in `convert_hf_to_gguf.py`?**
`convert_hf_to_gguf.py:5046-5095` (new code inside class `Qwen3_5TextModel`
registered as `Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM`).

**Q: New `LLM_TENSOR_MTP_*` enums added?**
**No** — the six `LLM_TENSOR_NEXTN_*` constants already existed (DeepSeek/GLM
scaffold). PR only adds them to QWEN35's allowed tensor set and retargets
their `llm_tensor_info` from `LAYER_OUTPUT` to `LAYER_REPEATING`.
See `src/llama-arch.cpp:1051-1063` and `:2753-2772`.

**Q: New `LLM_ARCH_*` or `LLM_KV_*`?**
**No.** Reuses existing `LLM_ARCH_QWEN35` and `LLM_KV_NEXTN_PREDICT_LAYERS`.

**Q: New `.cpp` under `src/models/` or inline?**
**Inline** — extends `src/models/qwen35.cpp` with a new `build_mtp_head()`
member (145 new lines tacked onto the existing `llm_build_qwen35`).

**Q: `SPECULATIVE_TYPE_MTP` in `common/speculative.h`?**
Yes, in `common/common.h:172-178` (enum `common_speculative_type`).
Drafter class is `common_speculative_state_mtp` in `common/speculative.cpp:466-550`.
CLI flag: `--spec-type mtp` via `common/arg.cpp:3474-3498`.

**Q: FastMTP vocab trim — separate tensor or runtime?**
**Runtime.** The vocabulary is trimmed via a `ggml_view_2d` over the first
32,768 rows of `lm_head` (or `nextn.shared_head_head`). No new GGUF tensor,
no vocab-mask. The 32K value is **hardcoded** in `src/models/qwen35.cpp` inside
`build_mtp_head`. Token IDs 0..32767 in the MTP output map directly to full
vocab IDs (relies on tokenizer ordering by frequency). Author flagged
"consider making configurable" in the TODO list.

**Q: Where is `--two-phase-decode` implemented?**
*Not a CLI flag* — it is always on when MTP is enabled in the server.
Logic is entirely in `tools/server/server-context.cpp`, roughly lines 2833-2990:
decode `[sampled]` alone (1-token batch), compare sampled vs the pending
`mtp_draft_token`; if match → decode draft in a second 1-token batch and
sample a bonus from its logits; if mismatch → skip the draft decode
entirely (state stays clean, no `seq_rm` needed). The plain `llama-cli`
MTP path uses the classic speculative framework (`common_speculative_draft` +
`common_batch_add(draft)` + partial `seq_rm` on reject).

## 4. Qwen3.5 vs Qwen3.6 MTP layout

Fetched both configs and tensor indices from HF. Result: **IDENTICAL** for the
27B variants.

| Field | Qwen3.5-27B | Qwen3.6-27B |
| --- | --- | --- |
| `architectures` | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| `model_type` | `qwen3_5` | `qwen3_5` |
| `num_hidden_layers` | 64 | 64 |
| `hidden_size` | 5120 | 5120 |
| `intermediate_size` | 17408 | 17408 |
| `head_dim` | 256 | 256 |
| `num_attention_heads` | 24 | 24 |
| `num_key_value_heads` | 4 | 4 |
| `full_attention_interval` | 4 | 4 |
| `vocab_size` | 248320 | 248320 |
| `mtp_num_hidden_layers` | 1 | 1 |
| `mtp_use_dedicated_embeddings` | false | false |
| `layer_types` pattern | 3 linear + 1 full ×16 | 3 linear + 1 full ×16 |
| MTP tensor count | **15** | **15** |
| MTP tensor names | all 15 match exactly (`mtp.fc.weight`, `mtp.pre_fc_norm_{embedding,hidden}.weight`, `mtp.layers.0.{input_layernorm,self_attn.{q,k,v,o}_proj,self_attn.{q,k}_norm,post_attention_layernorm,mlp.{gate,up,down}_proj}.weight`, `mtp.norm.weight`) | (same) |

Text-config diff is cosmetic only: Qwen3.6 adds `bos_token_id` (already
deducible), `output_gate_type: "swish"` (hardcoded in PR's gated-attn path),
`partial_rotary_factor: 0.25` (already present inside `rope_scaling`),
`tie_word_embeddings: false` (default), `pad_token_id: null`. None of these
change tensor layout or graph topology.

**Conclusion: PR #20700's converter + C++ code is 100% compatible with
Qwen3.6-27B at the file-format and graph-topology level. No Qwen3.6 delta
in tensor layout.**

However — see risks below — **the PR has not been validated on 27B, and
one 27B user reports word-salad**. Likely root causes:

1. FastMTP 32K trim may be wrong for 248K vocab when tokens beyond 32K are
   common for the distribution the 27B produces. (9B testing only hit code
   prompts.) `ggml_view_2d` grabs the first 32768 **rows** of `lm_head`,
   which are the first 32768 tokens in vocab-ID order, which may or may not
   be the most frequent 32768 — the PR assumes a frequency-ordered tokenizer.
2. Attention path in the MTP head uses the main model's joint QG projection
   (`n_embd_head_k * n_head * 2`) — the MTP tensors in HF are **plain
   `q_proj`** (`n_embd_head * n_head`, i.e. non-gated). The PR loader forces
   `wq` to the gated 2× size but Qwen3.5/3.6's `mtp.layers.0.self_attn.q_proj`
   is the non-gated size. This is an almost-certain load-tensor shape
   mismatch OR a silent half-read, which fits the word-salad symptom.
   Needs verification — see risks.

## 5. Concrete porting plan for Qwen3.6 on top of #20700

### Step 1 — files to touch (on top of a fresh `master` that already has #19493)

Exactly zero **new** files. Three files need changes beyond cherry-picking
the PR:

1. `src/models/qwen35.cpp` — fix the MTP attention head shape (non-gated q_proj).
2. `src/models/qwen35.cpp` — make FastMTP vocab trim size a hparam or disable it for safety.
3. `convert_hf_to_gguf.py` — optional: auto-flag `--keep-mtp-f16` so MTP weights survive quantization at full precision (matches the author's original 82% run).

### Step 2 — minimal change set

**A. Q/K projection shape for MTP layer only** (~20 LoC in `llama-model.cpp`):

```cpp
// In QWEN35 load_tensors, is_mtp_layer branch:
// Current PR code:
layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head * 2 }, 0);
// Qwen3.5/3.6 MTP uses non-gated q_proj (verify from safetensors dims)
layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head * 1 }, 0);
```

Qwen3.6's `mtp.layers.0.self_attn.q_proj.weight` has shape `[6144, 5120]`
(= `24 * 256`, non-gated). The main-model `q_proj` for full-attn layers has
`[12288, 5120]` (= `24 * 256 * 2`, gated). Need to verify from our already-downloaded
MTP shards — file `/tmp/qwen36-mtp-shards/model-00015-of-00015.safetensors`
has these tensors, can `safetensors.safe_open` to confirm shapes **before** touching code.

Corresponding change in `qwen35.cpp::build_mtp_head`: skip the gating
split in `build_layer_attn` for the MTP layer — easiest by branching on
`il >= n_transformer_layers` inside `build_layer_attn`.

**B. FastMTP vocab size as config** (~10 LoC):

```cpp
// Read from hparams; default 0 (= no trim).
const int64_t mtp_vocab_size = hparams.mtp_vocab_trim
    ? std::min(lm_head->ne[1], (int64_t)hparams.mtp_vocab_trim)
    : lm_head->ne[1];
```

For our first run: **disable the trim** (pass 0 / use full 248K vocab).
The 8× matmul cost is acceptable for verifying correctness; turn the
trim back on only after output is clean.

**C. MTP-F16 preservation in converter** (optional, ~10 LoC):

```python
# In Qwen3_5TextModel.modify_tensors, after remapping an mtp.* tensor:
if os.environ.get("LLAMA_KEEP_MTP_F16"):
    data_torch = data_torch.to(torch.float16)
# yield as usual
```

Together with `llama-quantize --pure` + manual mix table; or emit a separate
drafter GGUF (our original Path B from iter-3).

### Step 3 — test plan

1. Apply the diff, rebuild.
2. Run the PR's original converter on HF Qwen3.6-27B MTP shards, emit
   `Qwen3.6-27B-mtp-f16.gguf`.
3. Quantize main backbone to Q4_K_M (or reuse Unsloth's), preserving MTP
   tensors at F16 via `llama-quantize --keep-split-weights mtp.*` or
   a custom mix file.
4. Smoke test with `llama-cli --spec-type mtp` on a 300-token prompt:
   verify no word-salad.
5. `llama-bench` with and without `--spec-type mtp` to measure speedup.
6. If word-salad persists: binary-chop by disabling FastMTP trim, then
   disabling two-phase decode, then pinning `draft-max=1`.

## 6. Estimated Qwen3.6-specific LoC delta on top of #20700

- Strictly to get it working: **0 lines** (the diff already handles both
  Qwen3.5-27B and Qwen3.6-27B identically at GGUF level).
- To fix the 27B word-salad (`AImindPalace` report + likely our own bug):
  **~30-50 lines** in `llama-model.cpp` (Q shape) + `qwen35.cpp` (attn branch).
- To make FastMTP configurable / safer for unknown vocab distributions:
  **~20 lines** in `qwen35.cpp` + `llama-hparams.h` + converter hparam.
- To preserve MTP-F16 through quantization: **~20 lines** in converter.

**Total realistic delta: ~70-90 LoC** across three files; ~2-4 hours of
implementation + iteration time, assuming the 27B corruption is indeed the
non-gated-vs-gated q_proj mismatch.

## 7. Risks & unknowns

1. **27B corruption root cause** is unconfirmed. Our hypothesis (non-gated
   q_proj shape mismatch) fits the symptom but is not verified. Needs a
   `safetensors.safe_open` sanity check on our MTP shards before building.
2. **FastMTP 32K assumption** — Qwen3.6's tokenizer is not guaranteed to be
   frequency-ordered for the first 32K IDs. Disabling the trim on first run
   is the safe call.
3. **CONFLICTING status** — PR is behind master. Cherry-picking will hit
   merge conflicts in `src/llama-model.cpp`, `common/speculative.cpp`,
   `tools/server/server-context.cpp` (all are hot-churn files). Budget an
   extra 30-60 min for conflict resolution.
4. **DeltaNet recurrent-state rollback** — PR's `find_slot` keeps 8 checkpoint
   cells per seq. Our `rs_size` multiplication upstream (iter-9) already
   accounts for this. But on a 27B with `n_seq_max=1` we only get `rs_size=1`
   by default — **we must pass `-np 8` or bump `rs_size` manually** or the
   checkpoint ring never holds more than one entry and rollback degenerates
   to "state is stuck at the last decoded token, future drafts corrupt".
5. **PR author admits "bugs, needs refactor"** — expect rough edges especially
   in error paths, thread safety, and `seq_cp` behavior when multiple slots
   are active.
6. **No CI / no maintainer review yet** — we cannot rely on upstream fixes
   landing fast. If we adopt this PR we own the maintenance of the fork
   until the PR merges (likely months).
7. **`can_seq_rm` bypass already matches what we tried in iter-11** — the PR's
   `common_speculative_is_compat` relaxation does the same thing our manual
   patch does, so iter-11's build should be re-based on top of the PR rather
   than stacked under it.

## 8. Recommended immediate follow-up (sequenced)

Once iter-11 build finishes:

1. `safetensors.safe_open` on `/tmp/qwen36-mtp-shards/*.safetensors` — read
   shapes of `mtp.layers.0.self_attn.q_proj.weight`,
   `.k_proj.weight`, `.v_proj.weight`, `.o_proj.weight`.
   Confirm the q_proj shape hypothesis. (5 min.)
2. If confirmed: fetch PR #20700 as a git branch (don't merge), run
   `git diff master...origin/feat/qwen35-dense-mtp -- src/ common/ convert_hf_to_gguf.py`
   and replay onto our tree. (~30 min with conflict resolution.)
3. Patch the q_proj shape for MTP layer only + disable FastMTP trim. (~30 min.)
4. Run the converter on the 4.3 GiB MTP shards already downloaded. (~5 min.)
5. Rebuild and smoke test 27B Q2_K_XL backbone + F16 MTP head with
   `--spec-type mtp`. Measure acceptance + TPS. (~30 min.)
