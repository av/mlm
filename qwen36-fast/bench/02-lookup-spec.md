# Lookup / n-gram speculative-decoding benchmark — Qwen3.6-27B Q4_K_M

Run date: 2026-04-23 CEST (Thu, ~01:20)

## TL;DR

**llama.cpp's upstream prompt-lookup speculative decoding is BROKEN on
Qwen3.6** — the Qwen3.6 family uses M-RoPE (`rope type = 40`,
`rope.dimension_sections = [11, 11, 10, 0]`) which enforces a strict
"X < Y" position-monotonicity constraint across the batch. The lookup
loop in `examples/lookup/lookup.cpp` adds multiple draft tokens with
consecutive `n_past + i` positions and then rewinds KV via
`llama_memory_seq_rm` on rejection; this produces a start position
equal to (not strictly greater than) the last cached position and
every multi-token verification call fails with:

```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 1824
 - the tokens for sequence 0 in the input batch have a starting position of Y = 1823
 for M-RoPE, it is required that the position satisfies: X < Y
decode: failed to initialize batch
llama_decode: failed to decode, ret = -1
```

The generation loop ignores the failure, samples a token from stale
logits (garbage), records it in `n_predict`, then re-fails on the next
iteration. **The tps/acceptance numbers the tool prints are therefore
NOT real speedups; the output is corrupted.** We verified this by
reading the output text (gibberish with dropped / duplicated spans)
and by the ~25 % failure rate on every configuration where drafting
is enabled.

With drafting **disabled** (`--draft-max 0`) the same binary produces
**zero** decode failures and matches the clean baseline (10.83 tps vs
10.87 tps from `llama-bench`), which localises the bug to the
draft-batch + KV-rewind path.

### The numbers (all at d≈2000 ctx, n_predict=256, greedy)

| config                                | reported tps | decode failures | accept α | output quality     |
|---------------------------------------|-------------:|----------------:|---------:|--------------------|
| **baseline** `llama-bench` d=2048 ×3  |   **10.87**  |            0    |   n/a    | clean (reference)  |
| `llama-lookup` drafting off (`dm=0`)  |     10.83    |            0    |   n/a    | **clean** (matches)|
| `llama-lookup` `dm=1 dmin=1`          |     13.80    |           58    |  51.4 %  | **corrupted**      |
| `llama-lookup` `dm=4 dmin=1` (rep 1)  |     26.10    |           64    |  65.1 %  | **corrupted**      |
| `llama-lookup` `dm=4 dmin=1` (rep 2)  |     26.20    |           ~64   |  65.1 %  | **corrupted**      |
| `llama-lookup` `dm=8 dmin=1`          |     50.61    |          196    |  17.7 %  | **corrupted**      |

"decode failures" counts lines matching `decode: failed to decode,
ret = -1` in the log. Every one corresponds to a verification batch
where M-RoPE rejected the positions.

The non-monotonic α vs speedup relationship (α drops 65 % → 18 % as we
go from `dm=4` to `dm=8`, yet reported throughput climbs from 26 to
50 tps) is the smoking-gun: a well-behaved drafter sees higher α
reduce verify-cost, but with the M-RoPE bug, higher `dm` just lets the
loop skip more real decodes per wallclock second while printing
garbage.

### Verdict

- **Does the quick-win help?** No. `llama-cli`/`llama-lookup` with
  n-gram drafting cannot be used on Qwen3.6 in current upstream
  llama.cpp (commit `d6f303004`, build 8738) without a patch to
  either (a) the M-RoPE check in `llama-batch.cpp` /
  `llama-memory-*.cpp`, or (b) the lookup loop's KV-rewind semantics.
  The same bug will block the eventual MTP drafter path unless fixed.
- **MTP is still required** — in fact the MTP path will hit the SAME
  bug, so the MTP patch must include a fix for the M-RoPE X<Y check
  on draft-verify batches (likely: set draft-batch positions to
  `n_past, n_past+1, ...` but make sure `llama_memory_seq_rm` is
  called BEFORE the batch is built, and that the first draft token's
  position is strictly > the last cached position).

**Effective lookup speedup on a clean implementation:** unknown from
these data, because every speedup observation is contaminated by
corrupted output. The one reliable datum is that when drafting is
disabled the binary hits the same ~10.9 tps as `llama-bench` — which
rules out any hidden overhead in the lookup binary itself.

## Environment

- Host: fedora (Strix Halo, gfx1151, 96 GiB VRAM pool, 256 GB/s LPDDR5x)
- Image: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`
- llama.cpp build `d6f303004 (8738)` (same as iteration 5)
- Model: `unsloth/Qwen3.6-27B-GGUF` Q4_K_M, file at
  `/home/everlier/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q4_K_M.gguf`
  (symlink → blob `5ed60d0af465…`).
- Prompt: `/tmp/qwen36-lookup-bench/prompt_code.txt` — a senior-engineer
  code review request that asks the model to (1) write a structured
  review of a ~100-line Python module and (2) repeat the module
  verbatim afterwards. This is the *best case* for prompt-lookup
  drafting (high textual redundancy between prompt and output), so the
  acceptance α here is an **upper bound** on what to expect from
  lookup in general workloads (creative / conversational α is typically
  0.2–0.4). Prompt is 1892 tokens (within the 2 k target).

## Binaries used

- `llama-bench` — clean baseline, no spec decode (iteration 5 tooling)
- `llama-lookup` — the prompt-lookup speculative decoder from
  `examples/lookup/lookup.cpp` in upstream llama.cpp. Accepts the same
  `-m / -ngl / -fa / -n / -c` flags as `llama-cli`, plus
  `--draft-max N`, `--draft-min N`, `--lookup-cache-static FNAME`,
  `--lookup-cache-dynamic FNAME`. Prints:
  - `encoded N tokens in X s, speed Y t/s` (prefill)
  - `decoded N tokens in X s, speed Y t/s` (decode, drafting-aware)
  - `n_drafted / n_accept / accept = %`
  - full `common_perf_print` block (load / prompt eval / eval)

We did NOT build or load any external corpus; the dynamic cache was
built from the prompt itself + generated tokens (which is the default
behaviour of `llama-lookup` even without `--lookup-cache-dynamic`).
Building a static cache would not have helped since (a) the drafter
itself is what's broken, not the cache source, and (b) any static
cache gain sits on top of the decode loop that fails.

## Commands

### Baseline re-check (llama-bench)

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  -v /home/everlier/.cache/huggingface:/root/.cache/huggingface:ro \
  kyuz0/amd-strix-halo-toolboxes:rocm-7.2 \
  llama-bench -m <GGUF> -ngl 99 -fa 1 -r 3 -p 2048 -n 256 -d 2048
# → pp 306.02 ± 0.88 t/s, tg 10.87 ± 0.01 t/s  (matches iteration 5)
```

### Lookup (dynamic ngram cache, draft-max=4)

```bash
docker run --rm -i --device /dev/kfd --device /dev/dri \
  -v /home/everlier/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /tmp/qwen36-lookup-bench:/work \
  kyuz0/amd-strix-halo-toolboxes:rocm-7.2 \
  llama-lookup -m <GGUF> -ngl 99 -fa 1 -f /work/prompt_code.txt \
    -n 256 -c 4096 --temp 0 --seed 1 \
    --draft-max 4 --draft-min 1 < /dev/null
```

Same shape for `--draft-max {0, 1, 8}`.

## Raw logs

All in `/tmp/qwen36-lookup-bench/`:

- `00-baseline-recheck.log` — llama-bench sanity, confirms 10.87 tps
- `10-lookup-dyn-rep1.log` — lookup dm=4 rep 1 (26.10 tps, corrupted)
- `10-lookup-dyn-rep2.log` — lookup dm=4 rep 2 (26.20 tps, corrupted)
- `11-lookup-dm8.log`      — lookup dm=8 (50.61 tps, corrupted)
- `12-lookup-dm0.log`      — lookup drafting off (10.83 tps, CLEAN)
- `13-lookup-dm1.log`      — lookup dm=1 (13.80 tps, corrupted)

## Why this matters for the MTP path

The M-RoPE X<Y constraint is enforced at
`llama-memory-*.cpp::find_slot` / `llama-batch.cpp::init` in upstream.
It applies to **any** speculative-decode variant (NGRAM, EAGLE3, and a
future MTP) because they all batch multiple verify positions into one
`llama_decode`. Before we spend the converter + C++ builder effort on
the MTP path we should either:

1. Patch the M-RoPE position-consistency check to allow `X == Y` when
   the KV has been explicitly rewound to a draft-start position (the
   "rejected draft" case), OR
2. Change the draft-verify batch construction so the first draft
   token's position is always strictly > the last cached position
   (i.e. run `seq_rm` only to `n_past - 1` and re-submit the boundary
   token as position `n_past`), OR
3. Disable M-RoPE entirely for the target at decode time and rely on
   standard RoPE (at a quality cost we have not measured).

Option 2 is the correct fix and should be a small change. This
pre-work is **required** before MTP will give any real speedup.

## Updated gap-closing plan

Baseline: 10.87 tps, target 40 tps, required multiplier 3.68×.

- Lookup-only on a fixed M-RoPE: **unknown, but bounded above by the
  drafter cost structure**. For n-gram drafting where the draft is
  free (c ≈ 0), speedup = 1 + α·K. With K=4 and α=0.65 (our best clean
  number would-be if the bug were fixed) → 3.6×. That is right at the
  target. However: (a) the M-RoPE fix is prerequisite, (b) α=0.65
  above was on a self-referential "repeat this module verbatim"
  prompt — α on typical chat/reasoning workloads is 0.2–0.4 →
  1.8–2.6× speedup → 19–28 tps, below target.
- Lookup (fixed) + MTP together: MTP gives α ≈ 0.70 on general text
  at low c; add lookup on top for self-referential spans. Worth
  exploring once both work.

So **MTP is still the primary path**, and we now know a prerequisite:
the M-RoPE draft-verify fix.

## Safetensors shard download status

Background download of the two safetensors shards containing `mtp.*`
tensors from `Qwen/Qwen3.6-27B`:

- Shard list determined from `/tmp/qwen36-27b-meta/model.safetensors.index.json`:
  - `model-00013-of-00015.safetensors` — 7 of 15 mtp tensors (fc,
    layer-0 attn q/k/v_proj, layer-0 mlp gate/up/down)
  - `model-00015-of-00015.safetensors` — 8 of 15 mtp tensors (layer-0
    norms, o_proj, q/k_norm, mtp.norm, pre_fc_norm_{embedding,hidden})
- Command: `hf download Qwen/Qwen3.6-27B model-00013-of-00015.safetensors model-00015-of-00015.safetensors --local-dir /tmp/qwen36-mtp-shards`
- PID: `2419748` (completed in ~46 s — fast link)
- Output dir: `/tmp/qwen36-mtp-shards/`
- Shard sizes: 3.8 GiB + 486 MiB = **4.29 GiB total (7.9 %) vs 54 GiB
  full checkpoint** — 12.6× download saving.
- All 15 `mtp.*` tensors verified present and bf16-typed with expected
  shapes (see log block at bottom).

Verified shapes — these confirm the architecture described in
notes/03-drafter-strategy.md:

```
mtp.fc.weight                              [5120, 10240]  (concats hidden + embedding)
mtp.pre_fc_norm_embedding.weight           [5120]
mtp.pre_fc_norm_hidden.weight              [5120]
mtp.layers.0.input_layernorm.weight        [5120]
mtp.layers.0.post_attention_layernorm.weight [5120]
mtp.layers.0.self_attn.q_proj.weight       [12288, 5120]  (24 heads × 256 dim → GQA 6:1 to Q)
mtp.layers.0.self_attn.k_proj.weight       [1024, 5120]   (4 kv heads × 256)
mtp.layers.0.self_attn.v_proj.weight       [1024, 5120]
mtp.layers.0.self_attn.o_proj.weight       [5120, 6144]   (24 × 256 concat → 5120)
mtp.layers.0.self_attn.q_norm.weight       [256]
mtp.layers.0.self_attn.k_norm.weight       [256]
mtp.layers.0.mlp.gate_proj.weight          [17408, 5120]
mtp.layers.0.mlp.up_proj.weight            [17408, 5120]
mtp.layers.0.mlp.down_proj.weight          [5120, 17408]
mtp.norm.weight                            [5120]
```

Head counts / dims match the backbone exactly
(`qwen35.attention.head_count = 24`, `head_count_kv = 4`,
`key_length = value_length = 256`, `embedding_length = 5120`,
`feed_forward_length = 17408`) — MTP is a single full-attention
transformer block that drops straight into the architecture.

## Remaining high-impact work

Ranked by (tps impact) × (1 / effort):

1. **Fix the M-RoPE draft-verify position constraint** in llama.cpp.
   Prerequisite for any spec-decode path on Qwen3.6. Effort: small
   (change seq_rm / batch-start-position logic in `examples/lookup/`
   and in `common/speculative.cpp`). Without this, ALL of paths 2/3/4
   fail.
2. **Patch `convert_hf_to_gguf.py` to emit `qwen35mtp` drafter GGUF**
   (unchanged from iteration 5 plan). All shards already local.
3. **Wire `SPECULATIVE_TYPE_MTP` in `common/speculative.cpp`** with the
   M-RoPE fix from #1 applied.
4. **Re-benchmark with clean lookup + MTP drafter** once #1 is
   applied. Measure α on realistic prompts, not the synthetic
   self-referential one used here (that number is an upper bound).
5. Skip: trying more `--lookup-cache-static` corpora or `llama-server`
   lookup mode — all would hit the same M-RoPE bug.
