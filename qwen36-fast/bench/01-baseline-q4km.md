# Baseline decode benchmark — Qwen3.6-27B Q4_K_M

Run date: 2026-04-23 CEST (Thu, ~00:45)

## Model

- Repo: `unsloth/Qwen3.6-27B-GGUF`
- File: `Qwen3.6-27B-Q4_K_M.gguf`
- Arch (per llama.cpp): `qwen35 27B Q4_K - Medium`
- Params: 26.90 B
- File size: 16.817 GB / 15.662 GiB / 16,817,244,384 bytes
- HF revision: `82d411acf4a06cfb8d9b073a5211bf410bfc29bf`
- Blob content hash (HF cache filename): `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`
- Host path: `/home/everlier/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q4_K_M.gguf`

## Backend

- llama.cpp ROCm 7.2 build
- Image: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`
- llama.cpp build: `d6f303004 (8738)` — stock upstream with full qwen35 dense arch support
- GPU: Radeon 8060S Graphics, `gfx1151`, 96 GiB (98304 MiB) VRAM pool, 256 GB/s LPDDR5x
- Host: fedora (Strix Halo)
- Tool: `llama-bench` (not `llama-server`), run via `docker run` directly (bypasses Harbor
  compose command-formatting bug — see notes at bottom).

## Command

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  -v /home/everlier/.cache/huggingface:/root/.cache/huggingface:ro \
  kyuz0/amd-strix-halo-toolboxes:rocm-7.2 \
  llama-bench -m <GGUF_PATH> -ngl 99 -fa 1 -r 3 \
    -p <PP> -n 256 [-d <DEPTH>]
```

Per-invocation parameters: `-ngl 99` (full offload), `-fa 1` (flash attention on),
`-r 3` (3 repetitions per test). No `--no-mmap`, no `-dio`, no `-np`, no `--kv-unified`
— clean baseline.

Default knobs left as llama-bench defaults: `-b 2048 -ub 512 -ctk f16 -ctv f16 -t 16`.

## Results

| scenario                                        | prefill (pp) tps | decode (tg) tps (mean ± std) |
|-------------------------------------------------|-----------------:|-----------------------------:|
| short prompt — pp8  →  tg256 (depth 0)          |  38.62 ± 0.65    | **10.97 ± 0.01**             |
| 2 k context     — pp2048 → tg256 at depth 2048  | 308.40 ± 0.22    | **10.87 ± 0.01**             |
| 7 k context     — pp7000 → tg256 at depth 7000  | 266.77 ± 0.39    | **10.67 ± 0.01**             |

(3 repetitions, ±1 stddev across reps, no warmup override so first-token overhead is amortised.)

### Observations

- Decode falls only ~2.7 % from 0-depth → 7 k depth (10.97 → 10.67 tps).
  KV-cache scan cost is negligible inside 7 k at Q4_K_M with flash attention on.
- Prefill throughput peaks around 2 k (308 tps), drops to 267 tps at 7 k.
  Fine for near-term targets; this workload is decode-bound.
- Short-prompt "pp8" result (38.62 tps) is dominated by per-batch fixed overhead,
  not representative of sustained prefill.

## Peak VRAM

Sampled with `rocm-smi --showmeminfo vram` while the 7k test was running:

- Baseline (no model loaded): 1.75 GiB (system/display overhead)
- During 7 k-context bench: **18.79 GiB** peak
- Delta for model + KV: **~17.0 GiB** (15.66 GiB weights + ~1.3 GiB activations / KV for 7 k ctx with FA)

Headroom against 96 GiB pool: ~77 GiB free — plenty of room for a draft model
(+ Qwen3-0.6B Q4 adds ~0.4 GiB; +MTP drafter variant would be ~1 GiB on top of backbone).

## Bandwidth-ceiling analysis

Q4_K_M is ~4.83 bpw on this GGUF (15.66 GiB / 26.90 B params × 8 bits/byte ≈ 5.00 bpw
including overhead; the model-weight-only sweep per decode token is therefore ~15.66 GiB
of weight reads plus KV reads).

On 256 GB/s LPDDR5x shared memory:

- Weights-only ceiling: 256 GB/s ÷ 15.66 GiB ≈ **15.6 tps**
- Orchestrator-supplied assumed ceiling (4.5 bpw → 15 GB/token): ~17 tps

Measured decode: **10.87 tps (mid-ctx)** → **~65–72 % of bandwidth ceiling**.

That is *good* — similar-quality fraction to mature CUDA kernels on a well-tuned model.
It suggests the ROCm 7.2 HIP kernels on gfx1151 are not the bottleneck; remaining 30 %
is KV-cache reads, dispatch overhead, and fused-op gaps (hybrid Gated DeltaNet layers
likely less optimal than pure attention layers).

## Comparison to user-reported reference

| reference                                          | measured |
|----------------------------------------------------|---------:|
| user quoted ~15 tps at Q2 (backend unconfirmed)    | n/a      |
| user quoted ~5 tps at Q8                           | n/a      |
| **this bench: Q4_K_M on ROCm 7.2 via Harbor image**| **10.9 tps** |

The Harbor ROCm 7.2 stack on Q4_K_M already lands between the user's Q8 and Q2 data
points as expected by linear bandwidth scaling (Q4 is ~2× the size of Q2). No evidence
that Harbor is leaving perf on the table vs a hand-rolled llama.cpp build.

## Gap to 40 tps target

- Baseline decode (Q4_K_M, no drafting): **10.87 tps**
- Target: **40 tps**
- Required speedup: **3.7×**

A drafter with acceptance α and draft-cost c relative to target-step cost yields:

  speedup ≈ (1 + α·K) / (1 + K·c)

For MTP-head drafter where c ≈ 1/n_layers ≈ 1/64 ≈ 0.016 (a single block vs the full
backbone), and assuming K=4 draft tokens per step:

- α = 0.60 → (1 + 2.4) / (1 + 0.064) ≈ **3.19×** → ~34.7 tps
- α = 0.70 → (1 + 2.8) / (1.064) ≈ **3.57×** → ~38.8 tps
- α = 0.75 → (1 + 3.0) / (1.064) ≈ **3.76×** → ~40.9 tps ✅
- α = 0.80 → (1 + 3.2) / (1.064) ≈ **3.95×** → ~42.9 tps ✅

**Verdict**: a 1.7–2.2× "classical" drafter (α ≈ 0.5, K=2) is **not sufficient** —
it would yield only 18–24 tps. To hit 40 tps at Q4_K_M we need either:

1. An MTP-style drafter with α ≥ 0.75 at K=4, **OR**
2. Q4 → Q3/Q2 quant (ceiling rises to ~22–36 tps, then a 1.7–2.2× drafter is enough), **OR**
3. Both (safest path — Q3 + MTP ≈ 40 tps with α≈0.6).

The MTP head is confirmed present in the HF safetensors (see notes/03-drafter-strategy.md)
and consists of a **single full-attention transformer block** sharing the backbone's
embedding and lm_head. That matches path 1's assumed cost structure very well.

## Remaining high-impact work

Ranked by (tps impact) × (1 / effort):

1. **Patch `convert_hf_to_gguf.py` to emit a separate drafter GGUF** (`qwen35mtp` arch).
   Directly unlocks path 1. Effort: ~50–200 LoC C++/Py. Upstream-mergeable.
2. **Wire MTP drafter into `common/speculative.cpp`** as a new
   `SPECULATIVE_TYPE_MTP` variant (target+draft share embed+lm_head, draft reuses
   target's last-layer hidden state through `mtp.fc`). Effort: medium.
3. **Measure α** empirically on a calibration corpus once steps 1-2 compile.
   The 3.7× target is sensitive to α — this is the gating number.
4. **Try Q3_K_S** (another 30 % bandwidth headroom) as insurance in case α lands
   below 0.70. Cheap — just re-run this benchmark with a different quant.
5. Only if above falls short: the DFlash block-verify port (high effort, high risk,
   needs Gated DeltaNet kernel work for its 3:1 hybrid blocks).

## Harbor state / known issues

**Harbor is currently DOWN for llamacpp.** Root cause: Harbor's compose command
template is `${MODEL_SPECIFIER} ${EXTRA_ARGS} --port 8080 --host 0.0.0.0`, which
for the ROCm image (no entrypoint) produces `argv[0] = "-m"` and fails at
container-start. The original working config had `EXTRA_ARGS="llama-server --no-mmap -dio -ngl 99 -c 256000 -np 4 --kv-unified"` which produces `argv = ["-m","<path>","llama-server",...]` — same broken shape. The original service cannot have been running as-is; suspect it was running with an older image that DID have `llama-server` as entrypoint, or a stale cached container. Worth a follow-up to properly fix the template (moving `llama-server` into `MODEL_SPECIFIER` or adding an `entrypoint:` line in `compose.x.llamacpp.rocm.yml`).

Not blocking this task: `docker run` directly against the image works fine for
all benchmarks here.

Harbor `.env` restored at end of iteration:

- `HARBOR_LLAMACPP_GGUF="/root/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-UD-IQ3_XXS.gguf"`
- `HARBOR_LLAMACPP_EXTRA_ARGS="llama-server --no-mmap -dio -ngl 99 -c 256000 -np 4 --kv-unified"`

Harbor llamacpp **container is down** (was stopped to free VRAM for benches). Bringing
it back up with `./harbor.sh up llamacpp` will hit the same compose template bug that
blocked this iteration's attempt to run the 27B via Harbor — i.e. not a regression,
pre-existing issue. Saved the original env snapshot to `/tmp/harbor-orig-config.env`.

## Raw logs

- `bench/01-short-prompt.log` — pp8 + tg256
- `bench/02-medium-2k.log`    — pp2048 + tg256 @ d2048
- `bench/03-long-7k.log`      — pp7000 + tg256 @ d7000
