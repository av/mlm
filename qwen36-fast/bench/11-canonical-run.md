# Canonical run-best.sh full-mode measurement

- **Date/time**: 2026-04-23 04:30 CEST (morning-after of the timeboxed sprint)
- **Host**: fedora, Strix Halo gfx1151, ROCm 7.2 (docker image `kyuz0/amd-strix-halo-toolboxes:rocm-7.2`)
- **Script**: `bench/run-best.sh` (iter-17 canonical reproduction script), full mode (n_predict=256)
- **Log**: `bench/11-canonical-run.log` (53 kB, full stdout+stderr), plus per-run log `bench/run-best-full-20260423-043015.log`

## Command executed

```
cd /home/everlier/code/mlm/qwen36-fast && ./bench/run-best.sh 2>&1 | tee bench/11-canonical-run.log
```

Underlying docker invocation (same flags that produced iter-13's 30.05 tps):

```
docker run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video \
    -v "$HOME/.cache/huggingface/hub:/models:ro" \
    -v "$ROOT/deps/llama.cpp/build-rocm:/bld:ro" \
    -v "$ROOT/prompts:/prompts:ro" \
    -e LD_LIBRARY_PATH=/bld/bin:/opt/rocm-7.2.0/lib \
    kyuz0/amd-strix-halo-toolboxes:rocm-7.2 \
    /bld/bin/llama-lookup \
        -m /models/.../Qwen3.6-27B-UD-Q2_K_XL.gguf \
        -ngl 99 -fa on \
        -f /prompts/prompt_code.txt \
        -n 256 --draft-max 4
```

## Result

| metric        | iter-13 reference | canonical run | delta   |
|---------------|-------------------|---------------|---------|
| decode tps    | 30.05             | **30.21**     | +0.16   |
| alpha         | 65.29%            | **66.94%**    | +1.65pp |
| n_drafted     | 242               | 248           | +6      |
| n_accept      | 158               | 166           | +8      |
| encode tps    | (not captured)    | 356.32        | —       |
| n_predict     | 256               | 261           | +5      |
| decode seconds| 8.618             | 8.640         | +0.02   |

**Verdict: PASS** (floor was 25 tps).

The 0.5% tps uptick and 1.65pp alpha uptick are within normal run-to-run noise from the dynamic n-gram cache warmup plus thermal variance. Script exit code was 0.

## Output sample (first ~220 chars of generation, after filtering MTP-SEQRM debug noise)

```
DEFAULT_NGRAM_MIN = 1
DEFAULT_NGRAM_MAX = 4
DEFAULT_CACHE_PATH = os.path.expanduser("~/.cache/llama/ngram.bin")


class NgramCache:
    """A simple in-memory n-gram cache.

    The cache maps a tuple of token ids (the n-gram key) to a Counter of
    candidate next tokens.
```

The model went on to produce a full `NgramCache` class definition, a `merge()` helper, then entered the structured code review with `<think>` reasoning tags — identical behavioural regime as iter-13. Output is fully coherent; no token loops, no gibberish.

## Warnings / noise observed

- **~50 `decode: failed to decode, ret = -1` lines** interspersed with **~150 `[MTP-FINDSLOT]` / `[MTP-SEQRM]` debug prints** — expected and matches iter-13. These come from the M-RoPE `X < Y` rejection path; PR #19493's checkpoint+restore retries silently handle them. They do not affect final tps or output correctness, they are just verbose debug output from the PR #20700 patched binary (iter-14 build).
- Initial `-fit` memory reducer kicked in: n_ctx trimmed from 262144 → 128768 because projected model+context+compute (27965 MiB) exceeded 20501 MiB free at startup (other processes held ~78 GB VRAM). This is automatic and does not affect decode throughput; the 1766-token prompt fits comfortably inside 128k ctx.
- Peak actual device memory used: 19300 MiB (10608 model + 8197 context + 495 compute). No OOM, no swap.

## Reproducibility note for Ivan

**This is the number to expect when running `./bench/run-best.sh` tomorrow morning**: ~30 tps decode, ~65-67% alpha on the canonical 1766-token code-review prompt. Expect ±1 tps run-to-run variance from dynamic n-gram cache warmup and thermal. The floor in the script is set to 25 tps; anything below that indicates a regression (thermal throttling, competing VRAM consumer, or a patched-binary swap).

Host conditions during this run: no `llama-server` serving (Harbor had the toolbox image but no active container holding VRAM), docker idle, laptop at rest. The script's own `-fit` preflight trimmed ctx to fit whatever was free at invocation time, so a cold run after a fresh boot would see slightly more VRAM headroom (the script doesn't require any; ctx reduction is silent).

## Interpretation

The iter-13 breakthrough number is **fully reproducible** on the exact binary + GGUF + prompt triple that the script freezes. This validates:

1. The iter-11 `can_seq_rm` relaxation patch path.
2. The iter-14 PR #20700 merge (even though MTP is dormant in this code path, the linked `libllama-common` changes did not regress lookup-spec).
3. The iter-17 canonical prompt + flags.

The 40 tps target remains out of reach via lookup on this hardware — consistent with the iter-16 saturation finding. The EAGLE-3 future path (see `notes/09-eagle3-future-path.md`) is the recommended follow-up if Ivan wants to close the remaining 10 tps gap.
