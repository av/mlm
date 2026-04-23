#!/usr/bin/env bash
# run-best.sh - reproduce the iter-13 30.05 tps lookup-spec result on Qwen3.6-27B.
#
# Usage:
#   ./bench/run-best.sh            # full run, n_predict=256, ~40-60s
#   ./bench/run-best.sh --short    # sanity-check, n_predict=64, ~15-25s
#   ./bench/run-best.sh --help     # this message
#
# Exit codes:
#   0 = measured tps >= floor (the "still reproducing" floor)
#   1 = measured tps < floor (regression)
#   2 = GGUF missing
#   3 = patched binary missing
#   4 = docker / device prerequisites missing
#
# Reference numbers (from iter-13, bench/06-patched-lookup.md):
#   tps = 30.05, alpha = 0.653, n_predict = 256, prompt = 1766 tokens.

# Make script self-contained: don't assume caller's env exists.
# PATH must include docker; default to Fedora/Debian-ish PATH superset.
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
# HOME is needed to locate HF model cache. Fall back to invoking user's home.
if [[ -z "${HOME:-}" ]]; then
    export HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
fi

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="full"
if [[ "${1:-}" == "--short" ]]; then
    MODE="short"
fi

N_PREDICT=256
DRAFT_MAX=4
TPS_FLOOR_FULL=25
# Short-mode floor is intentionally loose: with n_predict=64 on a 1766-token prompt,
# the dynamic n-gram cache hasn't warmed, so alpha stays near 0.15-0.30 and tps is
# in the 15-20 range. This mode verifies the pipeline works, not the 30 tps result.
TPS_FLOOR_SHORT=14
if [[ "$MODE" == "short" ]]; then
    N_PREDICT=64
fi

GGUF_REL="models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-UD-Q2_K_XL.gguf"
GGUF_HOST="${HOME}/.cache/huggingface/hub/${GGUF_REL}"
GGUF_CONTAINER="/models/${GGUF_REL}"

BIN_HOST="${ROOT_DIR}/deps/llama.cpp/build-rocm"
BIN_CONTAINER_PATH="/bld/bin/llama-lookup"

PROMPT_HOST="${ROOT_DIR}/prompts/prompt_code.txt"
PROMPT_CONTAINER="/prompts/prompt_code.txt"

IMAGE="kyuz0/amd-strix-halo-toolboxes:rocm-7.2"

echo "=== Qwen3.6-27B lookup-spec reproduction bench ==="
echo "mode=${MODE}  n_predict=${N_PREDICT}  draft_max=${DRAFT_MAX}"
echo

# Preflight 1: GGUF
if [[ ! -r "$GGUF_HOST" ]]; then
    echo "ERROR: UD-Q2_K_XL GGUF not found at:" >&2
    echo "  $GGUF_HOST" >&2
    echo >&2
    echo "Download it with (uses ~11 GiB of network + disk):" >&2
    echo "  huggingface-cli download unsloth/Qwen3.6-27B-GGUF \\" >&2
    echo "      Qwen3.6-27B-UD-Q2_K_XL.gguf \\" >&2
    echo "      --local-dir-use-symlinks True" >&2
    exit 2
fi

# Preflight 2: patched binary
if [[ ! -x "${BIN_HOST}/bin/llama-lookup" ]]; then
    echo "ERROR: patched llama-lookup binary not found at:" >&2
    echo "  ${BIN_HOST}/bin/llama-lookup" >&2
    echo >&2
    echo "Build it with the iter-11 patch applied:" >&2
    echo "  cd ${ROOT_DIR}/deps/llama.cpp" >&2
    echo "  git apply ${ROOT_DIR}/patches/llamacpp-qwen36-spec-decode.patch" >&2
    echo "  # then build inside the ROCm 7.2 toolbox image:" >&2
    echo "  docker run --rm -v \$PWD:/src -w /src ${IMAGE} \\" >&2
    echo "      bash -lc 'cmake -S . -B build-rocm -G Ninja \\" >&2
    echo "          -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \\" >&2
    echo "          -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \\" >&2
    echo "          && ninja -C build-rocm llama-lookup'" >&2
    exit 3
fi

# Preflight 3: prompt
if [[ ! -r "$PROMPT_HOST" ]]; then
    echo "ERROR: prompt file missing: $PROMPT_HOST" >&2
    exit 3
fi

# Preflight 4: docker + devices
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH" >&2
    exit 4
fi
for dev in /dev/kfd /dev/dri; do
    if [[ ! -e "$dev" ]]; then
        echo "ERROR: $dev missing (ROCm device not present)" >&2
        exit 4
    fi
done

# Check image locally; pull quietly if missing (may take a while first run).
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Pulling $IMAGE (first run only)..."
    docker pull "$IMAGE"
fi

LOG_DIR="${ROOT_DIR}/bench"
LOG_FILE="${LOG_DIR}/run-best-${MODE}-$(date +%Y%m%d-%H%M%S).log"

echo "Invoking docker run (log -> $LOG_FILE)..."
echo

# Actual invocation - same flags that produced iter-13's 30.05 tps.
# --temp left at default (0.8) to match iter-13 regime.
set +e
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video \
    -v "$HOME/.cache/huggingface/hub:/models:ro" \
    -v "${BIN_HOST}:/bld:ro" \
    -v "${ROOT_DIR}/prompts:/prompts:ro" \
    -e LD_LIBRARY_PATH=/bld/bin:/opt/rocm-7.2.0/lib \
    "$IMAGE" \
    "$BIN_CONTAINER_PATH" \
        -m "$GGUF_CONTAINER" \
        -ngl 99 -fa on \
        -f "$PROMPT_CONTAINER" \
        -n "$N_PREDICT" --draft-max "$DRAFT_MAX" 2>&1 | tee "$LOG_FILE"
RC=${PIPESTATUS[0]}
set -e

echo
echo "=== Parse results ==="

# Extract decode tps: line looks like:
# decoded  259 tokens in    8.618 seconds, speed:   30.053 t/s
TPS_LINE=$(grep -E "^decoded .* speed: .* t/s" "$LOG_FILE" | tail -1 || true)
ACCEPT_LINE=$(grep -E "^accept\s+=\s+[0-9.]+%" "$LOG_FILE" | tail -1 || true)

if [[ -z "$TPS_LINE" ]]; then
    echo "FAIL: no 'decoded ... t/s' line in log. Run did not complete." >&2
    exit 1
fi

TPS=$(echo "$TPS_LINE" | grep -oE "[0-9]+\.[0-9]+ t/s" | tail -1 | awk '{print $1}')
ALPHA=$(echo "$ACCEPT_LINE" | grep -oE "[0-9]+\.[0-9]+%" | tail -1 || echo "n/a")

echo "Measured: tps=${TPS}  alpha=${ALPHA}"
echo "Reference (iter-13, bench/06-patched-lookup.md):"
echo "           tps=30.05   alpha=65.29%"
echo

# Diff vs expected.
if [[ "$MODE" == "short" ]]; then
    FLOOR="$TPS_FLOOR_SHORT"
else
    FLOOR="$TPS_FLOOR_FULL"
fi

# bash arithmetic doesn't do floats; use awk.
PASS=$(awk -v tps="$TPS" -v floor="$FLOOR" 'BEGIN { print (tps+0 >= floor+0) ? 1 : 0 }')

if [[ "$PASS" == "1" ]]; then
    DELTA=$(awk -v tps="$TPS" 'BEGIN { printf "%.2f", tps - 30.05 }')
    echo "PASS: tps=${TPS} >= ${FLOOR} (delta vs iter-13: ${DELTA})"
    exit 0
else
    echo "FAIL: tps=${TPS} < ${FLOOR} (regression vs iter-13's 30.05)" >&2
    echo "Check ${LOG_FILE} for decode failures, OOM, or model-load errors." >&2
    exit 1
fi
