#!/usr/bin/env bash
# file-upstream-pr.sh — copy-paste helper for filing the can_seq_rm patch
# upstream to ggml-org/llama.cpp.
#
# This does NOT run gh or git. It prints:
#   1. The filing instructions from patches/upstream-pr-draft/README.md.
#   2. The exact `gh pr create` command with the body piped from
#      PR-DESCRIPTION.md.
#
# Copy + paste the shown commands once you have a llama.cpp fork checked out.
#
# Usage:
#   bin/file-upstream-pr.sh
#   bin/file-upstream-pr.sh --help

set -euo pipefail

case "${1:-}" in
    --help|-h)
        sed -n '2,14p' "$0" | sed 's/^# //; s/^#//'
        exit 0
        ;;
    "")
        ;;
    *)
        echo "unknown arg: $1 (try --help)" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRAFT_DIR="$QWEN_ROOT/patches/upstream-pr-draft"

if [[ ! -d "$DRAFT_DIR" ]]; then
    echo "ERROR: $DRAFT_DIR does not exist." >&2
    echo "Expected the upstream-pr-draft produced in iter-22." >&2
    exit 1
fi

PATCH_FILE="$DRAFT_DIR/0001-common-relax-can_seq_rm-probe-to-try-checkpoint-roun.patch"
DESC_FILE="$DRAFT_DIR/PR-DESCRIPTION.md"
README_FILE="$DRAFT_DIR/README.md"

for f in "$PATCH_FILE" "$DESC_FILE" "$README_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required file: $f" >&2
        exit 1
    fi
done

echo "=============================================================="
echo "  Filing instructions (from patches/upstream-pr-draft/README)"
echo "=============================================================="
echo
cat "$README_FILE"
echo
echo "=============================================================="
echo "  Exact command to run (in YOUR llama.cpp fork checkout)"
echo "=============================================================="
echo
echo "# 1) Edit From: and Signed-off-by: in the patch, then:"
echo "cd <your-llama.cpp-fork>"
echo "git checkout -b relax-can-seq-rm-checkpoint-probe"
echo "git am $PATCH_FILE"
echo "git push -u origin relax-can-seq-rm-checkpoint-probe"
echo
echo "# 2) File the PR (body piped from PR-DESCRIPTION.md):"
echo
echo "gh pr create \\"
echo "  --repo ggml-org/llama.cpp \\"
echo "  --base master \\"
echo "  --head \"\$(git config user.login || echo YOUR_USER):relax-can-seq-rm-checkpoint-probe\" \\"
echo "  --title 'common: relax can_seq_rm probe to try checkpoint round-trip' \\"
echo "  --body-file $DESC_FILE"
echo
echo "=============================================================="
echo "  NOT EXECUTING — this is a copy-paste helper by design."
echo "  Review the README above, edit the patch authorship, then"
echo "  run the commands yourself in your llama.cpp fork."
echo "=============================================================="
