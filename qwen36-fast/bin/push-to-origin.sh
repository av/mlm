#!/usr/bin/env bash
# push-to-origin.sh — guarded push of main to origin for the qwen36-fast work.
#
# Shows the recent commit log, refuses to push if the working tree is dirty,
# and asks for explicit Y/n confirmation before touching the remote.
#
# Usage:
#   bin/push-to-origin.sh              # interactive push
#   bin/push-to-origin.sh --dry-run    # show what would happen, no push
#   bin/push-to-origin.sh --help
#
# Exit codes:
#   0  — success (push completed, or dry-run finished cleanly)
#   1  — refused (dirty tree, user said no, unexpected branch, etc.)
#   2  — unexpected error (git command failed)

set -euo pipefail

DRY_RUN=0
case "${1:-}" in
    --help|-h)
        sed -n '2,16p' "$0" | sed 's/^# //; s/^#//'
        exit 0
        ;;
    --dry-run)
        DRY_RUN=1
        ;;
    "")
        ;;
    *)
        echo "unknown arg: $1 (try --help)" >&2
        exit 1
        ;;
esac

# Resolve repo root from script location so it works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- preflight: correct repo? ---
if [[ ! -d "$REPO_ROOT/.git" ]] && ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: $REPO_ROOT is not a git repo" >&2
    exit 2
fi

# --- preflight: on main? ---
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
    echo "ERROR: current branch is '$BRANCH', expected 'main'. Refusing." >&2
    exit 1
fi

# --- preflight: clean tree? ---
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree is not clean. Refusing to push." >&2
    echo "---" >&2
    git status --short >&2
    exit 1
fi

# --- show the commits that would be pushed ---
echo "=== Branch: $BRANCH ==="
AHEAD="$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "?")"
echo "=== Commits ahead of origin/main: $AHEAD ==="
echo
echo "=== git log --oneline -30 ==="
git log --oneline -30
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== DRY-RUN: would run 'git push origin main' ==="
    echo "(no remote contact made)"
    exit 0
fi

# --- confirm ---
read -rp "Push $AHEAD commits from $BRANCH to origin/main? [Y/n] " reply
reply="${reply:-Y}"
case "$reply" in
    [Yy]|[Yy][Ee][Ss])
        ;;
    *)
        echo "Aborted by user."
        exit 1
        ;;
esac

# --- push ---
echo "=== Pushing ==="
git push origin main
echo "=== Done ==="
exit 0
