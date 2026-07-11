#!/usr/bin/env bash
# Publish each live-indicator package as its own GitHub repository.
# Run from the repo root on a machine where `gh auth login` has been done.
# Usage: bash lppls_paper_vs_repo/scripts/publish_indicators.sh [--public]
set -euo pipefail

VISIBILITY="--private"
[ "${1:-}" = "--public" ] && VISIBILITY="--public"

BASE="$(cd "$(dirname "$0")/.." && pwd)/live_indicators"
for pkg in "$BASE"/*/; do
    name=$(basename "$pkg")
    echo "=== $name"
    tmp=$(mktemp -d)
    cp -r "$pkg"/. "$tmp"/
    git -C "$tmp" init -q -b main
    git -C "$tmp" add -A
    git -C "$tmp" commit -q -m "Initial release: $name"
    gh repo create "$name" $VISIBILITY --source "$tmp" --push
    rm -rf "$tmp"
done
echo "All indicator repos published."
