#!/usr/bin/env bash
# Pull upstream mainline, merge into the working branch, then CUDA build.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-master}"   # llama.cpp mainline branch
TARGET_BRANCH="${TARGET_BRANCH:-chore/jay1}"   # branch to pull into
CUDA_ARCH="${CUDA_ARCH:-120}"                  # RTX 5060 Ti = compute capability 12.0
BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc)}"

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--clean]

Pull origin/$UPSTREAM_BRANCH, merge it into $TARGET_BRANCH, then run a
CUDA (GGML_CUDA=ON) Release build for compute capability $CUDA_ARCH.

  --clean   remove $BUILD_DIR first (full rebuild)

Overridable env vars: UPSTREAM_BRANCH, TARGET_BRANCH, CUDA_ARCH, BUILD_DIR, JOBS
EOF
      exit 0
      ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$REPO_DIR"
echo "==> repo:   $REPO_DIR"
echo "==> merge:  origin/$UPSTREAM_BRANCH -> $TARGET_BRANCH"

current="$(git rev-parse --abbrev-ref HEAD)"
if [ "$current" != "$TARGET_BRANCH" ]; then
  echo "==> checkout $TARGET_BRANCH"
  git checkout "$TARGET_BRANCH"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "!! working tree has uncommitted changes; merge may fail" >&2
fi

git fetch origin
# fast-forward the local mainline ref too (non-fatal if it has diverged)
git fetch origin "$UPSTREAM_BRANCH:$UPSTREAM_BRANCH" 2>/dev/null || true

if ! git merge --no-edit "origin/$UPSTREAM_BRANCH"; then
  echo "!! merge failed (conflicts?). Resolve them, then re-run this script." >&2
  exit 1
fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

if [ "$CLEAN" = 1 ]; then
  echo "==> clean: rm -rf $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

echo "==> configure CUDA build (arch $CUDA_ARCH)"
cmake -B "$BUILD_DIR" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

echo "==> build (-j$JOBS)"
cmake --build "$BUILD_DIR" --config Release -j"$JOBS"

echo "==> done"
ls -lh "$BUILD_DIR/bin/llama-server" 2>/dev/null || true
