#!/usr/bin/env bash
# One-time setup after cloning: fetch both simulators, apply all patches, build them.
# Safe to re-run: every patch is applied only if it is not already present.
#
#   LLMSimulator/            SSD-offloading copy (GPU-only + expert offloading)  -> Q2, Q3, Q4
#   external/LLMSimulator/   original LLMSimulator (Duplex, Logic-PIM, Ramulator) -> Q1
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PATCHES="$ROOT/results-ca/simulator_patches"
JOBS=${JOBS:-$(nproc)}

# apply_once <repo dir> <patch file> [-p0]
apply_once() {
  local dir=$1 patch=$2 strip=${3:--p1}
  if git -C "$dir" apply --check $strip --whitespace=nowarn "$patch" 2>/dev/null; then
    git -C "$dir" apply $strip --whitespace=nowarn "$patch"
    echo "   applied   $(basename "$patch") -> ${dir#$ROOT/}"
  elif git -C "$dir" apply --check -R $strip --whitespace=nowarn "$patch" 2>/dev/null; then
    echo "   present   $(basename "$patch") -> ${dir#$ROOT/}"
  else
    echo "ERROR: $(basename "$patch") does not apply cleanly in ${dir#$ROOT/}" >&2
    exit 1
  fi
}

build() {
  echo "▶ building ${1#$ROOT/}"
  mkdir -p "$1/build"
  (cd "$1/build" && cmake .. > /dev/null && make -j"$JOBS" run > /dev/null)
  echo "   built     ${1#$ROOT/}/build/run"
}

echo "▶ fetching submodules"
git -C "$ROOT" submodule update --init --recursive

echo "▶ patching SSD-offloading simulator (LLMSimulator/)"
SSD="$ROOT/LLMSimulator"
apply_once "$SSD/src/dram/ramulator2" "$SSD/patch/ramulator2_pim.patch"
# our patch sits on top of the authors' patch, so if ours is present both are
if git -C "$SSD" apply --check -R --whitespace=nowarn "$PATCHES/ssd_offloading_llmsim.patch" 2>/dev/null; then
  echo "   present   ssd_offloading.patch + ssd_offloading_llmsim.patch -> LLMSimulator"
else
  apply_once "$SSD" "$ROOT/patch/ssd_offloading.patch" -p0    # SSD-offloading authors
  apply_once "$SSD" "$PATCHES/ssd_offloading_llmsim.patch"     # this project
fi

echo "▶ patching original simulator (external/LLMSimulator/)"
ORIG="$ROOT/external/LLMSimulator"
apply_once "$ORIG/src/dram/ramulator2" "$ORIG/patch/ramulator2_pim.patch"
apply_once "$ORIG" "$PATCHES/original_llmsim.patch"            # this project

build "$SSD"
build "$ORIG"

echo "▶ done. Python packages: pip install -r results-ca/requirements.txt"
