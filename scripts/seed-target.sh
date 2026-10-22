#!/bin/bash
# A new worktree's target directory as a clone of the host's warm build, so the worktree's first
# build compiles only the workspace's crates, not the ~87 packages they depend on.
#
#   scripts/seed-target.sh        clone the seed into $CARGO_TARGET_DIR, which must not exist yet
#   scripts/seed-target.sh save   make $CARGO_TARGET_DIR's build the seed: run it in the merge
#                                 worktree once its check passed, with no build running there
#
# The seed (~/.cache/reims-vgpu-seed; SEED moves it) is a saved target directory with Cargo's
# profile and target directories alone, no incremental cache, and no fingerprint of a workspace
# crate. Cargo takes a dependency's build as fresh from its fingerprint, and builds a unit with
# none again, so a seeded worktree compiles its own crates and reuses the rest. A workspace
# crate's fingerprint left in the seed would let a worktree whose files predate it run another
# tree's build of that crate (docs/lessons.md), so `save` removes every one.
#
# The clone is an APFS clone on macOS (cp -c), which writes nothing until a file changes, and
# cp --reflink=auto on Linux: a reflink where the filesystem has them, a copy where it has not.
# The seed and the target directories sit on one volume (~/.cache), or cp copies every byte.
set -euo pipefail
seed=${SEED:-$HOME/.cache/reims-vgpu-seed}
target=${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR to the target directory of this worktree}
root=$(cd "$(dirname "$0")/.." && pwd)

# clone SRC DST: DST, which does not exist, as a clone of the directory SRC.
clone() {
  if [[ $(uname) == Darwin ]]; then
    cp -c -p -R "$1" "$2"
  else
    cp -a --reflink=auto "$1" "$2"
  fi
}

if [[ ${1:-} != save ]]; then
  [[ ! -e $target ]] || { echo "seed-target: $target exists: seed only a new target directory" >&2; exit 1; }
  if [[ ! -d $seed ]]; then
    echo "seed-target: no seed at $seed: the first build compiles everything"
    exit 0
  fi
  mkdir -p "$(dirname "$target")"
  clone "$seed" "$target"
  echo "seed-target: $target seeded from $seed"
  exit 0
fi

[[ -d $target ]] || { echo "seed-target: no build at $target" >&2; exit 1; }
# The workspace's packages, by the name in each manifest's [package] table. reims-ctl's, a
# workspace of its own, is among them; removing its fingerprints costs nothing.
members=$(awk '/^\[/ { t = $0 } t == "[package]" && /^name = / { gsub(/"/, "", $3); print $3 }' \
  "$root"/crates/*/Cargo.toml "$root"/xtask/Cargo.toml)
new=$seed.new-$$
trap 'rm -rf "$new"' EXIT
clone "$target" "$new"
# Cargo's profile directories (with a .fingerprint) and target directories (holding them) stay;
# the tree's own outputs go: the dist trees, the option ROM, the Linux check's archive.
for e in "$new"/*; do
  [[ -d $e/.fingerprint ]] && continue
  compgen -G "$e/*/.fingerprint" > /dev/null && continue
  [[ $(basename "$e") == CACHEDIR.TAG ]] && continue
  rm -rf "$e"
done
find "$new" -type d -name .fingerprint -prune | while read -r fp; do
  rm -rf "$(dirname "$fp")/incremental"
  for m in $members; do
    rm -rf "$fp/$m"-*
  done
done
old=$seed.old-$$
[[ ! -e $seed ]] || /bin/mv "$seed" "$old"
/bin/mv "$new" "$seed"
trap - EXIT
rm -rf "$old"
echo "seed-target: $seed saved from $target"
