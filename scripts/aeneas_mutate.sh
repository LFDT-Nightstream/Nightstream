#!/usr/bin/env bash
# Plants one change at a time in crates/neo-verified and confirms that a theorem of
# formal/compression-lean/CompressionLeanRust no longer builds. A change that still builds
# is behavior that no theorem pins. Each change costs one translation and one Lean build, so
# run it once per phase of .audit/aeneas-plan.md, not in CI.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/crates/neo-verified/src/layout.rs
backup=$(mktemp)
cp "$src" "$backup"
restore() {
  cp "$backup" "$src"
  rm -f "$backup"
  "$root/scripts/aeneas_extract.sh" >/dev/null
}
trap restore EXIT

mutations=(
  's/if rest == 0 {/if rest == 1 {/'
  's/\[wide, 0\]/[wide, 1]/'
  's/^        .checked_sub(1)$/        .checked_sub(2)/'
  's/n.checked_sub(1)/n.checked_sub(0)/'
  's/if count <= 2 {/if count <= 1 {/'
  's/^        1$/        2/'
  's/1usize.wrapping_shl(k)/2usize.wrapping_shl(k)/'
)
survived=0
for mutation in "${mutations[@]}"; do
  cp "$backup" "$src"
  sed "$mutation" "$backup" >"$src"
  if cmp -s "$backup" "$src"; then
    echo "did not apply: $mutation"
    survived=1
  elif ! "$root/scripts/aeneas_extract.sh" >/dev/null 2>&1; then
    echo "rejected by the extraction gate: $mutation"
  elif (cd "$root/formal/compression-lean" && lake build CompressionLeanRust >/dev/null 2>&1); then
    echo "SURVIVED: $mutation"
    survived=1
  else
    echo "killed: $mutation"
  fi
done
exit "$survived"
