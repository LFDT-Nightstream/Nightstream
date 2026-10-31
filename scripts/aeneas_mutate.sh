#!/usr/bin/env bash
# Plants one change at a time in crates/neo-verified and confirms that a theorem of
# formal/compression-lean/CompressionLeanRust no longer builds. A change that still builds
# is behavior that no theorem pins. Each change costs one translation and one Lean build, so
# run it once per phase of .audit/aeneas-plan.md, not in CI.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/crates/neo-verified/src
backup=$(mktemp -d)
cp "$src"/*.rs "$backup/"
restore() {
  cp "$backup"/*.rs "$src/"
  rm -rf "$backup"
  "$root/scripts/aeneas_extract.sh" >/dev/null
}
trap restore EXIT

# Each entry is `<file in crates/neo-verified/src>|<sed expression>`.
mutations=(
  'layout.rs|s/if rest == 0 {/if rest == 1 {/'
  'layout.rs|s/\[wide, 0\]/[wide, 1]/'
  'layout.rs|s/^        .checked_sub(1)$/        .checked_sub(2)/'
  'layout.rs|s/n.checked_sub(1)/n.checked_sub(0)/'
  'layout.rs|s/if count <= 2 {/if count <= 1 {/'
  'layout.rs|s/^        1$/        2/'
  'layout.rs|s/1usize.wrapping_shl(k)/2usize.wrapping_shl(k)/'
  'layout.rs|s/^    let _fits = .*$//'
  'layout.rs|s/let start = wide.checked_mul(count)/let start = narrow.checked_mul(count)/'
  'layout.rs|s/Part { start: 0, stride: wide }/Part { start: 1, stride: wide }/'
  'layout.rs|s/if k < wide.stride {/if k <= wide.stride {/'
  'layout.rs|s/(narrow, k.checked_sub(wide.stride).unwrap())/(narrow, k)/'
  'layout.rs|s/slot(kind.rows(), count, b, k)/slot(kind.cells(), count, b, k)/'
  'layout.rs|s/slot(kind.cells(), count, b, k)/slot(kind.cells(), count, k, b)/'
  'layout.rs|s/rows_end(shape).checked_add(g)/cells_end(shape).checked_add(g)/'
  'layout.rs|s/rows_end(shape).checked_add(shape.glue_rows)/rows_end(shape).checked_add(shape.glue_cells)/'
  'layout.rs|s/shape.publics.checked_add(1)/shape.publics.checked_add(2)/'
  'layout.rs|s/two_pow(cell_variables(shape) as u32)/two_pow(row_variables(shape) as u32)/'
  'layout.rs|s/Wire::Glue(cell) => cells_end(shape)/Wire::Glue(cell) => rows_end(shape)/'
  'layout.rs|s/block_cell(shape, kind, block as usize, cell as usize)/block_cell(shape, kind, cell as usize, block as usize)/'
  'layout.rs|s/Wire::Public(index) => one(shape)/Wire::Public(index) => cells_end(shape)/'
  'layout.rs|s/None => one(shape),/None => cells_end(shape),/'
  'circuit.rs|s/pub const SBOXES: usize = 150;/pub const SBOXES: usize = 151;/'
  'circuit.rs|s/Kind::Permutation => CELLS,/Kind::Permutation => ROWS,/'
)
survived=0
for mutation in "${mutations[@]}"; do
  file=$src/${mutation%%|*}
  expression=${mutation#*|}
  cp "$backup"/*.rs "$src/"
  sed "$expression" "$backup/${mutation%%|*}" >"$file"
  if cmp -s "$backup/${mutation%%|*}" "$file"; then
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
