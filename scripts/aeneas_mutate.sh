#!/usr/bin/env bash
# Plants one change at a time in crates/neo-verified and confirms that a theorem of
# formal/compression-lean/CompressionLeanRust no longer builds. A change that still builds
# is behavior that no theorem pins. Each change costs one translation and one Lean build, so
# run it once per phase of .audit/aeneas-plan.md, not in CI. With file names, such as
# `scripts/aeneas_mutate.sh field.rs`, it runs only the changes in those files.
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
  'circuit.rs|s/pub const MATRICES: usize = 8;/pub const MATRICES: usize = 9;/'
  'degrees.rs|s/pub const RUN_ORACLES: usize = 4;/pub const RUN_ORACLES: usize = 5;/'
  'degrees.rs|s/pub const FOLD: usize = 3;/pub const FOLD: usize = 4;/'
  'degrees.rs|s/pub const DEGREE: usize = 8;/pub const DEGREE: usize = 7;/'
  'degrees.rs|s/\[RUN_ORACLES - 1, 2 \* FOLD\]/[RUN_ORACLES, 2 * FOLD]/'
  'degrees.rs|s/^    let _fits = .*$//'
  'degrees.rs|s/1usize.wrapping_shl(v as u32)/2usize.wrapping_shl(v as u32)/'
  'degrees.rs|s/add(add(add(kappa, 2), mul(2, matrices)), public_blocks)/add(add(add(kappa, 1), mul(2, matrices)), public_blocks)/'
  'degrees.rs|s/deepest = deepest.max(depths\[t\]);/deepest = depths[t];/'
  'degrees.rs|s/sum = add(sum, depths\[t\]);/sum = add(sum, 1);/'
  'degrees.rs|s/add(mul(2, mul(deepest, add(deepest, 1))), sum)/add(mul(4, mul(deepest, add(deepest, 1))), sum)/'
  'degrees.rs|s/add(relation.lane_variables, relation.block_variables)/add(relation.lane_variables, relation.lane_variables)/'
  'degrees.rs|s/add(exp2(cube), add(mul(2, relation.norm_bound as usize), 1)),/add(exp2(cube), mul(2, relation.norm_bound as usize)),/'
  'degrees.rs|s/mul(3, add(exp2(runs), exp2(rows))),/mul(3, add(exp2(runs), exp2(runs))),/'
  'degrees.rs|s/gkr(&\[cube, runs, rows, runs, pairs\]),/gkr(\&[cube, runs, rows, pairs, pairs]),/'
  'degrees.rs|s/add(add(add(add(norm, lookup), scatter), trees), test)/add(add(add(add(norm, lookup), scatter), trees), mul(2, test))/'
  'degrees.rs|s/        sub(rows, 1),/        sub(rows, 2),/'
  'degrees.rs|s/mul(2, sub(relation.d, 1)),/mul(2, relation.d),/'
  'degrees.rs|s/let blocks = add(relation.slot_variables, 1);/let blocks = add(relation.slot_variables, 0);/'
  'degrees.rs|s/gkr(&\[blocks, relation.block_variables\]),/gkr(\&[blocks, blocks]),/'
  'degrees.rs|s/    add(scatter, trees)/    add(scatter, mul(2, trees))/'
  'degrees.rs|s/add(early(relation), linear(relation))/add(early(relation), early(relation))/'
  'degrees.rs|s/run_group), fold_partials)/run_group), mul(2, fold_partials))/'
  'degrees.rs|s/sub(MATRICES, 1)/sub(MATRICES, 2)/'
  'degrees.rs|s/mul(2, add(cells, 1))/mul(2, add(cells, 2))/'
  'degrees.rs|s/add(add(add(point, outer), batching), inner)/add(add(mul(point, outer), batching), inner)/'
  'field.rs|s/pub const MODULUS: u64 = 0xffff_ffff_0000_0001;/pub const MODULUS: u64 = 0xffff_ffff_0000_0003;/'
  'field.rs|s/pub const GL_ONE: Gl = Gl(1);/pub const GL_ONE: Gl = Gl(2);/'
  'field.rs|/^pub const EXT_ONE/,/^};/s/c0: GL_ONE,/c0: GL_ZERO,/'
  'field.rs|/^pub fn canonical/,/^}/s/word < MODULUS/word <= MODULUS/'
  'field.rs|s/    Gl(reduce_word(word))/    Gl(word)/'
  'field.rs|/^fn reduce_word/,/^}/s/word.wrapping_sub(MODULUS)/word.wrapping_sub(1)/'
  'field.rs|s/    if a < room {/    if a <= room {/'
  'field.rs|/^fn sub_words/,/^}/s/if b <= a {/if b < a {/'
  'field.rs|s/word as u32 as u64/word as u16 as u64/'
  'field.rs|s/word.wrapping_shr(32)/word.wrapping_shr(31)/'
  'field.rs|s/low.wrapping_shl(32).wrapping_sub(low)/low.wrapping_shl(32)/'
  'field.rs|s/let t1 = p01.wrapping_add(low_half(t0));/let t1 = p01.wrapping_add(high_half(t0));/'
  'field.rs|/^pub fn gl_add/,/^}/s/add_words(reduce_word(a.0), reduce_word(b.0))/add_words(a.0, reduce_word(b.0))/'
  'field.rs|/^pub fn gl_sub/,/^}/s/sub_words(reduce_word(a.0), reduce_word(b.0))/sub_words(reduce_word(b.0), reduce_word(a.0))/'
  'field.rs|s/    gl_sub(GL_ZERO, a)/    gl_sub(a, GL_ZERO)/'
  'field.rs|/^pub fn gl_mul/,/^}/s/reduce_word(b.0))/reduce_word(b.0.wrapping_add(1)))/'
  'field.rs|/^pub fn gl_pow/,/^}/s/gl_mul(half, a)/gl_mul(half, half)/'
  'field.rs|/^pub fn gl_pow/,/^}/s/        GL_ONE/        GL_ZERO/'
  'field.rs|s/gl_pow(a, MODULUS.wrapping_sub(2))/gl_pow(a, MODULUS.wrapping_sub(1))/'
  'field.rs|/^pub fn ext_of_gl/,/^}/s/c1: GL_ZERO,/c1: GL_ONE,/'
  'field.rs|/^pub fn ext_add/,/^}/s/c2: gl_add(a.c2, b.c2),/c2: gl_add(a.c2, b.c1),/'
  'field.rs|/^pub fn ext_sub/,/^}/s/c0: gl_sub(a.c0, b.c0),/c0: gl_sub(b.c0, a.c0),/'
  'field.rs|s/    ext_sub(EXT_ZERO, a)/    ext_sub(a, EXT_ZERO)/'
  'field.rs|s/let d4 = gl_mul(a.c2, b.c2);/let d4 = gl_mul(a.c2, b.c1);/'
  'field.rs|/^pub fn ext_mul/,/^}/s/c0: gl_add(gl_mul(a.c0, b.c0), d3),/c0: gl_mul(a.c0, b.c0),/'
  'field.rs|/^pub fn ext_scale/,/^}/s/c1: gl_mul(a.c1, by),/c1: gl_mul(a.c0, by),/'
  'field.rs|/^pub fn ext_pow/,/^}/s/ext_mul(half, a)/ext_mul(half, half)/'
  'field.rs|/^pub fn ext_pow/,/^}/s/if e.is_multiple_of(2) {/if !e.is_multiple_of(2) {/'
  'poseidon2.rs|s/    while round < PARTIAL_ROUNDS {/    while round < 21 {/'
  'poseidon2.rs|s/    while round < HALF_FULL_ROUNDS {/    while round < 3 {/'
  'poseidon2.rs|s/^        15504881536434223753,$/        15504881536434223754,/'
  'poseidon2.rs|s/^    3785078240232899765,$/    3785078240232899766,/'
  'poseidon2.rs|s/^        7482194551502142718,$/        7482194551502142719,/'
  'poseidon2.rs|/^pub const DIAGONAL/,/^];/s/^    2,$/    5,/'
  'poseidon2.rs|s/    gl_mul(gl_mul(x4, x2), x)/    gl_mul(x4, x2)/'
  'poseidon2.rs|s/let y1 = gl_add(t01123, gl_add(x2, x2));/let y1 = gl_add(t01123, x2);/'
  'poseidon2.rs|s/let y3 = gl_add(t01233, gl_add(x0, x0));/let y3 = gl_add(t01233, gl_add(x1, x1));/'
  'poseidon2.rs|s/out\[at\] = gl_add(mixed\[at\], sum0);/out[at] = gl_add(mixed[at], sum1);/'
  'poseidon2.rs|s/gl_add(gl_mul(reduce(DIAGONAL\[lane\]), state\[lane\]), sum)/gl_add(state[lane], sum)/'
  'poseidon2.rs|s/sum = gl_add(sum, state\[lane\]);/sum = gl_add(sum, state[0]);/'
  'poseidon2.rs|s/boxed\[0\] = sbox(gl_add(state\[0\], reduce(constant)));/boxed[1] = sbox(gl_add(state[1], reduce(constant)));/'
  'poseidon2.rs|s/reduce(constants\[lane\])/reduce(constants[0])/'
  'poseidon2.rs|s/full_rounds(state, \&TERMINAL)/full_rounds(state, \&INITIAL)/'
  'poseidon2.rs|s/full_rounds(external_layer(input), \&INITIAL)/full_rounds(input, \&INITIAL)/'
)
survived=0
for mutation in "${mutations[@]}"; do
  if [ "$#" -gt 0 ] && [[ " $* " != *" ${mutation%%|*} "* ]]; then continue; fi
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
