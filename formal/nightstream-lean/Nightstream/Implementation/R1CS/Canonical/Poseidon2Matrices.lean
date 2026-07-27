import Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

/-!
Contract: the two concrete linear layers of the selected width-8 Goldilocks
Poseidon2.

Owns: the 4x4 MDS block, the external matrix built from it, the internal
diagonal, and the internal matrix — as concrete `Fin 8 → Fin 8 → Nat` maps in
canonical Goldilocks residues.

Does not own: the round schedule that applies them, round constants, or any row
program.

## Provenance

Transcribed from `crates/neo-fold-clean/src/engine/ccs_native/poseidon2.rs`
(`value_apply_mat4`, `value_external_linear`, `value_internal_linear`,
`internal_diag`).  This is the transcription `Poseidon2Core`'s authority note
permits: the canonical encoding exists to re-encode *the selected permutation*,
so it must compute the same function, and matrix values are what "the same
function" means.  What is never taken from Rust is a row count or a row layout.

`internal_diag` is written in Rust with field negation and `2⁻¹`; the residues
here are the canonical `Nat` images, and `internalDiag_half_inverse` /
`internalDiag_neg_half` prove the two non-obvious entries really are `2⁻¹` and
`-2⁻¹` rather than transcription noise.

## Shape

    mat4     = circulant [2,3,1,1]
    external = [[2·mat4,   mat4], [  mat4, 2·mat4]]
    internal = J + diag(d)          (J = all ones)

Both matrices are dense: `externalMatrix_nonzero` and `internalMatrix_nonzero`
prove no entry vanishes, which is what makes the later nonzero-coefficient
count a product rather than a survey.

Every fact here is closed by `decide` over the finite index type, so the
statements are exhaustive over all 16 or 64 entries rather than sampled.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

/-! ## The 4x4 MDS block

Read off `value_apply_mat4`, whose four assignments expand to

    out0 = 2x0 + 3x1 +  x2 +  x3
    out1 =  x0 + 2x1 + 3x2 +  x3
    out2 =  x0 +  x1 + 2x2 + 3x3
    out3 = 3x0 +  x1 +  x2 + 2x3
-/

def mat4 (row col : Fin 4) : Nat :=
  match row.val, col.val with
  | 0, 0 => 2 | 0, 1 => 3 | 0, 2 => 1 | 0, 3 => 1
  | 1, 0 => 1 | 1, 1 => 2 | 1, 2 => 3 | 1, 3 => 1
  | 2, 0 => 1 | 2, 1 => 1 | 2, 2 => 2 | 2, 3 => 3
  | 3, 0 => 3 | 3, 1 => 1 | 3, 2 => 1 | 3, 3 => 2
  | _, _ => 0

/-- Every block entry is a nonzero canonical residue. -/
theorem mat4_nonzero : ∀ row col : Fin 4, mat4 row col ≠ 0 := by decide

theorem mat4_lt : ∀ row col : Fin 4, mat4 row col < goldilocksP := by decide

/-! ## External layer

`value_external_linear` applies `mat4` to the low and high halves, forms
`sums[j] = lo[j] + hi[j]`, and returns `block[i] + sums[i % 4]`.  Expanding,
a source in the *same* half carries twice its block coefficient and a source in
the other half carries it once. -/

def externalMatrix (row col : Fin width) : Nat :=
  if row.val / 4 = col.val / 4 then
    2 * mat4 ⟨row.val % 4, by omega⟩ ⟨col.val % 4, by omega⟩
  else
    mat4 ⟨row.val % 4, by omega⟩ ⟨col.val % 4, by omega⟩

theorem externalMatrix_nonzero :
    ∀ row col : Fin width, externalMatrix row col ≠ 0 := by decide

theorem externalMatrix_lt :
    ∀ row col : Fin width, externalMatrix row col < goldilocksP := by decide

/-! ## Internal layer

`value_internal_linear` returns `sum + diag[i] * input[i]`, so the matrix is
the all-ones matrix plus the diagonal. -/

/-- Canonical residues of `internal_diag`.  Entries 3 and 5 are `2⁻¹` and
`-2⁻¹`; the rest are small integers and their negations. -/
def internalDiag (lane : Fin width) : Nat :=
  match lane.val with
  | 0 => goldilocksP - 2
  | 1 => 1
  | 2 => 2
  | 3 => (goldilocksP + 1) / 2
  | 4 => 3
  | 5 => (goldilocksP - 1) / 2
  | 6 => goldilocksP - 3
  | _ => goldilocksP - 4

/-- **Lane 3 really is `2⁻¹`.**  Guards the transcription of
`F::from_u64(2).inverse()`. -/
theorem internalDiag_half_inverse :
    2 * internalDiag ⟨3, by decide⟩ % goldilocksP = 1 := by decide

/-- **Lane 5 really is `-2⁻¹`.** -/
theorem internalDiag_neg_half :
    (internalDiag ⟨5, by decide⟩ + internalDiag ⟨3, by decide⟩) % goldilocksP = 0 := by
  decide

theorem internalDiag_lt : ∀ lane : Fin width, internalDiag lane < goldilocksP := by
  decide

def internalMatrix (row col : Fin width) : Nat :=
  if row.val = col.val then (1 + internalDiag row) % goldilocksP else 1

/-- No internal entry vanishes: off-diagonal is `1`, and no `1 + diag` is a
multiple of the prime. -/
theorem internalMatrix_nonzero :
    ∀ row col : Fin width, internalMatrix row col ≠ 0 := by decide

theorem internalMatrix_lt :
    ∀ row col : Fin width, internalMatrix row col < goldilocksP := by decide

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
