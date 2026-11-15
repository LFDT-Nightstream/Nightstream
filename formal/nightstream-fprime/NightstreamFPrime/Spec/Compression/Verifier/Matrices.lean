import NightstreamFPrime.Spec.Compression.Verifier.Model

/-! The CCS matrices of a relation and the matrix weights that they give (SPEC.md, Algorithm 9,
step 7). Cell `x = D·j + h` of a witness is coefficient `h` of its ring column `j`. Ring column `j`
of `M_mᵀ · eq(r, ·)` is `w_mj`, and the key element of block `j` in `Eval_A` of matrix `m` is
`conjugate(w_mj)` (`Matrices.form`). -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra

/-- The CCS matrices of one relation (SuperNeo v1.2 Definition 19, Rust `MatrixShape`): `entry m
row x` is the entry of matrix `m` at `row` and cell `x`. -/
structure Matrices (F : Type) where
  rows : ℕ
  columns : ℕ
  entry : ℕ → ℕ → ℕ → F

variable {F K : Type} [Field F] [Field K] [Algebra F K]

/-- Lane `h` of the key element of block `j` in `Eval_A` of matrix `m` at the point `r`:
`conjugate(w_mj)`, where lane `l` of `w_mj` is `Σ_row eq(r, row) · M_m[row, D·j + l]`. -/
def Matrices.form (M : Matrices F) (D : ℕ) (r : List K) (m j : ℕ) : ℕ → K :=
  conjugate D fun l => ∑ row ∈ range M.rows, eqAt r row * algebraMap F K (M.entry m row (D * j + l))

end NightstreamFPrime.Spec.Compression.Verifier
