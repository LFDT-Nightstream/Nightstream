import CompressionLean.Lib.Cube
import Mathlib.Algebra.Algebra.Defs
import Mathlib.Algebra.BigOperators.Fin

/-!
Owns the port of the shrink CCS (`crates/neo-spartan/src/shrink/mod.rs`): a CCS
over `2^rowVars` rows and the column vector `z = (w, 1, x)` on
`2^(cellVars + 1)` columns, and the sums that the shrink verifier evaluates.

- The eight matrices `X, A_0, B_0, A_1, B_1, A_2, B_2, C`
  (`circuit/block.rs::MATRICES`) are one list of nonzero entries. A row holds
  when `(X·z)^7 + Σ_i (A_i·z)·(B_i·z) - C·z = 0` (`constraint`,
  `mod.rs::constraint`).
- `z = (w, 1, x)` (`mod.rs::z_table`): `w` on the low `2^cellVars` columns,
  then the constant one and the statement words.
- `rowValue` is `M_j·z`. `weight` is `Σ_j ρ^j·M_j(rx, ·)`, the inner
  sum-check's table (`mod.rs::weights`). `publicValue` is `x̃`
  (`mod.rs::verify`).

The relation, its entry sum and the extension lemmas are in `Shrink/Ccs.lean`.
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib

/-- One nonzero of the eight matrices: matrix `j` (`X = 0`, `A_i = 1 + 2i`,
`B_i = 2 + 2i`, `C = 7`), row, column of `z`, coefficient. -/
structure Entry (F : Type) where
  matrix : Fin 8
  row : ℕ
  col : ℕ
  coef : F

/-- The shrink CCS of one program: its sizes, its entries and its statement
words (`x`). -/
structure Ccs (F : Type) where
  rowVars : ℕ
  cellVars : ℕ
  entries : List (Entry F)
  statement : List F

/-- `P(v) = v_X^7 + v_{A0}·v_{B0} + v_{A1}·v_{B1} + v_{A2}·v_{B2} - v_C`
(`mod.rs::constraint`). -/
def constraint {R : Type} [CommRing R] (v : Fin 8 → R) : R :=
  v 0 ^ 7 + v 1 * v 2 + v 3 * v 4 + v 5 * v 6 - v 7

section Base

variable {F : Type} [Field F]

/-- `(M_j·z)(r)`: row `r` of matrix `j` against `z`. -/
def rowValue (es : List (Entry F)) (z : ℕ → F) (j : Fin 8) (r : ℕ) : F :=
  (es.map fun e => if e.matrix = j ∧ e.row = r then e.coef * z e.col else 0).sum

namespace Ccs

variable (C : Ccs F)

/-- The upper half of `z`: the constant one, then the statement words
(`mod.rs::z_table`); zero after them. -/
def publicTable (i : ℕ) : F := if i = 0 then 1 else C.statement.getD (i - 1) 0

/-- `z = (w, 1, x)` (`mod.rs::z_table`). -/
def z (w : ℕ → F) (c : ℕ) : F :=
  if c < 2 ^ C.cellVars then w c else C.publicTable (c - 2 ^ C.cellVars)

end Ccs

end Base

section Extension

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

/-- `Σ_j ρ^j·M_j(rx, y)` at column `y` (`weights`, `mod.rs::weights`). -/
def weight (es : List (Entry F)) (rx : List K) (ρ : K) (y : ℕ) : K :=
  (es.map fun e => if e.col = y then ρ ^ (e.matrix : ℕ) * eqAt rx e.row * ι e.coef else 0).sum

/-- `x̃ = Σ_i eq(q, i)·public_i` over `public = (1, statement)`
(`mod.rs::verify`). -/
def Ccs.publicValue (C : Ccs F) (q : List K) : K :=
  ∑ i ∈ Finset.range (C.statement.length + 1), eqAt q i * ι (C.publicTable i)

end Extension

end CompressionLean.Shrink
