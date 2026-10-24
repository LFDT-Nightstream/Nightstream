import CompressionLean.Ports.Shrink.Ccs
import CompressionLean.Lib.Combination

/-!
Owns the port of the tables of the shrink argument's two sum-checks and the
batching between them (`mod.rs:233-288`).

- Outer: the zero-check `Σ_x eq(τ, x)·P(M_0·z(x), ..., M_7·z(x)) = 0` over
  the rows. Its rounds are `combRound` of nine tables (`eq(τ, ·)` and the
  `M_j·z`) under `outerComb = X_eq·P`, of total degree 8 (`DEGREE`,
  `mod.rs:40-41`).
- Batching: `Σ_j ρ^j·v_j` (`mod.rs:255-256`), and `batchPoly`, the
  difference of two batches as a polynomial in `ρ`.
- Inner: `Σ_y weight(y)·z(y)` over the columns, as `combRound` of two tables
  under `productComb`.

`Shrink/Rounds.lean` proves their true sums.
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

section Outer

/-- `X_eq·P(X_0, ..., X_7)`: the outer sum-check's combination. -/
noncomputable def outerComb : MvPolynomial (Option (Fin 8)) K :=
  MvPolynomial.X none * constraint fun j => MvPolynomial.X (some j)

/-- The outer sum-check's tables: `eq(τ, ·)` and the eight `M_j·z` on the
row cube. -/
def outerTables (es : List (Entry F)) (z : ℕ → F) (τ : List K) : Option (Fin 8) → ℕ → K
  | none => eqAt τ
  | some j => fun r => ι (rowValue es z j r)

/-- The constraint table: `P` of the row values. It is zero on the row cube
when the rows hold. -/
def constraintTable (es : List (Entry F)) (z : ℕ → F) (r : ℕ) : K :=
  ι (constraint fun j => rowValue es z j r)

end Outer

section Batch

/-- `Σ_j ρ^j·v_j` (`mod.rs:255-256`). -/
def batch (ρ : K) (v : Fin 8 → K) : K := ∑ j : Fin 8, ρ ^ (j : ℕ) * v j

/-- `Σ_j (v_j - t_j)·X^j`: claimed values against true values. -/
noncomputable def batchPoly (v t : Fin 8 → K) : K[X] := ∑ j : Fin 8, C (v j - t j) * X ^ (j : ℕ)

end Batch

section Inner

/-- The inner sum-check's tables: the weights `Σ_j ρ^j·M_j(rx, ·)` and `z`. -/
def innerTables (es : List (Entry F)) (z : ℕ → F) (rx : List K) (ρ : K) : Fin 2 → ℕ → K :=
  fun i => if i = 0 then weight es rx ρ else fun y => ι (z y)

end Inner

end CompressionLean.Shrink
