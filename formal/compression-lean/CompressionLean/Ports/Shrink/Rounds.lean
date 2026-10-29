import CompressionLean.Ports.Shrink.Ccs
import CompressionLean.Lib.Combination

namespace CompressionLean.Shrink

open CompressionLean.Lib Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

section Outer

noncomputable def outerComb : MvPolynomial (Option (Fin 8)) K :=
  MvPolynomial.X none * constraint fun j => MvPolynomial.X (some j)

def outerTables (es : List (Entry F)) (z : ℕ → F) (τ : List K) : Option (Fin 8) → ℕ → K
  | none => eqAt τ
  | some j => fun r => ι (rowValue es z j r)

def constraintTable (es : List (Entry F)) (z : ℕ → F) (r : ℕ) : K :=
  ι (constraint fun j => rowValue es z j r)

end Outer

section Batch

def batch (ρ : K) (v : Fin 8 → K) : K := ∑ j : Fin 8, ρ ^ (j : ℕ) * v j

noncomputable def batchPoly (v t : Fin 8 → K) : K[X] := ∑ j : Fin 8, C (v j - t j) * X ^ (j : ℕ)

end Batch

section Inner

def innerTables (es : List (Entry F)) (z : ℕ → F) (rx : List K) (ρ : K) : Fin 2 → ℕ → K :=
  fun i => if i = 0 then weight es rx ρ else fun y => ι (z y)

end Inner

end CompressionLean.Shrink
