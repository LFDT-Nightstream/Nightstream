import CompressionLean.Lib.Cube
import Mathlib.Algebra.Algebra.Defs
import Mathlib.Algebra.BigOperators.Fin

namespace CompressionLean.Shrink

open CompressionLean.Lib

structure Entry (F : Type) where
  matrix : Fin 8
  row : ℕ
  col : ℕ
  coef : F

structure Ccs (F : Type) where
  rowVars : ℕ
  cellVars : ℕ
  entries : List (Entry F)
  statement : List F

def constraint {R : Type} [CommRing R] (v : Fin 8 → R) : R :=
  v 0 ^ 7 + v 1 * v 2 + v 3 * v 4 + v 5 * v 6 - v 7

section Base

variable {F : Type} [Field F]

def rowValue (es : List (Entry F)) (z : ℕ → F) (j : Fin 8) (r : ℕ) : F :=
  (es.map fun e => if e.matrix = j ∧ e.row = r then e.coef * z e.col else 0).sum

namespace Ccs

variable (C : Ccs F)

def publicTable (i : ℕ) : F := if i = 0 then 1 else C.statement.getD (i - 1) 0

def z (w : ℕ → F) (c : ℕ) : F :=
  if c < 2 ^ C.cellVars then w c else C.publicTable (c - 2 ^ C.cellVars)

end Ccs

end Base

section Extension

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

def weight (es : List (Entry F)) (rx : List K) (ρ : K) (y : ℕ) : K :=
  (es.map fun e => if e.col = y then ρ ^ (e.matrix : ℕ) * eqAt rx e.row * ι e.coef else 0).sum

def Ccs.publicValue (C : Ccs F) (q : List K) : K :=
  ∑ i ∈ Finset.range (C.statement.length + 1), eqAt q i * ι (C.publicTable i)

end Extension

end CompressionLean.Shrink
