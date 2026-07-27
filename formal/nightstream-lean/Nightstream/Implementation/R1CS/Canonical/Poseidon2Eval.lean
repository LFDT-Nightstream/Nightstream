import Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
import Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-!
Contract: how a carried combination evaluates.

Owns: the modular algebra relating a symbolically carried `LinComb` to the
field value it denotes, and the reference matrix-vector product.

Does not own: the round schedule (`Poseidon2Schedule`), the support bound
(`Poseidon2Support`), or any row program.

## Why this is the missing hinge

`Poseidon2Core.applyMatrix_emits_no_rows` is **definitional** — it records that
the chosen normal form emits no row for a linear layer, and says nothing about
whether the carried combinations implement the matrix.  That gap is what
`lcEval_applyMatrix` closes: evaluating the combination a linear layer produces
gives exactly the matrix-vector product of the evaluations of its sources.

Without it, "linear layers are free" is a naming convention.  With it, it is a
theorem, and the round induction has a base to stand on.

The plumbing is unavoidable: `scale` reduces each coefficient modulo the prime
as it goes, so a scaled combination is only *congruent* to the scaled sum, not
equal to it.  `rawSum_scale_mod` is where that congruence is discharged.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-! ## Modular plumbing -/

theorem scale_cons
    (factor : Nat) (term : Nat × Nat) (rest : Poseidon2Core.LinComb) :
    scale factor (term :: rest)
      = (term.1, factor * term.2 % goldilocksP) :: scale factor rest := rfl

/-- Reducing a product before multiplying again changes nothing modulo the
prime.  This is the one step `scale` forces at every coefficient. -/
theorem mul_mod_shift (a b c : Nat) :
    (a * b % goldilocksP) * c % goldilocksP = a * (b * c) % goldilocksP := by
  rw [Nat.mod_mul_mod, Nat.mul_assoc]

/-- Multiplying by an already-reduced value is the same as reducing after. -/
theorem mul_mod_right_reduce (a b : Nat) :
    a * (b % goldilocksP) % goldilocksP = a * b % goldilocksP := by
  rw [Nat.mul_comm a, Nat.mod_mul_mod, Nat.mul_comm]

/-- **Scaling a combination scales its value.** -/
theorem rawSum_scale_mod
    (z : Nat → Nat) (factor : Nat) (comb : Poseidon2Core.LinComb) :
    rawSum z (scale factor comb) % goldilocksP
      = factor * rawSum z comb % goldilocksP := by
  induction comb with
  | nil => simp [scale, rawSum]
  | cons term rest hypothesis =>
      rw [scale_cons, rawSum_cons, rawSum_cons, Nat.mul_add]
      dsimp only
      rw [Nat.add_mod, hypothesis, mul_mod_shift, ← Nat.add_mod]

theorem lcEval_scale
    (z : Nat → Nat) (factor : Nat) (comb : Poseidon2Core.LinComb) :
    lcEval z (scale factor comb) = factor * lcEval z comb % goldilocksP := by
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum, rawSum_scale_mod,
    mul_mod_right_reduce]

/-- Pointwise congruence lifts to a sum. -/
theorem sum_mod_congr {α : Type} (list : List α) (f g : α → Nat)
    (agree : ∀ x, f x % goldilocksP = g x % goldilocksP) :
    (list.map f).sum % goldilocksP = (list.map g).sum % goldilocksP := by
  induction list with
  | nil => simp
  | cons head tail hypothesis =>
      simp only [List.map_cons, List.sum_cons]
      rw [Nat.add_mod, agree head, hypothesis, ← Nat.add_mod]

/-! ## The reference linear layer

Values, not combinations: this is what the encoding must reproduce. -/

def applyMatrixValues
    (matrix : Fin width → Fin width → Nat) (values : Fin width → Nat)
    (target : Fin width) : Nat :=
  ((List.finRange width).map
    (fun source => matrix target source * values source)).sum % goldilocksP

/-- **A linear layer is free *and correct*.**  The combination a linear layer
produces evaluates to the matrix-vector product of its sources' evaluations —
so emitting no row loses nothing.  This upgrades
`applyMatrix_emits_no_rows` from a definitional record to a semantic fact. -/
theorem lcEval_applyMatrix
    (z : Nat → Nat) (matrix : Fin width → Fin width → Nat)
    (state : State) (target : Fin width) :
    lcEval z (applyMatrix matrix state target)
      = applyMatrixValues matrix (fun source => lcEval z (state source))
          target := by
  rw [lcEval_eq_rawSum]
  unfold applyMatrix applyMatrixValues
  rw [rawSum_normalize, rawSum_flatMap]
  refine sum_mod_congr _ _ _ ?_
  intro source
  dsimp only
  rw [rawSum_scale_mod, lcEval_eq_rawSum, mul_mod_right_reduce]

/-! ## Round constants -/

/-- Adding a round constant on the constant wire adds it to the value, provided
the wire carries one. -/
theorem lcEval_addConstant
    (z : Nat → Nat) (constant : Nat) (comb : Poseidon2Core.LinComb)
    (constantWire : z 0 = 1) :
    lcEval z (addConstant constant comb)
      = (constant + lcEval z comb) % goldilocksP := by
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum, addConstant, rawSum_cons]
  dsimp only
  rw [constantWire, Nat.mul_one, Nat.add_mod_mod]

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
