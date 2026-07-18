import Nightstream.Implementation.R1CS.Core.Program

/-!
Shared semantic decoder for the subtraction-form linear equalities used by the
`Pi_RLC` selection tail.

Owns: the proof that an exact row `(left - right) * 1 = 0` forces equality of
the two canonical field linear combinations.

Does not own: any particular selector/binding equation, integer no-wrap
bounds, production placement, Rust conformance, row removal, or costs.

Emits constraints: no.

Authority boundary: this theorem decodes an already accepted R1CS equation.
It does not decide which side is semantically authoritative; each caller must
provide that protocol interpretation separately.

| Protocol | Phase | Constraint family | Input row | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/selection | subtraction-form linear equality | `(left ++ negate right) * 1 = 0` | `lcEval left = lcEval right` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.LinearEquality

open Nightstream.Implementation.R1CS

private theorem rawLcEval_append
    (assignment : Nat -> Nat) (left right : List (Nat × Nat)) :
    Program.rawLcEval assignment (left ++ right) =
      Program.rawLcEval assignment left + Program.rawLcEval assignment right := by
  induction left with
  | nil => simp [Program.rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [Program.rawLcEval, inductionHypothesis, Nat.add_assoc]

private theorem lcEval_append
    (assignment : Nat -> Nat) (left right : List (Nat × Nat)) :
    lcEval assignment (left ++ right) =
      (lcEval assignment left + lcEval assignment right) % goldilocksP := by
  rw [Program.lcEval_eq_raw_mod, rawLcEval_append, Nat.add_mod,
    ← Program.lcEval_eq_raw_mod, ← Program.lcEval_eq_raw_mod]

/-- Decode one exact subtraction-form equality row. -/
theorem sound
    {assignment : Nat -> Nat}
    (one : assignment 0 = 1)
    (left right : List (Nat × Nat))
    (rightCanonical : Program.CanonicalTerms right)
    (holds : RowHolds assignment
      ⟨left ++ Program.negateTerms right, [(0, 1)], []⟩) :
    lcEval assignment left = lcEval assignment right := by
  have combinedZero :
      lcEval assignment (left ++ Program.negateTerms right) = 0 := by
    simpa [RowHolds, lcEval, one] using holds
  have rightCancel := Program.lcEval_append_negateTerms_eq_zero
    assignment right rightCanonical
  rw [lcEval_append] at combinedZero rightCancel
  have modulusPositive : 0 < goldilocksP := by decide
  have leftLt : lcEval assignment left < goldilocksP := by
    rw [Program.lcEval_eq_raw_mod]
    exact Nat.mod_lt _ modulusPositive
  have rightLt : lcEval assignment right < goldilocksP := by
    rw [Program.lcEval_eq_raw_mod]
    exact Nat.mod_lt _ modulusPositive
  have complementLt :
      lcEval assignment (Program.negateTerms right) < goldilocksP := by
    rw [Program.lcEval_eq_raw_mod]
    exact Nat.mod_lt _ modulusPositive
  simp only [goldilocksP] at combinedZero rightCancel leftLt rightLt complementLt ⊢
  omega

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.LinearEquality
