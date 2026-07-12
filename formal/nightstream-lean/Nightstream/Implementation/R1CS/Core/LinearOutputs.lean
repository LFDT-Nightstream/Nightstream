import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: semantic decoding for linear verifier output checks.

Rust emits both `output - rhs = 0` and `rhs - output = 0`, depending on
which side of `enforce_eq` is a computed linear combination. This module
normalizes those exact row shapes to the independent equation
`assignment output = expected` and proves soundness and completeness.
-/

namespace Nightstream.Implementation.R1CS.LinearOutputs

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

inductive Orientation where
  | forward
  | reverse
deriving DecidableEq, Repr

structure Check where
  output : Nat
  terms : List (Nat × Nat)
  orientation : Orientation
deriving DecidableEq, Repr

def Check.row (check : Check) : Row :=
  match check.orientation with
  | .forward => builderLinearRow check.output check.terms
  | .reverse =>
      ⟨check.terms ++ [(check.output, goldilocksP - 1)], [(0, 1)], []⟩

def Check.expected (assignment : Nat → Nat) (check : Check) : Nat :=
  lcEval assignment check.terms

def Check.Canonical (check : Check) : Prop :=
  CanonicalTerms check.terms

instance (check : Check) : Decidable check.Canonical := by
  unfold Check.Canonical
  infer_instance

private theorem rawLcEval_append (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    rawLcEval assignment (left ++ right) =
      rawLcEval assignment left + rawLcEval assignment right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]

private theorem reverse_sound
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (check : Check)
    (orientation : check.orientation = .reverse)
    (holds : RowHolds assignment check.row) :
    assignment check.output = check.expected assignment := by
  have modulusPositive : 0 < goldilocksP := by decide
  have outputLt := canonicalAssignment check.output
  have expectedLt : check.expected assignment < goldilocksP := by
    exact Nat.mod_lt _ modulusPositive
  have claimedCancel :
      (rawLcEval assignment check.terms +
        (goldilocksP - 1) * assignment check.output) % goldilocksP = 0 := by
    simpa [Check.row, orientation, RowHolds, lcEval_eq_raw_mod,
      rawLcEval_append, rawLcEval, one, Nat.add_assoc] using holds
  have outputCancel :
      (assignment check.output +
        (goldilocksP - 1) * assignment check.output) % goldilocksP = 0 := by
    have factor :
        assignment check.output +
            (goldilocksP - 1) * assignment check.output =
          goldilocksP * assignment check.output := by
      simp only [goldilocksP]
      omega
    rw [factor]
    simp
  have rawMod :
      rawLcEval assignment check.terms % goldilocksP =
        check.expected assignment := by
    exact (lcEval_eq_raw_mod assignment check.terms).symm
  have expectedCancel :
      (check.expected assignment +
        (goldilocksP - 1) * assignment check.output) % goldilocksP = 0 := by
    rw [Nat.add_mod, Nat.mod_eq_of_lt expectedLt]
    rw [Nat.add_mod, rawMod] at claimedCancel
    exact claimedCancel
  simp only [goldilocksP] at outputLt expectedLt outputCancel expectedCancel ⊢
  omega

theorem Check.sound
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (check : Check)
    (canonical : check.Canonical)
    (holds : RowHolds assignment check.row) :
    assignment check.output = check.expected assignment := by
  cases orientation : check.orientation with
  | forward =>
      exact builderLinearRow_sound canonicalAssignment one check.output
        check.terms canonical (by simpa [Check.row, orientation] using holds)
  | reverse => exact reverse_sound canonicalAssignment one check orientation holds

private theorem reverse_complete
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (check : Check)
    (orientation : check.orientation = .reverse)
    (equal : assignment check.output = check.expected assignment) :
    RowHolds assignment check.row := by
  have outputLt := canonicalAssignment check.output
  have outputCancel :
      (assignment check.output +
        (goldilocksP - 1) * assignment check.output) % goldilocksP = 0 := by
    have factor :
        assignment check.output +
            (goldilocksP - 1) * assignment check.output =
          goldilocksP * assignment check.output := by
      simp only [goldilocksP]
      omega
    rw [factor]
    simp
  have rawMod :
      rawLcEval assignment check.terms % goldilocksP =
        assignment check.output := by
    calc
      rawLcEval assignment check.terms % goldilocksP =
          check.expected assignment :=
        (lcEval_eq_raw_mod assignment check.terms).symm
      _ = assignment check.output := equal.symm
  have claimedCancel :
      (rawLcEval assignment check.terms +
        (goldilocksP - 1) * assignment check.output) % goldilocksP = 0 := by
    calc
      (rawLcEval assignment check.terms +
          (goldilocksP - 1) * assignment check.output) % goldilocksP =
          (rawLcEval assignment check.terms % goldilocksP +
            ((goldilocksP - 1) * assignment check.output) % goldilocksP) %
              goldilocksP := Nat.add_mod _ _ _
      _ = (assignment check.output +
            ((goldilocksP - 1) * assignment check.output) % goldilocksP) %
              goldilocksP := by rw [rawMod]
      _ = (assignment check.output % goldilocksP +
            ((goldilocksP - 1) * assignment check.output) % goldilocksP) %
              goldilocksP := by rw [Nat.mod_eq_of_lt outputLt]
      _ = (assignment check.output +
            (goldilocksP - 1) * assignment check.output) % goldilocksP :=
          (Nat.add_mod _ _ _).symm
      _ = 0 := outputCancel
  simpa [Check.row, orientation, RowHolds, lcEval_eq_raw_mod,
    rawLcEval_append, rawLcEval, one, Nat.add_assoc] using claimedCancel

theorem Check.complete
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (check : Check)
    (canonical : check.Canonical)
    (equal : assignment check.output = check.expected assignment) :
    RowHolds assignment check.row := by
  cases orientation : check.orientation with
  | forward =>
      simpa [Check.row, orientation] using
        builderLinearRow_complete one check.output check.terms canonical equal
  | reverse =>
      exact reverse_complete canonicalAssignment one check orientation equal

def rows (checks : List Check) : List Row := checks.map Check.row

def Canonical (checks : List Check) : Prop :=
  ∀ check ∈ checks, check.Canonical

instance (checks : List Check) : Decidable (Canonical checks) := by
  unfold Canonical
  infer_instance

theorem rows_sound
    {checks : List Check} {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (canonical : Canonical checks)
    (satisfies : Satisfies (rows checks) assignment) :
    ∀ check ∈ checks,
      assignment check.output = check.expected assignment := by
  intro check member
  exact check.sound canonicalAssignment one (canonical check member)
    (satisfies check.row (List.mem_map.mpr ⟨check, member, rfl⟩))

theorem rows_complete
    {checks : List Check} {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (canonical : Canonical checks)
    (equalities : ∀ check ∈ checks,
      assignment check.output = check.expected assignment) :
    Satisfies (rows checks) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨check, checkMember, rfl⟩
  exact check.complete canonicalAssignment one (canonical check checkMember)
    (equalities check checkMember)

end Nightstream.Implementation.R1CS.LinearOutputs
