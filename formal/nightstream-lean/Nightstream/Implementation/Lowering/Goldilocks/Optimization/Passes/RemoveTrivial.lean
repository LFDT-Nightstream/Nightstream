import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Pass
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.R1CS

/-!
Contract: remove syntactically trivial normalized R1CS rows.

Assurance tier: model-level.

Owns: removal of `0 * B = 0` and `A * 0 = 0`, exact acceptance,
observable preservation, and source-row provenance for every retained row.

Does not own: normalization, allocation removal, duplicate equations,
protocol-specific identities, or a manifest.

Emits constraints: the retained source rows in their original order.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.RemoveTrivial

open Nightstream.Implementation.Lowering.Goldilocks

universe u

private abbrev Assignment := R1CS.Assignment

/-- This predicate is intentionally syntactic. Run normalization first. -/
def Trivial (row : OwnedRow) : Prop :=
  row.row.c = [] /\ (row.row.a = [] \/ row.row.b = [])

instance trivialDecidable (row : OwnedRow) : Decidable (Trivial row) :=
  by
    unfold Trivial
    infer_instance

def rows (source : List OwnedRow) : List OwnedRow :=
  source.filter fun row => decide (not (Trivial row))

theorem trivial_holds
    (row : OwnedRow)
    (assignment : Assignment)
    (trivial : Trivial row) :
    row.row.Holds assignment := by
  rcases trivial with ⟨cEmpty, aEmpty | bEmpty⟩
  · simp [Row.Holds, aEmpty, cEmpty, Fin.zero_mul]
  · simp [Row.Holds, bEmpty, cEmpty, Fin.mul_zero]

theorem satisfies_rows_iff
    (source : List OwnedRow)
    (assignment : Assignment) :
    Goldilocks.Satisfies (rows source) assignment <->
      Goldilocks.Satisfies source assignment := by
  induction source with
  | nil =>
      rfl
  | cons row source inductionHypothesis =>
      change
        Goldilocks.Satisfies
            ((row :: source).filter fun item =>
              decide (not (Trivial item))) assignment <->
          Goldilocks.Satisfies (row :: source) assignment
      simp only [List.filter_cons]
      by_cases trivial : Trivial row
      · rw [if_neg (by simpa using trivial)]
        simp only [Goldilocks.satisfies_cons]
        change
          Goldilocks.Satisfies (rows source) assignment <->
            row.row.Holds assignment /\
              Goldilocks.Satisfies source assignment
        rw [inductionHypothesis]
        exact
          ⟨fun tail => ⟨trivial_holds row assignment trivial, tail⟩,
            fun all => all.2⟩
      · rw [if_pos (by simpa using trivial)]
        simp only [Goldilocks.satisfies_cons]
        change
          (row.row.Holds assignment /\
              Goldilocks.Satisfies (rows source) assignment) <->
            row.row.Holds assignment /\
              Goldilocks.Satisfies source assignment
        rw [inductionHypothesis]

def target
    {Observable : Type u}
    (one : ColumnId)
    (source : List OwnedRow)
    (observe : Assignment -> Observable) :=
  R1CS.system one (rows source) observe

def replacement
    {Observable : Type u}
    (one : ColumnId)
    (source : List OwnedRow)
    (observe : Assignment -> Observable)
    (degreeLimit : Nat)
    (withinLimit : R1CS.degree <= degreeLimit) :
    Optimization.Replacement
      (R1CS.system one source observe)
      (target one source observe)
      degreeLimit where
  recover := fun assignment => assignment
  derive := fun assignment => assignment
  sound := by
    intro assignment accepted
    exact ⟨accepted.1,
      (satisfies_rows_iff source assignment).mp accepted.2⟩
  complete := by
    intro assignment accepted
    exact ⟨accepted.1,
      (satisfies_rows_iff source assignment).mpr accepted.2⟩
  recover_observes := fun _ _ => rfl
  derive_observes := fun _ _ => rfl
  source_degree := withinLimit
  target_degree := withinLimit

def result
    {Observable : Type u}
    (one : ColumnId)
    (source : List OwnedRow)
    (observe : Assignment -> Observable)
    (degreeLimit : Nat)
    (withinLimit : R1CS.degree <= degreeLimit) :
    Optimization.Result
      (R1CS.system one source observe) degreeLimit where
  target := target one source observe
  replacement :=
    replacement one source observe degreeLimit withinLimit

theorem retained_from_source
    (source : List OwnedRow)
    (row : OwnedRow)
    (member : row ∈ rows source) :
    row ∈ source := by
  exact (List.mem_filter.1 member).1

theorem rows_length_le (source : List OwnedRow) :
    (rows source).length <= source.length := by
  exact List.length_filter_le _ _

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.RemoveTrivial
