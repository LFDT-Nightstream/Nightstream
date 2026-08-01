import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.R1CS
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Pass

/-!
Contract: certified sparse-row normalization.

Assurance tier: model-level.

Owns: merging repeated columns, removing zero coefficients, and sorting each
linear combination without changing accepted assignments or observables.

Does not own: row removal, column removal, a manifest, or protocol-specific
rewrites.

Emits constraints: the same row occurrences with normalized combinations.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.Normalize

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

universe u

private abbrev Assignment := R1CS.Assignment

def rows (source : List OwnedRow) : List OwnedRow :=
  source.map normalizeOwnedRow

def target
    {Observable : Type u}
    (one : ColumnId)
    (source : List OwnedRow)
    (observe : Assignment -> Observable) :=
  R1CS.system one (rows source) observe

theorem accepts_iff
    {Observable : Type u}
    (one : ColumnId)
    (source : List OwnedRow)
    (observe : Assignment -> Observable)
    (assignment : Assignment) :
    (target one source observe).Accepts assignment <->
      (R1CS.system one source observe).Accepts assignment := by
  unfold target R1CS.system
  change
    (assignment one = 1 /\
        Goldilocks.Satisfies (source.map normalizeOwnedRow) assignment) <->
      (assignment one = 1 /\ Goldilocks.Satisfies source assignment)
  exact and_congr_right
    (fun _ => satisfies_map_normalizeOwnedRow source assignment)

/-- Normalization is an identity-witness replacement. -/
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
    exact (accepts_iff one source observe assignment).mp accepted
  complete := by
    intro assignment accepted
    exact (accepts_iff one source observe assignment).mpr accepted
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

theorem rows_length (source : List OwnedRow) :
    (rows source).length = source.length := by
  simp [rows]

theorem row_ids (source : List OwnedRow) :
    (rows source).map (fun row => row.id) =
      source.map (fun row => row.id) := by
  simp [rows, normalizeOwnedRow]

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.Normalize
