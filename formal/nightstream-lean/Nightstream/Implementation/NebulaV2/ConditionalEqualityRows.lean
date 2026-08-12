import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: equality rows enabled exactly when a phase bit is zero.

Assurance tier: implementation model.

Owns the R1CS gate `(left-right) * (1-phase) = 0`, sound extraction of all
equalities in phase zero, and completeness for canonical phase zero and one.

Does not own the Boolean row for the phase wire or the meaning of each pair.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ConditionalEqualityRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

def row (phaseColumn : Nat) (pair : Nat × Nat) : Row :=
  ⟨[(pair.1, 1), (pair.2, goldilocksP - 1)],
    [(0, 1), (phaseColumn, goldilocksP - 1)], []⟩

def rows (phaseColumn : Nat) (pairs : List (Nat × Nat)) : List Row :=
  pairs.map (row phaseColumn)

theorem rows_length (phaseColumn : Nat) (pairs : List (Nat × Nat)) :
    (rows phaseColumn pairs).length = pairs.length := by
  simp [rows]

private theorem equality_row_of_closed_gate
    {phaseColumn : Nat} {pair : Nat × Nat} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (holds : RowHolds assignment (row phaseColumn pair)) :
    RowHolds assignment (builderLinearRow pair.1 [(pair.2, 1)]) := by
  simpa [row, builderLinearRow, negateTerms, negCoeff, RowHolds, lcEval,
    one, phaseClosed, goldilocksP] using holds

theorem rows_sound_closed
    {phaseColumn : Nat} {pairs : List (Nat × Nat)}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (holds : Satisfies (rows phaseColumn pairs) assignment) :
    ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  intro pair member
  have gated := holds _ (List.mem_map.mpr ⟨pair, member, rfl⟩)
  have equalityRow := equality_row_of_closed_gate one phaseClosed gated
  have fieldEqual := builderLinearRow_sound canonical one pair.1
    [(pair.2, 1)] (by simp [CanonicalTerms]; decide) equalityRow
  simpa [lcEval, Nat.mod_eq_of_lt (canonical pair.2)] using fieldEqual

theorem rows_complete_closed
    {phaseColumn : Nat} {pairs : List (Nat × Nat)}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (equalities : ∀ pair ∈ pairs,
      assignment pair.1 = assignment pair.2) :
    Satisfies (rows phaseColumn pairs) assignment := by
  intro candidate member
  rcases List.mem_map.mp member with ⟨pair, pairMember, rfl⟩
  have equal := equalities pair pairMember
  have builder :
      RowHolds assignment (builderLinearRow pair.1 [(pair.2, 1)]) := by
    apply builderLinearRow_complete one pair.1 [(pair.2, 1)]
      (by simp [CanonicalTerms]; decide)
    simpa [lcEval, Nat.mod_eq_of_lt (canonical pair.2)] using equal
  simpa [row, builderLinearRow, negateTerms, negCoeff, RowHolds, lcEval,
    one, phaseClosed, goldilocksP] using builder

theorem rows_complete_active
    {phaseColumn : Nat} {pairs : List (Nat × Nat)}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseActive : assignment phaseColumn = 1) :
    Satisfies (rows phaseColumn pairs) assignment := by
  intro candidate member
  rcases List.mem_map.mp member with ⟨pair, _pairMember, rfl⟩
  simp [row, RowHolds, lcEval, one, phaseActive, goldilocksP,
    Nat.add_mod]

end Nightstream.Implementation.NebulaV2.ConditionalEqualityRows
