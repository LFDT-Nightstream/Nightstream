import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics

/-!
Contract: selector composition with one shared residual family per lifecycle
group and one phase-local residual family per selectable arm.

Owns: group-weight sums, the one link equation per phase, soundness of sharing
common rows once, honest completeness, and the exact existential
characterization of the selected common-plus-phase relation.

Does not own: emitted matrix coefficients, group or phase ordering, assignment
layouts, any concrete F' row family, or proof that two source programs have the
same common rows.

Emits constraints: no. An executable compiler must emit the global selector
total, every phase gate, every group-common gate, and one
`phaseWeight * groupWeight = phaseWeight` link per phase.

Authority boundary: the link equations are checked obligations. A compiler
label that says rows are shared cannot replace them or the matrix-action proof.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics

universe uRow

/-- Sum of all phase weights assigned to one lifecycle group. -/
def groupWeight {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (weights : Fin armCount → F) (group : Fin groupCount) : F :=
  selectorSum fun arm =>
    if groupOf arm = group then weights arm else 0

private theorem selectorSum_zero_of_pointwise_zero
    {armCount : Nat} {weights : Fin armCount → F}
    (everyZero : ∀ arm, weights arm = 0) :
    selectorSum weights = 0 := by
  induction armCount with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [selectorSum]
      rw [everyZero 0, Fin.zero_add]
      apply inductionHypothesis
      intro arm
      exact everyZero arm.succ

theorem groupWeight_unit_selected
    {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (selected : Fin armCount) :
    groupWeight groupOf (unitWeights selected) (groupOf selected) = 1 := by
  have sameWeights :
      (fun arm =>
        if groupOf arm = groupOf selected then
          unitWeights selected arm else 0) =
        unitWeights selected := by
    funext arm
    by_cases same : arm = selected
    · subst arm
      simp [unitWeights]
    · simp [unitWeights, same]
  unfold groupWeight
  rw [sameWeights]
  exact unitWeights_total selected

theorem groupWeight_unit_other
    {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (selected : Fin armCount) (group : Fin groupCount)
    (different : group ≠ groupOf selected) :
    groupWeight groupOf (unitWeights selected) group = 0 := by
  unfold groupWeight
  apply selectorSum_zero_of_pointwise_zero
  intro arm
  by_cases same : arm = selected
  · subst arm
    rw [if_neg (Ne.symm different)]
  · unfold unitWeights
    rw [if_neg same]
    split <;> rfl

/-- Acceptance contract for a shared-arena phase union.

`commonLinks` prevents an active phase from being paired with a zero group
weight. It is strictly weaker than selector Booleanity and is enough for the
soundness theorem below. -/
structure Accepts {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (weights : Fin armCount → F)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily) : Prop where
  total : SelectorTotal weights
  phaseGated : ∀ arm, GatedRowsZero (weights arm) (phases arm)
  commonGated : ∀ group,
    GatedRowsZero (groupWeight groupOf weights group) (common group)
  commonLinks : ∀ arm,
    weights arm * groupWeight groupOf weights (groupOf arm) = weights arm

/-- Executable form with one stored selector coordinate per lifecycle group.

The linear `groupEqualities` rows make these coordinates equal to the sums
used by `Accepts`. The product `commonLinks` rows then retain soundness without
assuming that either selector family is Boolean. -/
structure LinkedAccepts {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (weights : Fin armCount → F)
    (groupWeights : Fin groupCount → F)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily) : Prop where
  total : SelectorTotal weights
  groupEqualities : ∀ group,
    groupWeights group = groupWeight groupOf weights group
  phaseGated : ∀ arm, GatedRowsZero (weights arm) (phases arm)
  commonGated : ∀ group,
    GatedRowsZero (groupWeights group) (common group)
  commonLinks : ∀ arm,
    weights arm * groupWeights (groupOf arm) = weights arm

theorem LinkedAccepts.toAccepts
    {armCount groupCount : Nat}
    {groupOf : Fin armCount → Fin groupCount}
    {weights : Fin armCount → F}
    {groupWeights : Fin groupCount → F}
    {common : Fin groupCount → ResidualFamily}
    {phases : Fin armCount → ResidualFamily}
    (accepted : LinkedAccepts groupOf weights groupWeights common phases) :
    Accepts groupOf weights common phases := by
  refine ⟨accepted.total, accepted.phaseGated, ?_, ?_⟩
  · intro group row
    rw [← accepted.groupEqualities group]
    exact accepted.commonGated group row
  · intro arm
    rw [← accepted.groupEqualities (groupOf arm)]
    exact accepted.commonLinks arm

/-- Semantic result of the grouped composition: one selected phase and its
shared lifecycle rows both vanish. -/
def Selected {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily) : Prop :=
  ∃ arm, RowsZero (common (groupOf arm)) ∧ RowsZero (phases arm)

theorem accepts_sound
    (noZeroProducts : NoZeroProducts)
    {armCount groupCount : Nat}
    {groupOf : Fin armCount → Fin groupCount}
    {weights : Fin armCount → F}
    {common : Fin groupCount → ResidualFamily}
    {phases : Fin armCount → ResidualFamily}
    (accepted : Accepts groupOf weights common phases) :
    Selected groupOf common phases := by
  rcases nonzero_selector_of_total accepted.total with ⟨arm, active⟩
  have phaseZero : RowsZero (phases arm) :=
    rowsZero_of_gated noZeroProducts active (accepted.phaseGated arm)
  have groupNonzero :
      groupWeight groupOf weights (groupOf arm) ≠ 0 := by
    intro groupZero
    have linked := accepted.commonLinks arm
    rw [groupZero, Fin.mul_zero] at linked
    exact active linked.symm
  have commonZero : RowsZero (common (groupOf arm)) :=
    rowsZero_of_gated noZeroProducts groupNonzero
      (accepted.commonGated (groupOf arm))
  exact ⟨arm, commonZero, phaseZero⟩

/-- Honest completeness uses one unit phase weight. The selected group's
common rows are active; every other common and phase row has a zero gate. -/
theorem accepts_complete
    {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily)
    (selected : Fin armCount)
    (commonZero : RowsZero (common (groupOf selected)))
    (phaseZero : RowsZero (phases selected)) :
    Accepts groupOf (unitWeights selected) common phases := by
  refine ⟨unitWeights_total selected, ?_, ?_, ?_⟩
  · exact (Semantics.accepts_complete phases selected phaseZero).gated
  · intro group row
    by_cases same : group = groupOf selected
    · subst group
      rw [groupWeight_unit_selected, Fin.one_mul]
      exact commonZero row
    · rw [groupWeight_unit_other groupOf selected group same]
      exact Fin.zero_mul _
  · intro arm
    by_cases same : arm = selected
    · subst arm
      have selectedWeight : unitWeights selected selected = 1 := by
        unfold unitWeights
        rw [if_pos rfl]
      rw [selectedWeight, groupWeight_unit_selected, Fin.one_mul]
    · have inactiveWeight : unitWeights selected arm = 0 := by
        unfold unitWeights
        rw [if_neg same]
      rw [inactiveWeight, Fin.zero_mul]

theorem linkedAccepts_sound
    (noZeroProducts : NoZeroProducts)
    {armCount groupCount : Nat}
    {groupOf : Fin armCount → Fin groupCount}
    {weights : Fin armCount → F}
    {groupWeights : Fin groupCount → F}
    {common : Fin groupCount → ResidualFamily}
    {phases : Fin armCount → ResidualFamily}
    (accepted : LinkedAccepts groupOf weights groupWeights common phases) :
    Selected groupOf common phases :=
  accepts_sound noZeroProducts accepted.toAccepts

theorem linkedAccepts_complete
    {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily)
    (selected : Fin armCount)
    (commonZero : RowsZero (common (groupOf selected)))
    (phaseZero : RowsZero (phases selected)) :
    LinkedAccepts groupOf (unitWeights selected)
      (fun group => groupWeight groupOf (unitWeights selected) group)
      common phases := by
  let accepted :=
    accepts_complete groupOf common phases selected commonZero phaseZero
  exact {
    total := accepted.total
    groupEqualities := fun _ => rfl
    phaseGated := accepted.phaseGated
    commonGated := accepted.commonGated
    commonLinks := accepted.commonLinks
  }

/-- Exact selector contract for sharing common rows once. -/
theorem exists_accepts_iff_selected
    (noZeroProducts : NoZeroProducts)
    {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily) :
    (∃ weights, Accepts groupOf weights common phases) ↔
      Selected groupOf common phases := by
  constructor
  · rintro ⟨weights, accepted⟩
    exact accepts_sound noZeroProducts accepted
  · rintro ⟨selected, commonZero, phaseZero⟩
    exact ⟨unitWeights selected,
      accepts_complete groupOf common phases selected commonZero phaseZero⟩

theorem exists_linkedAccepts_iff_selected
    (noZeroProducts : NoZeroProducts)
    {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily) :
    (∃ weights groupWeights,
      LinkedAccepts groupOf weights groupWeights common phases) ↔
      Selected groupOf common phases := by
  constructor
  · rintro ⟨weights, groupWeights, accepted⟩
    exact linkedAccepts_sound noZeroProducts accepted
  · rintro ⟨selected, commonZero, phaseZero⟩
    exact ⟨unitWeights selected,
      (fun group => groupWeight groupOf (unitWeights selected) group),
      linkedAccepts_complete groupOf common phases selected commonZero
        phaseZero⟩

/-- Independent refinement interface for concrete lifecycle-plus-phase
semantics. -/
structure ExactRefinement {armCount groupCount : Nat}
    (groupOf : Fin armCount → Fin groupCount)
    (common : Fin groupCount → ResidualFamily)
    (phases : Fin armCount → ResidualFamily)
    (semantics : Fin armCount → Prop) : Prop where
  sound : ∀ arm,
    RowsZero (common (groupOf arm)) → RowsZero (phases arm) →
      semantics arm
  complete : ∀ arm, semantics arm →
    RowsZero (common (groupOf arm)) ∧ RowsZero (phases arm)

theorem selected_iff_semantics
    {armCount groupCount : Nat}
    {groupOf : Fin armCount → Fin groupCount}
    {common : Fin groupCount → ResidualFamily}
    {phases : Fin armCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactRefinement groupOf common phases semantics) :
    Selected groupOf common phases ↔ ∃ arm, semantics arm := by
  constructor
  · rintro ⟨arm, commonZero, phaseZero⟩
    exact ⟨arm, refinement.sound arm commonZero phaseZero⟩
  · rintro ⟨arm, holds⟩
    exact ⟨arm, refinement.complete arm holds⟩

theorem exists_accepts_iff_semantics
    (noZeroProducts : NoZeroProducts)
    {armCount groupCount : Nat}
    {groupOf : Fin armCount → Fin groupCount}
    {common : Fin groupCount → ResidualFamily}
    {phases : Fin armCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactRefinement groupOf common phases semantics) :
    (∃ weights, Accepts groupOf weights common phases) ↔
      ∃ arm, semantics arm := by
  rw [exists_accepts_iff_selected noZeroProducts groupOf common phases,
    selected_iff_semantics refinement]

theorem exists_linkedAccepts_iff_semantics
    (noZeroProducts : NoZeroProducts)
    {armCount groupCount : Nat}
    {groupOf : Fin armCount → Fin groupCount}
    {common : Fin groupCount → ResidualFamily}
    {phases : Fin armCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactRefinement groupOf common phases semantics) :
    (∃ weights groupWeights,
      LinkedAccepts groupOf weights groupWeights common phases) ↔
      ∃ arm, semantics arm := by
  rw [exists_linkedAccepts_iff_selected noZeroProducts groupOf common phases,
    selected_iff_semantics refinement]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon
