import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.GroupedCommon

/-!
Contract: exact schedule selection with shared lifecycle rows and shared
phase-kind rows.

Owns one schedule-selector total, checked lifecycle and phase-kind selector
sums, both activation-link families, arm-local authority rows, soundness,
honest completeness, and exact semantic refinement.

Does not own emitted matrices, cursor encoding, a concrete schedule, component
row semantics, or recursive proof integration.

Emits constraints: no. An executable compiler must emit every field in
`LinkedAccepts`; labels or host-side maps are not authority.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon

/-- Executable semantic contract for a schedule over two shared row families.

`schedule` contains the small arm-specific authority rows, such as exact
before/after cursor equations. The large lifecycle and phase-kind families are
stored once and selected through checked group sums. -/
structure LinkedAccepts
    {armCount lifecycleCount phaseKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (weights : Fin armCount → F)
    (lifecycleWeights : Fin lifecycleCount → F)
    (phaseKindWeights : Fin phaseKindCount → F)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily) : Prop where
  total : SelectorTotal weights
  lifecycleEqualities : ∀ group,
    lifecycleWeights group = groupWeight lifecycleOf weights group
  phaseKindEqualities : ∀ kind,
    phaseKindWeights kind = groupWeight phaseKindOf weights kind
  commonGated : ∀ group,
    GatedRowsZero (lifecycleWeights group) (common group)
  phaseKindGated : ∀ kind,
    GatedRowsZero (phaseKindWeights kind) (phaseKinds kind)
  scheduleGated : ∀ arm,
    GatedRowsZero (weights arm) (schedule arm)
  lifecycleLinks : ∀ arm,
    weights arm * lifecycleWeights (lifecycleOf arm) = weights arm
  phaseKindLinks : ∀ arm,
    weights arm * phaseKindWeights (phaseKindOf arm) = weights arm

/-- Exact selected result of the scheduled composition. -/
def Selected
    {armCount lifecycleCount phaseKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily) : Prop :=
  ∃ arm,
    RowsZero (common (lifecycleOf arm)) ∧
      RowsZero (phaseKinds (phaseKindOf arm)) ∧
        RowsZero (schedule arm)

theorem linkedAccepts_sound
    (noZeroProducts : NoZeroProducts)
    {armCount lifecycleCount phaseKindCount : Nat}
    {lifecycleOf : Fin armCount → Fin lifecycleCount}
    {phaseKindOf : Fin armCount → Fin phaseKindCount}
    {weights : Fin armCount → F}
    {lifecycleWeights : Fin lifecycleCount → F}
    {phaseKindWeights : Fin phaseKindCount → F}
    {common : Fin lifecycleCount → ResidualFamily}
    {phaseKinds : Fin phaseKindCount → ResidualFamily}
    {schedule : Fin armCount → ResidualFamily}
    (accepted : LinkedAccepts lifecycleOf phaseKindOf weights
      lifecycleWeights phaseKindWeights common phaseKinds schedule) :
    Selected lifecycleOf phaseKindOf common phaseKinds schedule := by
  rcases nonzero_selector_of_total accepted.total with ⟨arm, active⟩
  have lifecycleActive : lifecycleWeights (lifecycleOf arm) ≠ 0 := by
    intro zero
    have linked := accepted.lifecycleLinks arm
    rw [zero, Fin.mul_zero] at linked
    exact active linked.symm
  have phaseKindActive : phaseKindWeights (phaseKindOf arm) ≠ 0 := by
    intro zero
    have linked := accepted.phaseKindLinks arm
    rw [zero, Fin.mul_zero] at linked
    exact active linked.symm
  exact ⟨arm,
    rowsZero_of_gated noZeroProducts lifecycleActive
      (accepted.commonGated (lifecycleOf arm)),
    rowsZero_of_gated noZeroProducts phaseKindActive
      (accepted.phaseKindGated (phaseKindOf arm)),
    rowsZero_of_gated noZeroProducts active
      (accepted.scheduleGated arm)⟩

theorem linkedAccepts_complete
    {armCount lifecycleCount phaseKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily)
    (selected : Fin armCount)
    (commonZero : RowsZero (common (lifecycleOf selected)))
    (phaseKindZero : RowsZero (phaseKinds (phaseKindOf selected)))
    (scheduleZero : RowsZero (schedule selected)) :
    LinkedAccepts lifecycleOf phaseKindOf (unitWeights selected)
      (fun group => groupWeight lifecycleOf (unitWeights selected) group)
      (fun kind => groupWeight phaseKindOf (unitWeights selected) kind)
      common phaseKinds schedule := by
  refine ⟨unitWeights_total selected, fun _ => rfl, fun _ => rfl,
    ?_, ?_, ?_, ?_, ?_⟩
  · intro group row
    by_cases same : group = lifecycleOf selected
    · subst group
      rw [groupWeight_unit_selected, Fin.one_mul]
      exact commonZero row
    · rw [groupWeight_unit_other lifecycleOf selected group same]
      exact Fin.zero_mul _
  · intro kind row
    by_cases same : kind = phaseKindOf selected
    · subst kind
      rw [groupWeight_unit_selected, Fin.one_mul]
      exact phaseKindZero row
    · rw [groupWeight_unit_other phaseKindOf selected kind same]
      exact Fin.zero_mul _
  · exact (Semantics.accepts_complete schedule selected scheduleZero).gated
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

theorem exists_linkedAccepts_iff_selected
    (noZeroProducts : NoZeroProducts)
    {armCount lifecycleCount phaseKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily) :
    (∃ weights lifecycleWeights phaseKindWeights,
      LinkedAccepts lifecycleOf phaseKindOf weights lifecycleWeights
        phaseKindWeights common phaseKinds schedule) ↔
      Selected lifecycleOf phaseKindOf common phaseKinds schedule := by
  constructor
  · rintro ⟨weights, lifecycleWeights, phaseKindWeights, accepted⟩
    exact linkedAccepts_sound noZeroProducts accepted
  · rintro ⟨selected, commonZero, phaseKindZero, scheduleZero⟩
    exact ⟨unitWeights selected,
      (fun group => groupWeight lifecycleOf (unitWeights selected) group),
      (fun kind => groupWeight phaseKindOf (unitWeights selected) kind),
      linkedAccepts_complete lifecycleOf phaseKindOf common phaseKinds
        schedule selected commonZero phaseKindZero scheduleZero⟩

/-- Independent refinement interface for one exact scheduled arm. -/
structure ExactRefinement
    {armCount lifecycleCount phaseKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily)
    (semantics : Fin armCount → Prop) : Prop where
  sound : ∀ arm,
    RowsZero (common (lifecycleOf arm)) →
    RowsZero (phaseKinds (phaseKindOf arm)) →
    RowsZero (schedule arm) → semantics arm
  complete : ∀ arm, semantics arm →
    RowsZero (common (lifecycleOf arm)) ∧
      RowsZero (phaseKinds (phaseKindOf arm)) ∧
        RowsZero (schedule arm)

theorem selected_iff_semantics
    {armCount lifecycleCount phaseKindCount : Nat}
    {lifecycleOf : Fin armCount → Fin lifecycleCount}
    {phaseKindOf : Fin armCount → Fin phaseKindCount}
    {common : Fin lifecycleCount → ResidualFamily}
    {phaseKinds : Fin phaseKindCount → ResidualFamily}
    {schedule : Fin armCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactRefinement lifecycleOf phaseKindOf common
      phaseKinds schedule semantics) :
    Selected lifecycleOf phaseKindOf common phaseKinds schedule ↔
      ∃ arm, semantics arm := by
  constructor
  · rintro ⟨arm, commonZero, phaseKindZero, scheduleZero⟩
    exact ⟨arm, refinement.sound arm commonZero phaseKindZero scheduleZero⟩
  · rintro ⟨arm, holds⟩
    exact ⟨arm, refinement.complete arm holds⟩

theorem exists_linkedAccepts_iff_semantics
    (noZeroProducts : NoZeroProducts)
    {armCount lifecycleCount phaseKindCount : Nat}
    {lifecycleOf : Fin armCount → Fin lifecycleCount}
    {phaseKindOf : Fin armCount → Fin phaseKindCount}
    {common : Fin lifecycleCount → ResidualFamily}
    {phaseKinds : Fin phaseKindCount → ResidualFamily}
    {schedule : Fin armCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactRefinement lifecycleOf phaseKindOf common
      phaseKinds schedule semantics) :
    (∃ weights lifecycleWeights phaseKindWeights,
      LinkedAccepts lifecycleOf phaseKindOf weights lifecycleWeights
        phaseKindWeights common phaseKinds schedule) ↔
      ∃ arm, semantics arm := by
  rw [exists_linkedAccepts_iff_selected noZeroProducts lifecycleOf phaseKindOf
    common phaseKinds schedule, selected_iff_semantics refinement]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped
