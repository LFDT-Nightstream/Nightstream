import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledGrouped

/-!
Contract: one schedule-selected private overlay linked to the exact phase
assignment.

Owns the overlay-kind selector sums, overlay activation links, gated overlay
rows, gated private-field link rows, soundness, honest completeness, and the
exact semantic refinement interface.

Does not own emitted matrices, a concrete schedule, radix decoding, component
row semantics, or recursive proof integration.

Emits constraints: no. An executable compiler must emit every field in
`LinkedAccepts`; host-side maps and matching witness values are not authority.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlay

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon

/-- Executable contract for a scheduled relation plus one private overlay.

`fieldLinks` contains the exact decoded-field equality residuals for each
overlay kind. The same checked overlay selector gates both these residuals and
the corresponding overlay body. -/
structure LinkedAccepts
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (overlayKindOf : Fin armCount → Fin overlayKindCount)
    (weights : Fin armCount → F)
    (lifecycleWeights : Fin lifecycleCount → F)
    (phaseKindWeights : Fin phaseKindCount → F)
    (overlayWeights : Fin overlayKindCount → F)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily)
    (overlays fieldLinks : Fin overlayKindCount → ResidualFamily) : Prop where
  scheduled : ScheduledGrouped.LinkedAccepts lifecycleOf phaseKindOf weights
    lifecycleWeights phaseKindWeights common phaseKinds schedule
  overlayEqualities : ∀ kind,
    overlayWeights kind = groupWeight overlayKindOf weights kind
  overlayGated : ∀ kind,
    GatedRowsZero (overlayWeights kind) (overlays kind)
  overlayLinks : ∀ arm,
    weights arm * overlayWeights (overlayKindOf arm) = weights arm
  fieldLinksGated : ∀ kind,
    GatedRowsZero (overlayWeights kind) (fieldLinks kind)

/-- Exact selected result of the scheduled base-plus-overlay composition. -/
def Selected
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (overlayKindOf : Fin armCount → Fin overlayKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily)
    (overlays fieldLinks : Fin overlayKindCount → ResidualFamily) : Prop :=
  ∃ arm,
    RowsZero (common (lifecycleOf arm)) ∧
      RowsZero (phaseKinds (phaseKindOf arm)) ∧
        RowsZero (schedule arm) ∧
          RowsZero (overlays (overlayKindOf arm)) ∧
            RowsZero (fieldLinks (overlayKindOf arm))

theorem linkedAccepts_sound
    (noZeroProducts : NoZeroProducts)
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    {lifecycleOf : Fin armCount → Fin lifecycleCount}
    {phaseKindOf : Fin armCount → Fin phaseKindCount}
    {overlayKindOf : Fin armCount → Fin overlayKindCount}
    {weights : Fin armCount → F}
    {lifecycleWeights : Fin lifecycleCount → F}
    {phaseKindWeights : Fin phaseKindCount → F}
    {overlayWeights : Fin overlayKindCount → F}
    {common : Fin lifecycleCount → ResidualFamily}
    {phaseKinds : Fin phaseKindCount → ResidualFamily}
    {schedule : Fin armCount → ResidualFamily}
    {overlays fieldLinks : Fin overlayKindCount → ResidualFamily}
    (accepted : LinkedAccepts lifecycleOf phaseKindOf overlayKindOf weights
      lifecycleWeights phaseKindWeights overlayWeights common phaseKinds
      schedule overlays fieldLinks) :
    Selected lifecycleOf phaseKindOf overlayKindOf common phaseKinds schedule
      overlays fieldLinks := by
  rcases nonzero_selector_of_total accepted.scheduled.total with ⟨arm, active⟩
  have lifecycleActive : lifecycleWeights (lifecycleOf arm) ≠ 0 := by
    intro zero
    have linked := accepted.scheduled.lifecycleLinks arm
    rw [zero, Fin.mul_zero] at linked
    exact active linked.symm
  have phaseKindActive : phaseKindWeights (phaseKindOf arm) ≠ 0 := by
    intro zero
    have linked := accepted.scheduled.phaseKindLinks arm
    rw [zero, Fin.mul_zero] at linked
    exact active linked.symm
  have overlayActive : overlayWeights (overlayKindOf arm) ≠ 0 := by
    intro zero
    have linked := accepted.overlayLinks arm
    rw [zero, Fin.mul_zero] at linked
    exact active linked.symm
  exact ⟨arm,
    rowsZero_of_gated noZeroProducts lifecycleActive
      (accepted.scheduled.commonGated (lifecycleOf arm)),
    rowsZero_of_gated noZeroProducts phaseKindActive
      (accepted.scheduled.phaseKindGated (phaseKindOf arm)),
    rowsZero_of_gated noZeroProducts active
      (accepted.scheduled.scheduleGated arm),
    rowsZero_of_gated noZeroProducts overlayActive
      (accepted.overlayGated (overlayKindOf arm)),
    rowsZero_of_gated noZeroProducts overlayActive
      (accepted.fieldLinksGated (overlayKindOf arm))⟩

theorem linkedAccepts_complete
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (overlayKindOf : Fin armCount → Fin overlayKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily)
    (overlays fieldLinks : Fin overlayKindCount → ResidualFamily)
    (selected : Fin armCount)
    (commonZero : RowsZero (common (lifecycleOf selected)))
    (phaseKindZero : RowsZero (phaseKinds (phaseKindOf selected)))
    (scheduleZero : RowsZero (schedule selected))
    (overlayZero : RowsZero (overlays (overlayKindOf selected)))
    (fieldLinksZero : RowsZero (fieldLinks (overlayKindOf selected))) :
    LinkedAccepts lifecycleOf phaseKindOf overlayKindOf
      (unitWeights selected)
      (fun group => groupWeight lifecycleOf (unitWeights selected) group)
      (fun kind => groupWeight phaseKindOf (unitWeights selected) kind)
      (fun kind => groupWeight overlayKindOf (unitWeights selected) kind)
      common phaseKinds schedule overlays fieldLinks := by
  refine {
    scheduled := ScheduledGrouped.linkedAccepts_complete lifecycleOf
      phaseKindOf common phaseKinds schedule selected commonZero phaseKindZero
        scheduleZero
    overlayEqualities := fun _ => rfl
    overlayGated := ?_
    overlayLinks := ?_
    fieldLinksGated := ?_
  }
  · intro kind row
    by_cases same : kind = overlayKindOf selected
    · subst kind
      rw [groupWeight_unit_selected, Fin.one_mul]
      exact overlayZero row
    · rw [groupWeight_unit_other overlayKindOf selected kind same]
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
  · intro kind row
    by_cases same : kind = overlayKindOf selected
    · subst kind
      rw [groupWeight_unit_selected, Fin.one_mul]
      exact fieldLinksZero row
    · rw [groupWeight_unit_other overlayKindOf selected kind same]
      exact Fin.zero_mul _

theorem exists_linkedAccepts_iff_selected
    (noZeroProducts : NoZeroProducts)
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (overlayKindOf : Fin armCount → Fin overlayKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily)
    (overlays fieldLinks : Fin overlayKindCount → ResidualFamily) :
    (∃ weights lifecycleWeights phaseKindWeights overlayWeights,
      LinkedAccepts lifecycleOf phaseKindOf overlayKindOf weights
        lifecycleWeights phaseKindWeights overlayWeights common phaseKinds
        schedule overlays fieldLinks) ↔
      Selected lifecycleOf phaseKindOf overlayKindOf common phaseKinds schedule
        overlays fieldLinks := by
  constructor
  · rintro ⟨weights, lifecycleWeights, phaseKindWeights, overlayWeights,
      accepted⟩
    exact linkedAccepts_sound noZeroProducts accepted
  · rintro ⟨selected, commonZero, phaseKindZero, scheduleZero, overlayZero,
      fieldLinksZero⟩
    exact ⟨unitWeights selected,
      (fun group => groupWeight lifecycleOf (unitWeights selected) group),
      (fun kind => groupWeight phaseKindOf (unitWeights selected) kind),
      (fun kind => groupWeight overlayKindOf (unitWeights selected) kind),
      linkedAccepts_complete lifecycleOf phaseKindOf overlayKindOf common
        phaseKinds schedule overlays fieldLinks selected commonZero
        phaseKindZero scheduleZero overlayZero fieldLinksZero⟩

/-- Independent refinement interface for one exact linked schedule arm. -/
structure ExactRefinement
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    (lifecycleOf : Fin armCount → Fin lifecycleCount)
    (phaseKindOf : Fin armCount → Fin phaseKindCount)
    (overlayKindOf : Fin armCount → Fin overlayKindCount)
    (common : Fin lifecycleCount → ResidualFamily)
    (phaseKinds : Fin phaseKindCount → ResidualFamily)
    (schedule : Fin armCount → ResidualFamily)
    (overlays fieldLinks : Fin overlayKindCount → ResidualFamily)
    (semantics : Fin armCount → Prop) : Prop where
  sound : ∀ arm,
    RowsZero (common (lifecycleOf arm)) →
    RowsZero (phaseKinds (phaseKindOf arm)) →
    RowsZero (schedule arm) →
    RowsZero (overlays (overlayKindOf arm)) →
    RowsZero (fieldLinks (overlayKindOf arm)) → semantics arm
  complete : ∀ arm, semantics arm →
    RowsZero (common (lifecycleOf arm)) ∧
      RowsZero (phaseKinds (phaseKindOf arm)) ∧
        RowsZero (schedule arm) ∧
          RowsZero (overlays (overlayKindOf arm)) ∧
            RowsZero (fieldLinks (overlayKindOf arm))

theorem selected_iff_semantics
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    {lifecycleOf : Fin armCount → Fin lifecycleCount}
    {phaseKindOf : Fin armCount → Fin phaseKindCount}
    {overlayKindOf : Fin armCount → Fin overlayKindCount}
    {common : Fin lifecycleCount → ResidualFamily}
    {phaseKinds : Fin phaseKindCount → ResidualFamily}
    {schedule : Fin armCount → ResidualFamily}
    {overlays fieldLinks : Fin overlayKindCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactRefinement lifecycleOf phaseKindOf overlayKindOf common
      phaseKinds schedule overlays fieldLinks semantics) :
    Selected lifecycleOf phaseKindOf overlayKindOf common phaseKinds schedule
      overlays fieldLinks ↔ ∃ arm, semantics arm := by
  constructor
  · rintro ⟨arm, commonZero, phaseKindZero, scheduleZero, overlayZero,
      fieldLinksZero⟩
    exact ⟨arm, refinement.sound arm commonZero phaseKindZero scheduleZero
      overlayZero fieldLinksZero⟩
  · rintro ⟨arm, holds⟩
    exact ⟨arm, refinement.complete arm holds⟩

theorem exists_linkedAccepts_iff_semantics
    (noZeroProducts : NoZeroProducts)
    {armCount lifecycleCount phaseKindCount overlayKindCount : Nat}
    {lifecycleOf : Fin armCount → Fin lifecycleCount}
    {phaseKindOf : Fin armCount → Fin phaseKindCount}
    {overlayKindOf : Fin armCount → Fin overlayKindCount}
    {common : Fin lifecycleCount → ResidualFamily}
    {phaseKinds : Fin phaseKindCount → ResidualFamily}
    {schedule : Fin armCount → ResidualFamily}
    {overlays fieldLinks : Fin overlayKindCount → ResidualFamily}
    {semantics : Fin armCount → Prop}
    (refinement : ExactRefinement lifecycleOf phaseKindOf overlayKindOf common
      phaseKinds schedule overlays fieldLinks semantics) :
    (∃ weights lifecycleWeights phaseKindWeights overlayWeights,
      LinkedAccepts lifecycleOf phaseKindOf overlayKindOf weights
        lifecycleWeights phaseKindWeights overlayWeights common phaseKinds
        schedule overlays fieldLinks) ↔ ∃ arm, semantics arm := by
  rw [exists_linkedAccepts_iff_selected noZeroProducts lifecycleOf phaseKindOf
    overlayKindOf common phaseKinds schedule overlays fieldLinks,
    selected_iff_semantics refinement]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlay
