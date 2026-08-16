import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateSequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledLinkedOverlay

/-!
Contract: exact 400-arm semantic interface for the production base-plus-overlay
F-prime relation.

Owns the exact 136-kind claim-coordinate and PiRLC-family overlay map, the
joint meaning of lifecycle, phase, schedule, overlay, and private-link rows on
one before/after pair, and refinement of accepted rows to the
verifier-selected program step.

Does not own emitted matrices, concrete row-family equivalences, claim-row
source semantics, recursive proof integration, terminal verification, or Rust
matrix conformance.

Emits constraints: no. Each concrete row family must prove the equivalences
used by `exactRefinement`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlay

abbrev WorkArm :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.WorkArm

def workItem (arm : WorkArm) : WorkItem :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.workItem arm

def lifecycleCircuit (arm : WorkArm) : Fin 2 :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.lifecycleCircuit arm

def phaseKind (arm : WorkArm) : Fin 23 :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.phaseKind arm

/-- Embed one claim-coordinate kind in the combined overlay vocabulary. -/
def liftClaimOverlayKind (kind : Fin 26) : Fin 136 :=
  ⟨kind.val, by omega⟩

/-- Exact combined overlay kind selected by one verifier-owned work item.
Invalid family indices map to no-op; they do not occur in the production
program. -/
def combinedOverlayKindForWorkItem (item : WorkItem) : Fin 136 :=
  if item.phase = .piRlcFamily then
    if bound : item.index < 110 then
      ⟨26 + item.index, by omega⟩
    else
      0
  else
    liftClaimOverlayKind (overlayKindForWorkItem item)

def overlayKind (arm : WorkArm) : Fin 136 :=
  combinedOverlayKindForWorkItem (workItem arm)

theorem overlayKind_nonOverlay
    (arm : WorkArm)
    (notClaim : (workItem arm).phase ≠ .claimReplay)
    (notPiRlc : (workItem arm).phase ≠ .piRlcFamily) :
    overlayKind arm = 0 := by
  simp [overlayKind, combinedOverlayKindForWorkItem,
    overlayKindForWorkItem, notClaim, notPiRlc, liftClaimOverlayKind]

theorem overlayKind_claim
    (arm : WorkArm) (claim : (workItem arm).phase = .claimReplay)
    (bound : (workItem arm).index < claimChunkCount) :
    overlayKind arm =
      liftClaimOverlayKind
        (overlayKindAt ⟨(workItem arm).index, bound⟩) := by
  simp [overlayKind, combinedOverlayKindForWorkItem,
    overlayKindForWorkItem, claim, bound, liftClaimOverlayKind]

theorem overlayKind_piRlcFamily
    (arm : WorkArm) (family : (workItem arm).phase = .piRlcFamily)
    (bound : (workItem arm).index < 110) :
    overlayKind arm = ⟨26 + (workItem arm).index, by omega⟩ := by
  simp [overlayKind, combinedOverlayKindForWorkItem, family, bound]

/-- Complete meaning of one selected base-plus-overlay arm. All row families
refer to the same runtime values. -/
def OverlayArmSemantics {State : Type}
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (overlaySemantics : WorkArm → Runtime State → Runtime State → Prop)
    (before after : Runtime State) (arm : WorkArm) : Prop :=
  commonSemantics (lifecycleCircuit arm) before after ∧
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.PhaseAtArm
      phaseSemantics arm before after ∧
      overlaySemantics arm before after

/-- Complete meaning of one selected arm when the phase body and the selected
overlay form one joint local relation. This is the production interface for
split phase bodies such as PiRLC family parity bodies. -/
def JointArmSemantics {State : Type}
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (before after : Runtime State) (arm : WorkArm) : Prop :=
  commonSemantics (lifecycleCircuit arm) before after ∧
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.PhaseAtArm
      phaseSemantics arm before after

/-- Exact refinement for a split local relation. The phase body, selected
overlay body, and decoded private-field links jointly imply the phase
semantics. No one component must authorize the phase by itself. -/
theorem jointExactRefinement {State : Type}
    (commonRows : Fin 2 → ResidualFamily)
    (phaseKindRows : Fin 23 → ResidualFamily)
    (scheduleRows : WorkArm → ResidualFamily)
    (overlayRows fieldLinkRows : Fin 136 → ResidualFamily)
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (before after : Runtime State)
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.CursorAtArm
          arm before after)
    (localExact : ∀ arm,
      (RowsZero (phaseKindRows (phaseKind arm)) ∧
        RowsZero (overlayRows (overlayKind arm)) ∧
          RowsZero (fieldLinkRows (overlayKind arm))) ↔
        phaseSemantics (workItem arm) before.value after.value) :
    ExactRefinement lifecycleCircuit phaseKind overlayKind commonRows
      phaseKindRows scheduleRows overlayRows fieldLinkRows
      (JointArmSemantics commonSemantics phaseSemantics before after) := by
  constructor
  · intro arm commonZero phaseZero scheduleZero overlayZero fieldLinksZero
    have cursor := (scheduleExact arm).mp scheduleZero
    exact ⟨
      (commonExact _).mp commonZero,
      ⟨cursor.1, cursor.2,
        (localExact arm).mp ⟨phaseZero, overlayZero, fieldLinksZero⟩⟩⟩
  · intro arm semantics
    have localRows := (localExact arm).mpr semantics.2.2.2
    exact ⟨
      (commonExact _).mpr semantics.1,
      localRows.1,
      (scheduleExact arm).mpr ⟨semantics.2.1, semantics.2.2.1⟩,
      localRows.2.1,
      localRows.2.2⟩

/-- Selector acceptance is exactly one verifier-owned arm when each split
phase body and overlay pair has an exact joint refinement. -/
theorem exists_linkedAccepts_iff_jointArmSemantics
    (noZeroProducts : NoZeroProducts)
    {State : Type}
    (commonRows : Fin 2 → ResidualFamily)
    (phaseKindRows : Fin 23 → ResidualFamily)
    (scheduleRows : WorkArm → ResidualFamily)
    (overlayRows fieldLinkRows : Fin 136 → ResidualFamily)
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (before after : Runtime State)
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.CursorAtArm
          arm before after)
    (localExact : ∀ arm,
      (RowsZero (phaseKindRows (phaseKind arm)) ∧
        RowsZero (overlayRows (overlayKind arm)) ∧
          RowsZero (fieldLinkRows (overlayKind arm))) ↔
        phaseSemantics (workItem arm) before.value after.value) :
    (∃ weights lifecycleWeights phaseKindWeights overlayWeights,
      LinkedAccepts lifecycleCircuit phaseKind overlayKind weights
        lifecycleWeights phaseKindWeights overlayWeights commonRows
        phaseKindRows scheduleRows overlayRows fieldLinkRows) ↔
      ∃ arm, JointArmSemantics commonSemantics phaseSemantics
        before after arm := by
  exact exists_linkedAccepts_iff_semantics noZeroProducts
    (jointExactRefinement commonRows phaseKindRows scheduleRows overlayRows
      fieldLinkRows commonSemantics phaseSemantics before after commonExact
      scheduleExact localExact)

/-- Soundness interface for production split rows. This direction is enough
for verification: the selected phase body, overlay body, and field links must
jointly imply the verifier-owned phase relation. -/
theorem linkedAccepts_implies_step_of_joint_sound
    (noZeroProducts : NoZeroProducts)
    {State : Type}
    {commonRows : Fin 2 → ResidualFamily}
    {phaseKindRows : Fin 23 → ResidualFamily}
    {scheduleRows : WorkArm → ResidualFamily}
    {overlayRows fieldLinkRows : Fin 136 → ResidualFamily}
    {phaseSemantics : WorkItem → State → State → Prop}
    {before after : Runtime State}
    (scheduleSound : ∀ arm,
      RowsZero (scheduleRows arm) →
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.CursorAtArm
          arm before after)
    (localSound : ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) →
      RowsZero (overlayRows (overlayKind arm)) →
      RowsZero (fieldLinkRows (overlayKind arm)) →
        phaseSemantics (workItem arm) before.value after.value)
    {weights : WorkArm → F}
    {lifecycleWeights : Fin 2 → F}
    {phaseKindWeights : Fin 23 → F}
    {overlayWeights : Fin 136 → F}
    (accepted :
      LinkedAccepts lifecycleCircuit phaseKind overlayKind weights
        lifecycleWeights phaseKindWeights overlayWeights commonRows
        phaseKindRows scheduleRows overlayRows fieldLinkRows) :
    Step phaseSemantics productionConfig before after := by
  rcases linkedAccepts_sound noZeroProducts accepted with
    ⟨arm, _, phaseZero, scheduleZero, overlayZero, fieldLinksZero⟩
  have cursor := scheduleSound arm scheduleZero
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.phaseAtArm_to_step
      ⟨cursor.1, cursor.2,
        localSound arm phaseZero overlayZero fieldLinksZero⟩

/-- Exact five-family refinement used by the executable production composer.
The overlay equivalence includes both the selected overlay body and its
decoded private-field links. -/
theorem exactRefinement {State : Type}
    (commonRows : Fin 2 → ResidualFamily)
    (phaseKindRows : Fin 23 → ResidualFamily)
    (scheduleRows : WorkArm → ResidualFamily)
    (overlayRows fieldLinkRows : Fin 136 → ResidualFamily)
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (overlaySemantics : WorkArm → Runtime State → Runtime State → Prop)
    (before after : Runtime State)
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (phaseKindExact : ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) ↔
        phaseSemantics (workItem arm) before.value after.value)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.CursorAtArm
          arm before after)
    (overlayExact : ∀ arm,
      (RowsZero (overlayRows (overlayKind arm)) ∧
        RowsZero (fieldLinkRows (overlayKind arm))) ↔
          overlaySemantics arm before after) :
    ExactRefinement lifecycleCircuit phaseKind overlayKind commonRows
      phaseKindRows scheduleRows overlayRows fieldLinkRows
      (OverlayArmSemantics commonSemantics phaseSemantics overlaySemantics
        before after) := by
  constructor
  · intro arm commonZero phaseZero scheduleZero overlayZero fieldLinksZero
    have cursor := (scheduleExact arm).mp scheduleZero
    exact ⟨
      (commonExact _).mp commonZero,
      ⟨cursor.1, cursor.2, (phaseKindExact arm).mp phaseZero⟩,
      (overlayExact arm).mp ⟨overlayZero, fieldLinksZero⟩⟩
  · intro arm semantics
    have overlay := (overlayExact arm).mpr semantics.2.2
    exact ⟨
      (commonExact _).mpr semantics.1,
      (phaseKindExact arm).mpr semantics.2.1.2.2,
      (scheduleExact arm).mpr ⟨semantics.2.1.1, semantics.2.1.2.1⟩,
      overlay.1,
      overlay.2⟩

/-- Selector acceptance is exactly one verifier-owned production arm with its
overlay semantics. -/
theorem exists_linkedAccepts_iff_overlayArmSemantics
    (noZeroProducts : NoZeroProducts)
    {State : Type}
    (commonRows : Fin 2 → ResidualFamily)
    (phaseKindRows : Fin 23 → ResidualFamily)
    (scheduleRows : WorkArm → ResidualFamily)
    (overlayRows fieldLinkRows : Fin 136 → ResidualFamily)
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (overlaySemantics : WorkArm → Runtime State → Runtime State → Prop)
    (before after : Runtime State)
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (phaseKindExact : ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) ↔
        phaseSemantics (workItem arm) before.value after.value)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.CursorAtArm
          arm before after)
    (overlayExact : ∀ arm,
      (RowsZero (overlayRows (overlayKind arm)) ∧
        RowsZero (fieldLinkRows (overlayKind arm))) ↔
          overlaySemantics arm before after) :
    (∃ weights lifecycleWeights phaseKindWeights overlayWeights,
      LinkedAccepts lifecycleCircuit phaseKind overlayKind weights
        lifecycleWeights phaseKindWeights overlayWeights commonRows
        phaseKindRows scheduleRows overlayRows fieldLinkRows) ↔
      ∃ arm, OverlayArmSemantics commonSemantics phaseSemantics
        overlaySemantics before after arm := by
  exact exists_linkedAccepts_iff_semantics noZeroProducts
    (exactRefinement commonRows phaseKindRows scheduleRows overlayRows
      fieldLinkRows commonSemantics phaseSemantics overlaySemantics before after
      commonExact phaseKindExact scheduleExact overlayExact)

/-- Any accepted production base-plus-overlay relation performs the exact next
program step. The overlay cannot authorize a different schedule arm. -/
theorem linkedAccepts_implies_step
    (noZeroProducts : NoZeroProducts)
    {State : Type}
    {commonRows : Fin 2 → ResidualFamily}
    {phaseKindRows : Fin 23 → ResidualFamily}
    {scheduleRows : WorkArm → ResidualFamily}
    {overlayRows fieldLinkRows : Fin 136 → ResidualFamily}
    {commonSemantics : Fin 2 → Runtime State → Runtime State → Prop}
    {phaseSemantics : WorkItem → State → State → Prop}
    {overlaySemantics : WorkArm → Runtime State → Runtime State → Prop}
    {before after : Runtime State}
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (phaseKindExact : ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) ↔
        phaseSemantics (workItem arm) before.value after.value)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.CursorAtArm
          arm before after)
    (overlayExact : ∀ arm,
      (RowsZero (overlayRows (overlayKind arm)) ∧
        RowsZero (fieldLinkRows (overlayKind arm))) ↔
          overlaySemantics arm before after)
    {weights : WorkArm → F}
    {lifecycleWeights : Fin 2 → F}
    {phaseKindWeights : Fin 23 → F}
    {overlayWeights : Fin 136 → F}
    (accepted :
      LinkedAccepts lifecycleCircuit phaseKind overlayKind weights
        lifecycleWeights phaseKindWeights overlayWeights commonRows
        phaseKindRows scheduleRows overlayRows fieldLinkRows) :
    Step phaseSemantics productionConfig before after := by
  have selected := linkedAccepts_sound noZeroProducts accepted
  have semantics :=
    (selected_iff_semantics
      (exactRefinement commonRows phaseKindRows scheduleRows overlayRows
        fieldLinkRows commonSemantics phaseSemantics overlaySemantics before
        after commonExact phaseKindExact scheduleExact overlayExact)).mp selected
  rcases semantics with ⟨arm, armSemantics⟩
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.phaseAtArm_to_step
    armSemantics.2.1

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation
