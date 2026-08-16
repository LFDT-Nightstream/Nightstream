import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyPhysicalOverlayRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateSequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedOverlayRelation

/-!
Contract: production source-semantics adapter for the 197-kind physical
overlay relation.

Owns the joint local soundness target for claim-coordinate and PiRLC-family
overlays. Claim source rows imply one exact additive commitment step. A shared
PiRLC parity body, one physical family overlay, and their exact field links
jointly imply one `FamilyPhaseRelation`.

Does not own normalized low-norm row decoding, Rust matrix conformance, other
phase kinds, recursive lifecycle integration, or terminal verification.

Emits constraints: no. `SourceSoundness` states the exact bridge that the
normalized production artifact must supply.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhysicalOverlayRelation

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence
open Nightstream.SuperNeo.Concrete

abbrev WorkArm :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.WorkArm

def workItem (arm : WorkArm) : WorkItem :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.workItem arm

def phaseKind (arm : WorkArm) : Fin 23 :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.phaseKind arm

def overlayKind (arm : WorkArm) : Fin 197 :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.overlayKind arm

/-- Algebraic claim-coordinate transition selected by one bounded claim
index. Invalid indices are false because they have no bound witness. -/
def ClaimPhaseRelation
    (production : ProductionSetup) (fields : Fields)
    (index : Nat) (before after : Accumulator) : Prop :=
  ∃ bound : index < claimChunkCount,
    StepAt production fields ⟨index, bound⟩ before after

/-- Algebraic PiRLC transition selected by one family ordinal. The family is
part of the verifier-owned semantic witness, not a prover-selected selector. -/
def PiRlcPhaseRelation
    (setup : InputBindingSetup) (index : Nat)
    (before after : FamilyState) : Prop :=
  ∃ family,
    ProductPiRlcAlgebraRows.familyOrdinal family = index ∧
      ∃ inputs output,
        FamilyPhaseRelation setup before after family inputs output

/-- Production phase semantics for the two overlay-backed phase classes.
Every other phase keeps its separately supplied semantic relation. -/
def PhaseSemantics {State : Type}
    (production : ProductionSetup) (fields : Fields)
    (setup : InputBindingSetup)
    (claimAccumulator : State → Accumulator)
    (piRlcState : State → FamilyState)
    (other : WorkItem → State → State → Prop)
    (item : WorkItem) (before after : State) : Prop :=
  if item.phase = .claimReplay then
    ClaimPhaseRelation production fields item.index
      (claimAccumulator before) (claimAccumulator after)
  else if item.phase = .piRlcFamily then
    PiRlcPhaseRelation setup item.index
      (piRlcState before) (piRlcState after)
  else
    other item before after

/-- Exact source evidence required from the normalized rows of each selected
arm. This structure prevents either a phase body or an overlay from acting as
independent authority. -/
structure SourceSoundness {State : Type}
    (production : ProductionSetup) (fields : Fields)
    (setup : InputBindingSetup)
    (claimAccumulator : State → Accumulator)
    (piRlcState : State → FamilyState)
    (other : WorkItem → State → State → Prop)
    (phaseKindRows : Fin 23 → ResidualFamily)
    (overlayRows fieldLinkRows : Fin 197 → ResidualFamily)
    (before after : Runtime State) : Prop where
  claim : ∀ arm,
    (workItem arm).phase = .claimReplay →
    RowsZero (phaseKindRows (phaseKind arm)) →
    RowsZero (overlayRows (overlayKind arm)) →
    RowsZero (fieldLinkRows (overlayKind arm)) →
      ∃ bound : (workItem arm).index < claimChunkCount,
        PhaseRowsAt production fields ⟨(workItem arm).index, bound⟩
          (claimAccumulator before.value) (claimAccumulator after.value)
  piRlc : ∀ arm,
    (workItem arm).phase = .piRlcFamily →
    RowsZero (phaseKindRows (phaseKind arm)) →
    RowsZero (overlayRows (overlayKind arm)) →
    RowsZero (fieldLinkRows (overlayKind arm)) →
      ∃ family,
        ProductPiRlcAlgebraRows.familyOrdinal family = (workItem arm).index ∧
          Nonempty (AcceptedRows setup family
            (piRlcState before.value) (piRlcState after.value))
  other : ∀ arm,
    (workItem arm).phase ≠ .claimReplay →
    (workItem arm).phase ≠ .piRlcFamily →
    RowsZero (phaseKindRows (phaseKind arm)) →
    RowsZero (overlayRows (overlayKind arm)) →
    RowsZero (fieldLinkRows (overlayKind arm)) →
      other (workItem arm) before.value after.value

namespace SourceSoundness

/-- The physical source evidence supplies the joint local-soundness premise
used by the 197-kind scheduled selector relation. -/
theorem localSound {State : Type}
    {production : ProductionSetup} {fields : Fields}
    {setup : InputBindingSetup}
    {claimAccumulator : State → Accumulator}
    {piRlcState : State → FamilyState}
    {other : WorkItem → State → State → Prop}
    {phaseKindRows : Fin 23 → ResidualFamily}
    {overlayRows fieldLinkRows : Fin 197 → ResidualFamily}
    {before after : Runtime State}
    (source : SourceSoundness production fields setup claimAccumulator
      piRlcState other phaseKindRows overlayRows fieldLinkRows before after) :
    ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) →
      RowsZero (overlayRows (overlayKind arm)) →
      RowsZero (fieldLinkRows (overlayKind arm)) →
        PhaseSemantics production fields setup claimAccumulator piRlcState
          other (workItem arm) before.value after.value := by
  intro arm phaseZero overlayZero fieldLinksZero
  by_cases claim : (workItem arm).phase = .claimReplay
  · rw [PhaseSemantics, if_pos claim]
    rcases source.claim arm claim phaseZero overlayZero fieldLinksZero with
      ⟨bound, accepted⟩
    exact ⟨bound, accepted.step⟩
  · by_cases piRlc : (workItem arm).phase = .piRlcFamily
    · rw [PhaseSemantics, if_neg claim, if_pos piRlc]
      rcases source.piRlc arm piRlc phaseZero overlayZero fieldLinksZero with
        ⟨family, ordinal, ⟨accepted⟩⟩
      exact ⟨family, ordinal, accepted.sound⟩
    · rw [PhaseSemantics, if_neg claim, if_neg piRlc]
      exact source.other arm claim piRlc phaseZero overlayZero fieldLinksZero

end SourceSoundness

/-- Any accepted scheduled physical relation performs one verifier-owned
source step for the claim and PiRLC overlay classes and the supplied relation
for every other phase. -/
theorem linkedAccepts_implies_step
    (noZeroProducts : NoZeroProducts)
    {State : Type}
    {production : ProductionSetup} {fields : Fields}
    {setup : InputBindingSetup}
    {claimAccumulator : State → Accumulator}
    {piRlcState : State → FamilyState}
    {other : WorkItem → State → State → Prop}
    {commonRows : Fin 2 → ResidualFamily}
    {phaseKindRows : Fin 23 → ResidualFamily}
    {scheduleRows : WorkArm → ResidualFamily}
    {overlayRows fieldLinkRows : Fin 197 → ResidualFamily}
    {before after : Runtime State}
    (source : SourceSoundness production fields setup claimAccumulator
      piRlcState other phaseKindRows overlayRows fieldLinkRows before after)
    (scheduleSound : ∀ arm,
      RowsZero (scheduleRows arm) →
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.CursorAtArm
          arm before after)
    {weights : WorkArm → F}
    {lifecycleWeights : Fin 2 → F}
    {phaseKindWeights : Fin 23 → F}
    {overlayWeights : Fin 197 → F}
    (accepted :
      LinkedAccepts
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.lifecycleCircuit
        phaseKind overlayKind weights lifecycleWeights phaseKindWeights
        overlayWeights commonRows phaseKindRows scheduleRows overlayRows
        fieldLinkRows) :
    Step (PhaseSemantics production fields setup claimAccumulator piRlcState
      other) productionConfig before after := by
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.linkedAccepts_implies_step_of_joint_sound
      noZeroProducts scheduleSound source.localSound accepted

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhysicalOverlayRelation
