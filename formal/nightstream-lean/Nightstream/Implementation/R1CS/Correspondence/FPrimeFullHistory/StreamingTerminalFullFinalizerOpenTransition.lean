import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaMuxRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound

/-!
Contract: typed `maybe_open` relation for the exact terminal Nebula lane.

The relation keeps the post-phase lane separate from the opened lane. It
owns the exclusive open branch, the new-segment index rule, the transcript
gamma selection, the delayed `D_pre` selection, and all copied lane fields.

It does not own source decoding, lifecycle authority, delayed fresh-opening
authority, later leaf hashes, advance, or close.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenTransition

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound

namespace OpenSound

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenAlgebraRowSound.Sound

end OpenSound

namespace MuxSound

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaMuxRowSound.Sound

end MuxSound

def postPhaseLane (assignment : Nat -> Nat) : Lane :=
  laneAt assignment rawArtifact.laneColumns

def delayedOpen (assignment : Nat -> Nat) : Bool :=
  assignment rawArtifact.openColumn == 1

def candidateGamma (assignment : Nat -> Nat) : Fin 2 -> K := fun index =>
  kAt assignment rawArtifact.gammaMuxOpenedColumns (2 * index.val)

def candidateDPre (assignment : Nat -> Nat) : Fin 3 -> Digest := fun index =>
  digestAt assignment rawArtifact.gammaMuxOpenedColumns (4 + 4 * index.val)

/-- Native fixed-shape `maybe_open`: open a closed lane with transcript gamma
and delayed roots, or carry an already-open lane unchanged. -/
def maybeOpenLane
    (before : Lane) (opens : Bool)
    (gamma : Fin 2 -> K) (dPre : Fin 3 -> Digest) : Lane where
  programBindingDigest := before.programBindingDigest
  isOpen := true
  segmentIndex := before.segmentIndex
  stepIndex := before.stepIndex
  timestamp := before.timestamp
  gamma := if opens then gamma else before.gamma
  products := before.products
  stackPointers := before.stackPointers
  dPre := if opens then dPre else before.dPre
  dSeen := before.dSeen
  dMem := before.dMem

structure MaybeOpen
    (before : Lane) (opens : Bool)
    (gamma : Fin 2 -> K) (dPre : Fin 3 -> Digest)
    (after : Lane) : Prop where
  segmentIndexZero : before.segmentIndex = 0
  exclusiveOpen : boolValue before.isOpen + boolValue opens = 1
  newOpenZeroIndex : opens = true -> before.stepIndex = 0
  outputExact : after = maybeOpenLane before opens gamma dPre

private theorem boolValue_eq_assignment
    {value : Nat} (exact : value = 0 ∨ value = 1) :
    boolValue (value == 1) = value := by
  rcases exact with rfl | rfl <;> rfl

private theorem mux_field_exact
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (sound : MuxSound.Sound assignment canonical)
    (index : Nat) (bounded : index < 16) :
    fieldValue assignment
        (rawArtifact.gammaMuxOutputColumns.getD index 0) =
      if assignment rawArtifact.openColumn = 1 then
        fieldValue assignment
          (rawArtifact.gammaMuxOpenedColumns.getD index 0)
      else
        fieldValue assignment
          (rawArtifact.gammaMuxCarriedColumns.getD index 0) := by
  apply Fin.ext
  have reduce : forall column,
      assignment column % goldilocksModulus = assignment column := by
    intro column
    exact Nat.mod_eq_of_lt (canonical column)
  by_cases present : assignment rawArtifact.openColumn = 1
  · simpa [fieldValue, present, reduce] using sound.outputs index bounded
  · simpa [fieldValue, present, reduce] using sound.outputs index bounded

private theorem mux_k_exact
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (sound : MuxSound.Sound assignment canonical)
    (start : Nat) (lowBound : start < 16) (highBound : start + 1 < 16) :
    kAt assignment rawArtifact.gammaMuxOutputColumns start =
      if assignment rawArtifact.openColumn = 1 then
        kAt assignment rawArtifact.gammaMuxOpenedColumns start
      else
        kAt assignment rawArtifact.gammaMuxCarriedColumns start := by
  by_cases present : assignment rawArtifact.openColumn = 1
  · simp only [present, if_pos, kAt, K.mk.injEq]
    constructor
    · simpa [fieldAt, columnAt, present] using
        mux_field_exact assignment canonical sound start lowBound
    · simpa [fieldAt, columnAt, present] using
        mux_field_exact assignment canonical sound (start + 1) highBound
  · simp [present, kAt, K.mk.injEq]
    constructor
    · simpa [fieldAt, columnAt, present] using
        mux_field_exact assignment canonical sound start lowBound
    · simpa [fieldAt, columnAt, present] using
        mux_field_exact assignment canonical sound (start + 1) highBound

private theorem mux_digest_exact
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (sound : MuxSound.Sound assignment canonical)
    (start : Nat) (bounded : start + 3 < 16) :
    digestAt assignment rawArtifact.gammaMuxOutputColumns start =
      if assignment rawArtifact.openColumn = 1 then
        digestAt assignment rawArtifact.gammaMuxOpenedColumns start
      else
        digestAt assignment rawArtifact.gammaMuxCarriedColumns start := by
  funext index
  by_cases present : assignment rawArtifact.openColumn = 1
  · simpa [digestAt, fieldAt, columnAt, present] using
      mux_field_exact assignment canonical sound (start + index.val) (by omega)
  · simpa [digestAt, fieldAt, columnAt, present] using
      mux_field_exact assignment canonical sound (start + index.val) (by omega)

private theorem opened_gamma_layout
    (assignment : Nat -> Nat) (index : Fin 2) :
    (openedLane assignment).gamma index =
      kAt assignment rawArtifact.gammaMuxOutputColumns (2 * index.val) := by
  fin_cases index <;> rfl

private theorem carried_gamma_layout
    (assignment : Nat -> Nat) (index : Fin 2) :
    (postPhaseLane assignment).gamma index =
      kAt assignment rawArtifact.gammaMuxCarriedColumns (2 * index.val) := by
  fin_cases index <;> rfl

private theorem opened_dPre_layout
    (assignment : Nat -> Nat) (index : Fin 3) :
    (openedLane assignment).dPre index =
      digestAt assignment rawArtifact.gammaMuxOutputColumns
        (4 + 4 * index.val) := by
  funext coordinate
  change
    fieldValue assignment
        (rawArtifact.openedLaneColumns.getD
          (22 + 4 * index.val + coordinate.val) 0) =
      fieldValue assignment
        (rawArtifact.gammaMuxOutputColumns.getD
          (4 + 4 * index.val + coordinate.val) 0)
  rw [opened_lane_dPre_mux_column]

private theorem carried_dPre_layout
    (assignment : Nat -> Nat) (index : Fin 3) :
    (postPhaseLane assignment).dPre index =
      digestAt assignment rawArtifact.gammaMuxCarriedColumns
        (4 + 4 * index.val) := by
  funext coordinate
  change
    fieldValue assignment
        (rawArtifact.laneColumns.getD
          (22 + 4 * index.val + coordinate.val) 0) =
      fieldValue assignment
        (rawArtifact.gammaMuxCarriedColumns.getD
          (4 + 4 * index.val + coordinate.val) 0)
  rw [carried_lane_dPre_mux_column]

/-- The retained open-algebra, gamma-transcript, and gamma-mux rows imply the
complete typed `maybe_open` output. -/
theorem rows_imply_maybeOpen
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (openSound : OpenSound.Sound assignment)
    (muxSound : MuxSound.Sound assignment canonical) :
    MaybeOpen (postPhaseLane assignment) (delayedOpen assignment)
      (candidateGamma assignment) (candidateDPre assignment)
      (openedLane assignment) := by
  refine {
    segmentIndexZero := ?_
    exclusiveOpen := ?_
    newOpenZeroIndex := ?_
    outputExact := ?_ }
  · exact openSound.segmentIndexZero
  · change
      boolValue (assignment rawArtifact.laneOpenColumn == 1) +
        boolValue (assignment rawArtifact.openColumn == 1) = 1
    rw [boolValue_eq_assignment openSound.laneOpenExact,
      boolValue_eq_assignment openSound.inputOpenExact]
    exact openSound.exactlyOneOpen
  · intro opened
    have inputOne : assignment rawArtifact.openColumn = 1 := by
      simpa [delayedOpen] using opened
    exact openSound.newOpenZeroIndex inputOne
  · apply Lane.ext
    · funext output
      fin_cases output <;> rfl
    · change
        (assignment (rawArtifact.openedLaneColumns.getD 4 0) == 1) = true
      rw [opened_lane_open_column, one]
      rfl
    · rfl
    · rfl
    · rfl
    · funext index
      rw [opened_gamma_layout assignment index]
      simp only [maybeOpenLane]
      rw [ite_apply]
      fin_cases index
      · simpa [maybeOpenLane, delayedOpen, candidateGamma,
          carried_gamma_layout] using
          mux_k_exact assignment canonical muxSound 0 (by decide) (by decide)
      · simpa [maybeOpenLane, delayedOpen, candidateGamma,
          carried_gamma_layout] using
          mux_k_exact assignment canonical muxSound 2 (by decide) (by decide)
    · funext index
      fin_cases index <;> rfl
    · funext index
      fin_cases index <;> rfl
    · funext index
      rw [opened_dPre_layout assignment index]
      simp only [maybeOpenLane]
      rw [ite_apply]
      fin_cases index
      · simpa [maybeOpenLane, delayedOpen, candidateDPre,
          carried_dPre_layout] using
          mux_digest_exact assignment canonical muxSound 4 (by decide)
      · simpa [maybeOpenLane, delayedOpen, candidateDPre,
          carried_dPre_layout] using
          mux_digest_exact assignment canonical muxSound 8 (by decide)
      · simpa [maybeOpenLane, delayedOpen, candidateDPre,
          carried_dPre_layout] using
          mux_digest_exact assignment canonical muxSound 12 (by decide)
    · funext index output
      fin_cases index <;> fin_cases output <;> rfl
    · funext output
      fin_cases output <;> rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenTransition
