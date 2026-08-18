import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeCollapsedDomainReceipt
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeDigestDomain.Checkpoint4

/-! Artifact-checked boundary permutation for the native Prelude transcript state. -/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Poseidon2ExtractedReference

def checkpoint5InputValues : List Nat :=
  inputValues

def checkpoint5InputState : PiRlcChallenge.TranscriptMachine.State :=
  stateFromValues checkpoint5InputValues ⟨4, by decide⟩

/-- Exact native state imported by `TranscriptGadget::new` after the full
application-domain block is collapsed. -/
def collapsedInitialValues : List Nat :=
  [2644702416324735075, 8852586734026622474,
   10961611613478088853, 3758899379070171657,
   9085448732628946009, 13680608928383082747,
   1991093790229263654, 6906233132260090641]

def checkpoint5InitialStates : List (List Nat) :=
  initialStates

def checkpoint5PartialStates : List (List Nat) :=
  partialStates

def checkpoint5TerminalStates : List (List Nat) :=
  terminalStates

private theorem checkpoint5_input_exact :
    absorbWords (checkpointState checkpoint4Values) domainBlock5 =
      checkpoint5InputState := by
  apply stateView_injective
  native_decide

private theorem checkpoint5_initial_valid :
    PhaseReceipt.Valid checkpoint5InitialStates halfFullRounds
      (applyMatrixValues Poseidon2Matrices.externalMatrix
        (valuesOf checkpoint5InputValues))
      (fun round => fullRoundValues (selected.initial round)) := by
  constructor
  · native_decide
  · funext lane
    fin_cases lane <;> native_decide
  · intro round roundLt
    change round < 4 at roundLt
    interval_cases round <;> native_decide

private theorem checkpoint5_partial_valid :
    PhaseReceipt.Valid checkpoint5PartialStates partialRounds
      (receiptState checkpoint5InitialStates halfFullRounds)
      (fun round => partialRoundValues (selected.internal round)) := by
  constructor
  · native_decide
  · funext lane
    fin_cases lane <;> native_decide
  · intro round roundLt
    change round < 22 at roundLt
    interval_cases round <;> native_decide

private theorem checkpoint5_terminal_valid :
    PhaseReceipt.Valid checkpoint5TerminalStates halfFullRounds
      (receiptState checkpoint5PartialStates partialRounds)
      (fun round => fullRoundValues (selected.terminal round)) := by
  constructor
  · native_decide
  · funext lane
    fin_cases lane <;> native_decide
  · intro round roundLt
    change round < 4 at roundLt
    interval_cases round <;> native_decide

private theorem checkpoint5_reference_exact :
    referencePermutation selected (valuesOf checkpoint5InputValues) =
      valuesOf collapsedInitialValues := by
  have initialExact := checkpoint5_initial_valid.finalExact
  rw [runInitial_eq_refInitial] at initialExact
  have partialExact := checkpoint5_partial_valid.finalExact
  rw [initialExact, runPartial_eq_refPartial] at partialExact
  have terminalExact := checkpoint5_terminal_valid.finalExact
  rw [partialExact, runTerminal_eq_refTerminal] at terminalExact
  rw [referencePermutation]
  exact terminalExact.symm.trans (by native_decide)

theorem checkpoint5_exact :
    permute (absorbWords (checkpointState checkpoint4Values) domainBlock5) =
      checkpointState collapsedInitialValues := by
  rw [checkpoint5_input_exact]
  apply state_ext
  · intro lane
    change
      Poseidon2PermutationSound.permute
          (laneNat checkpoint5InputState) lane.val % goldilocksP = _
    have inputFunctionsEqual :
        laneNat checkpoint5InputState =
          stateLaneValues checkpoint5InputState := by
      funext inputLane
      rfl
    rw [inputFunctionsEqual,
      Nat.mod_eq_of_lt
        (Poseidon2PermutationSound.permute_lt
          (stateLaneValues_canonical checkpoint5InputState) lane.val),
      Poseidon2ExtractedReference.permute_eq_reference
        (stateLaneValues_canonical checkpoint5InputState) lane]
    have inputExact :
        (fun inputLane : Fin Poseidon2Core.width =>
          stateLaneValues checkpoint5InputState inputLane.val) =
          valuesOf checkpoint5InputValues := by
      funext inputLane
      fin_cases inputLane <;> native_decide
    rw [inputExact, checkpoint5_reference_exact]
    simp [valuesOf, checkpointState, stateFromValues, fieldValue]
    fin_cases lane <;> native_decide
  · rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain
