import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeDigestDomain
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeRelation

/-!
Contract: production Poseidon2 semantics for the ten-field Prelude state digest.

Owns the native transcript fold, the exact `"state"` framing, and its
structural equality with the four-call fixed schedule. It owns no generated
row, column, or assignment.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeStateDigest

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelation
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

def stateFieldsWords (fields : List Nat) : List Nat :=
  [2] ++ packedBytesWithLen stateFieldsLabel ++ [fields.length] ++ fields

/-- Native `append_fields` followed by `digest_fields`, including the digest
pad word and final permutation. -/
def nativeStateDigestState (fields : List Nat) : State :=
  (digest (absorbWords collapsedInitialState (stateFieldsWords fields))).1

def stateDigest (fields : List Nat) : Digest :=
  fun lane => (nativeStateDigestState fields).lanes
    ⟨lane.val, by
      have laneLt := lane.isLt
      simp only [width]
      omega⟩

def productionSemantics : Semantics where
  stateDigest := stateDigest

def stateFieldsFrame10 : List Nat :=
  [2, 5, 435744240755, 10]

def zeroBlock4 : List Nat :=
  [0, 0, 0, 0]

def zeroBlock2 : List Nat :=
  [0, 0]

def frameState : State :=
  absorbWords collapsedInitialState stateFieldsFrame10

def firstFieldsState : State :=
  absorbWords (permute frameState) zeroBlock4

def secondFieldsState : State :=
  absorbWords (permute firstFieldsState) zeroBlock4

def digestInputState : State :=
  absorbWords (permute secondFieldsState) (zeroBlock2 ++ [1])

def scheduledInitialDigestState : State :=
  permute digestInputState

private theorem initial_words_exact :
    stateFieldsWords initialReplayFields =
      stateFieldsFrame10 ++ (zeroBlock4 ++ (zeroBlock4 ++ zeroBlock2)) := by
  native_decide

private theorem frameState_full : frameState.absorbed.val = rate := by
  rfl

private theorem firstFieldsState_full :
    firstFieldsState.absorbed.val = rate := by
  rfl

private theorem secondFieldsState_full :
    secondFieldsState.absorbed.val = rate := by
  rfl

/-- The independent native transcript fold has exactly the four permutation
boundaries emitted by Rust for the fixed ten-field Prelude input. -/
theorem nativeStateDigestState_initial_exact :
    nativeStateDigestState initialReplayFields =
      scheduledInitialDigestState := by
  unfold nativeStateDigestState
  rw [initial_words_exact, absorbWords_append]
  change
    (digest
      (absorbWords frameState (zeroBlock4 ++ (zeroBlock4 ++ zeroBlock2)))).1 = _
  rw [absorbWords_full frameState _ frameState_full (by decide)]
  rw [absorbWords_append]
  change
    (digest
      (absorbWords firstFieldsState (zeroBlock4 ++ zeroBlock2))).1 = _
  rw [absorbWords_full firstFieldsState _ firstFieldsState_full (by decide)]
  rw [absorbWords_append]
  change
    (digest (absorbWords secondFieldsState zeroBlock2)).1 = _
  rw [absorbWords_full secondFieldsState _ secondFieldsState_full (by decide)]
  rfl

theorem stateDigest_initial_exact :
    stateDigest initialReplayFields =
      fun lane => scheduledInitialDigestState.lanes
        ⟨lane.val, by
          have laneLt := lane.isLt
          simp only [width]
          omega⟩ := by
  funext lane
  unfold stateDigest
  rw [nativeStateDigestState_initial_exact]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeStateDigest
