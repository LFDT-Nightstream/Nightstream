import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeDigestDomain.Checkpoint5

/-!
Contract: model-level initialization of the streaming Prelude replay-state
Poseidon2 transcript.

Owns the exact composition of four bounded permutation checkpoints and the
post-initialization eight-lane state. It does not own generated columns,
source rows, public words, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-- Post-`Poseidon2Transcript::new` state for the Prelude replay-state
application label. -/
def domainInitialState : State :=
  appendMessage emptyState transcriptApplicationDomain stateDigestDomain

def initialStateValues : List Nat :=
  [27431110773469033, 30522878494336372, 32758250074896737, 829828965,
   1988541141149579427, 4859373221894732330,
   9937262314844071878, 8401668388730343368]

def expectedDomainState : State where
  lanes := fun lane => fieldValue (initialStateValues.getD lane.val 0)
  absorbed := ⟨4, by decide⟩

private theorem domain_block1_full :
    (absorbWords emptyState domainBlock1).absorbed.val = rate := by
  native_decide

private theorem domain_block2_full :
    (absorbWords (checkpointState checkpoint1Values)
      domainBlock2).absorbed.val = rate := by
  native_decide

private theorem domain_block3_full :
    (absorbWords (checkpointState checkpoint2Values)
      domainBlock3).absorbed.val = rate := by
  native_decide

private theorem domain_block4_full :
    (absorbWords (checkpointState checkpoint3Values)
      domainBlock4).absorbed.val = rate := by
  native_decide

private theorem domain_remainder_exact :
    absorbWords (checkpointState checkpoint4Values) domainBlock5 =
      expectedDomainState := by
  apply stateView_injective
  native_decide

/-- The fixed receipt leaves compose to the exact Prelude domain state. -/
theorem domain_initial_state_state_exact :
    domainInitialState = expectedDomainState := by
  unfold domainInitialState appendMessage
  rw [domain_framing_words_exact]
  simp only [List.append_assoc]
  rw [absorbWords_append]
  rw [absorbWords_full _ _ domain_block1_full (by native_decide)]
  rw [checkpoint1_exact]
  rw [absorbWords_append]
  rw [absorbWords_full _ _ domain_block2_full (by native_decide)]
  rw [checkpoint2_exact]
  rw [absorbWords_append]
  rw [absorbWords_full _ _ domain_block3_full (by native_decide)]
  rw [checkpoint3_exact]
  rw [absorbWords_append]
  rw [absorbWords_full _ _ domain_block4_full (by native_decide)]
  rw [checkpoint4_exact, domain_remainder_exact]

theorem domain_initial_state_exact :
    (∀ lane : Fin 8,
      (domainInitialState.lanes lane).val =
        initialStateValues.getD lane.val 0) ∧
      domainInitialState.absorbed.val = 4 := by
  rw [domain_initial_state_state_exact]
  constructor
  · native_decide
  · rfl

/-- The state that Rust imports into the Prelude transcript gadget after the
exact-block boundary permutation. -/
def collapsedInitialState : State :=
  permute domainInitialState

theorem collapsed_initial_state_state_exact :
    collapsedInitialState = checkpointState collapsedInitialValues := by
  unfold collapsedInitialState
  rw [domain_initial_state_state_exact, ← domain_remainder_exact,
    checkpoint5_exact]

theorem collapsed_initial_state_exact :
    (∀ lane : Fin 8,
      (collapsedInitialState.lanes lane).val =
        collapsedInitialValues.getD lane.val 0) ∧
      collapsedInitialState.absorbed.val = 0 := by
  rw [collapsed_initial_state_state_exact]
  constructor
  · native_decide
  · rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain
