import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyDigestDomain.Checkpoint4

/-!
Contract: independent static initialization of the streaming PiRLC
family-state Poseidon2 transcript.

Owns the exact composition of the first three shared checkpoints, the changed
fourth checkpoint, and the post-initialization eight-lane state.

Does not own generated family-state columns, state preimages, public digest
words, or lifecycle integration.

Emits constraints: no.

Assurance tier: artifact-checked against the canonical Poseidon2 evaluator,
which is Rust-conformant through `Poseidon2ExtractedReference`.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-- Independent post-`Poseidon2Transcript::new` state for the PiRLC
family-state application label. -/
def domainInitialState : State :=
  appendMessage emptyState transcriptApplicationDomain stateDigestDomain

/-- Exact Rust-emitted constants for the independently certified initial
state. These values are not digest advice. -/
def initialStateValues : List Nat :=
  [27431110772485234, 212436215156, 17097887264920317608,
   5045320036345004475, 2778744841876081974, 15963932331085499115,
   9417555079406418274, 11480438335165740302]

def expectedDomainState : State where
  lanes := fun lane => fieldValue (initialStateValues.getD lane.val 0)
  absorbed := ⟨2, by decide⟩

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
    absorbWords (checkpointState checkpoint4Values) domainRemainder =
      expectedDomainState := by
  apply stateView_injective
  native_decide

/-- The four certified checkpoints compose to the exact PiRLC family-state
domain state. -/
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

/-- The byte-level domain model computes the exact eight constants and the
exact rate cursor used by both PiRLC arms. -/
theorem domain_initial_state_exact :
    (∀ lane : Fin 8,
      (domainInitialState.lanes lane).val =
        initialStateValues.getD lane.val 0) ∧
      domainInitialState.absorbed.val = 2 := by
  rw [domain_initial_state_state_exact]
  constructor
  · native_decide
  · rfl

/-- Packed `"state"` is one little-endian field word. -/
theorem state_fields_label_exact :
    packedBytesWithLen stateFieldsLabel = [5, 435744240755] := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain
