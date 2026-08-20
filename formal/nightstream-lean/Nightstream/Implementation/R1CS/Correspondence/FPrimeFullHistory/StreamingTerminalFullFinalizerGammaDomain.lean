import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainWitness

/-!
Contract: fixed leaf certificate for the terminal Nebula gamma application
domain.

Owns one fixed Poseidon2 capacity-lane checkpoint and its composition with
the two shared transcript-domain checkpoints. It does not own generated
columns, gamma challenges, output muxes, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomain

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainModel
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainWitness

private theorem checkpoint3_remainder_exact :
    absorbWords (permute checkpoint3InputState) gammaDomainRemainder =
      expectedDomainState := by
  rcases checkpoint3_capacity_exact with
    ⟨lane3, lane4, lane5, lane6, lane7⟩
  apply state_ext
  · intro lane
    fin_cases lane
    · rfl
    · rfl
    · rfl
    · exact lane3
    · exact lane4
    · exact lane5
    · exact lane6
    · exact lane7
  · rfl

/-- The byte-level application label computes the exact state pinned by the
Rust transcript gadget. -/
theorem domain_initial_state_exact :
    domainInitialState = expectedDomainState := by
  unfold domainInitialState appendMessage
  rw [gamma_domain_framing_exact]
  simp only [List.append_assoc]
  rw [absorbWords_append]
  rw [absorbWords_full _ _ domain_block1_full (by decide)]
  rw [checkpoint1_exact]
  rw [absorbWords_append]
  rw [absorbWords_full _ _ domain_block2_full (by decide)]
  rw [checkpoint2_exact]
  rw [absorbWords_append]
  rw [absorbWords_full _ _ gamma_domain_block3_full (by decide)]
  rw [checkpoint3_input_exact, checkpoint3_remainder_exact]

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomain
