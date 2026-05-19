import DirectCcsFPrime.ReducedHandleTerminal

/-!
Interface for reduced-handle terminal composition.

Spec: `specs/ReducedHandleTerminal.spec.md`
-/

namespace DirectCcsFPrime

namespace ReducedHandleTerminalInterface

abbrev prior_authority_soundness :=
  @ReducedHandleTerminal.prior_authority_soundness

abbrev accepted_unreachable_prior_is_not_sound_authority :=
  @ReducedHandleTerminal.accepted_unreachable_prior_is_not_sound_authority

abbrev compressed_prior_soundness :=
  @ReducedHandleTerminal.compressed_prior_soundness

abbrev compressed_prior_soundness_of_opens_to_folded_authority :=
  @ReducedHandleTerminal.compressed_prior_soundness_of_opens_to_folded_authority

abbrev compressed_prior_soundness_of_sound_verifier :=
  @ReducedHandleTerminal.compressed_prior_soundness_of_sound_verifier

abbrev proof_carrying_soundness :=
  @ReducedHandleTerminal.proof_carrying_soundness

end ReducedHandleTerminalInterface

end DirectCcsFPrime
