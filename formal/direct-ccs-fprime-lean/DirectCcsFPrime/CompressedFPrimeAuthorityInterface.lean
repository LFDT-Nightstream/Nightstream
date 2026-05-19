import DirectCcsFPrime.CompressedFPrimeAuthority

/-!
Typed interface for compressed folded F' authority.

Spec: `specs/CompressedFPrimeAuthority.spec.md`
-/

namespace DirectCcsFPrime

namespace CompressedFPrimeAuthorityInterface

abbrev VerifierSound :=
  @CompressedFPrimeAuthority.VerifierSound

abbrev Accepts :=
  @CompressedFPrimeAuthority.Accepts

abbrev SoundVerifier :=
  @CompressedFPrimeAuthority.SoundVerifier

abbrev SoundVerifier.Accepts :=
  @CompressedFPrimeAuthority.SoundVerifier.Accepts

abbrev ProofFunctional :=
  @CompressedFPrimeAuthority.ProofFunctional

abbrev SoundVerifier.ProofFunctional :=
  @CompressedFPrimeAuthority.SoundVerifier.ProofFunctional

abbrev accepts_sound_of_verifier_sound :=
  @CompressedFPrimeAuthority.accepts_sound_of_verifier_sound

abbrev verifier_sound_of_opens_to_folded_authority :=
  @CompressedFPrimeAuthority.verifier_sound_of_opens_to_folded_authority

abbrev verifier_sound_of_sound_verifier :=
  @CompressedFPrimeAuthority.verifier_sound_of_sound_verifier

abbrev sound_verifier_prior_authority_sound :=
  @CompressedFPrimeAuthority.sound_verifier_prior_authority_sound

abbrev terminal_compression_reaches_final :=
  @CompressedFPrimeAuthority.terminal_compression_reaches_final

abbrev terminal_compression_reaches_final_of_opens_to_folded_authority :=
  @CompressedFPrimeAuthority.terminal_compression_reaches_final_of_opens_to_folded_authority

abbrev terminal_compression_reaches_final_of_sound_verifier :=
  @CompressedFPrimeAuthority.terminal_compression_reaches_final_of_sound_verifier

abbrev accepted_unreachable_is_not_sound_verifier :=
  @CompressedFPrimeAuthority.accepted_unreachable_is_not_sound_verifier

abbrev sound_verifier_does_not_imply_same_proof_functional :=
  @CompressedFPrimeAuthority.sound_verifier_does_not_imply_same_proof_functional

end CompressedFPrimeAuthorityInterface

end DirectCcsFPrime
