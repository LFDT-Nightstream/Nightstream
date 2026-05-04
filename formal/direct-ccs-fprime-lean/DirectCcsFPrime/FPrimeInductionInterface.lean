import DirectCcsFPrime.FPrimeInduction

/-!
Typed interface for F' induction authority.

Spec: `specs/FPrimeInduction.spec.md`
-/

namespace DirectCcsFPrime

namespace FPrimeInductionInterface

abbrev Reachable :=
  @FPrimeInduction.Reachable

abbrev PriorAuthoritySound :=
  @FPrimeInduction.PriorAuthoritySound

abbrev LatestStepSound :=
  @FPrimeInduction.LatestStepSound

abbrev TerminalCompressionAccepted :=
  @FPrimeInduction.TerminalCompressionAccepted

abbrev terminal_compression_reaches_final :=
  @FPrimeInduction.terminal_compression_reaches_final

abbrev BaseAuthorityAccepts :=
  @FPrimeInduction.BaseAuthorityAccepts

abbrev base_authority_sound :=
  @FPrimeInduction.base_authority_sound

abbrev digest_only_acceptance_not_sound_when_it_accepts_unreachable :=
  @FPrimeInduction.digest_only_acceptance_not_sound_when_it_accepts_unreachable

end FPrimeInductionInterface

end DirectCcsFPrime
