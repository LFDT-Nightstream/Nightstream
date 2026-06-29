import DirectCcsFPrime.Audit.FPrimeTraceAuthority

/-!
Typed interface for trace-carrying folded F' authority.

Spec: `specs/Audit/FPrimeTraceAuthority.spec.md`
-/

namespace DirectCcsFPrime

namespace FPrimeTraceAuthorityInterface

abbrev Authority :=
  @FPrimeTraceAuthority.Authority

abbrev reachable :=
  @FPrimeTraceAuthority.reachable

abbrev toFoldedAuthority :=
  @FPrimeTraceAuthority.toFoldedAuthority

abbrev toFoldedAuthority_accepts :=
  @FPrimeTraceAuthority.toFoldedAuthority_accepts

abbrev VerifierSound :=
  @FPrimeTraceAuthority.VerifierSound

abbrev verifierSound_priorAuthoritySound :=
  @FPrimeTraceAuthority.verifierSound_priorAuthoritySound

abbrev SoundVerifier :=
  @FPrimeTraceAuthority.SoundVerifier

abbrev soundVerifier_priorAuthoritySound :=
  @FPrimeTraceAuthority.SoundVerifier.priorAuthoritySound

abbrev soundVerifier_toCompressedSoundVerifier :=
  @FPrimeTraceAuthority.SoundVerifier.toCompressedSoundVerifier

end FPrimeTraceAuthorityInterface

end DirectCcsFPrime
