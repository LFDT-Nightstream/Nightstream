import DirectCcsFPrime.DirectParentOnlyProductionFPrimeRuntimeVerifier

/-!
Thin interface for the production prior F' runtime verifier.

Spec: `specs/DirectParentOnlyProductionFPrimeRuntimeVerifier.spec.md`
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionFPrimeRuntimeVerifierInterface

export DirectParentOnlyProductionFPrimeRuntimeVerifier (
  openingSurface
  verifyOpens
  auditOpens
  auditOpenAuthority_ne_none
  auditOpenedAuthorityAccepts
  auditReaches
  auditPublicImageInvariants
  auditCannotAcceptUnreachable
  compressedSoundness
  certified
  certifiedAcceptsAudit
  sound
  soundAcceptsAudit
  soundOpens
  soundSameProof
  acceptedTerminal
  endToEnd
  privateDecFacts
  stageAudit
)

end DirectParentOnlyProductionFPrimeRuntimeVerifierInterface

end DirectCcsFPrime
