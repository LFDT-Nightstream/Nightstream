import DirectCcsFPrime.ProofSystem.Production.Impl.Runtime.DirectParentOnlyProductionFPrimePriorVerifier

/-!
Thin interface for the production prior F' verifier.

Spec: `specs/ProofSystem/Production/Impl/Runtime/DirectParentOnlyProductionFPrimePriorVerifier.spec.md`
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionFPrimePriorVerifierInterface

export DirectParentOnlyProductionFPrimePriorVerifier (
  BackendSoundness
  OpeningSurface
  RuntimeExactSurface
  RuntimeExactLayout
  RuntimeExactVerify
  runtimeSoundness
  backendSoundnessOfOpening
  runtimeSoundnessOfOpening
  openingSurface
  verifyOpensOfOpening
  verifyAuditOfOpening
  auditOpens
  auditOpensOfOpening
  auditReaches
  auditReachesOfOpening
  certified
  certifiedOfOpening
  certifiedOfRuntimeExact
  sound
  soundOfOpening
  soundOfRuntimeExact
  soundAcceptsAudit
  soundAcceptsVerifyOfOpening
  verifyOpensOfRuntimeExact
  openedAuthorityAcceptsOfRuntimeExact
  cannotAcceptWithoutOpeningOfRuntimeExact
  soundAcceptsVerifyOfRuntimeExact
  soundSameProof
  soundSameProofOfOpening
  soundSameProofOfRuntimeExact
  publicImageInvariantsOfRuntimeExact
  endToEnd
  endToEndOfOpening
  endToEndOfRuntimeExact
  privateDecFacts
  privateDecFactsOfOpening
  privateDecFactsOfRuntimeExact
  privateDecNoSwapAuditOfRuntimeExact
  stageAudit
  stageAuditOfOpening
  stageAuditOfRuntimeExact
)

end DirectParentOnlyProductionFPrimePriorVerifierInterface

end DirectCcsFPrime
