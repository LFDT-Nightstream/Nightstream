import DirectCcsFPrime.DirectParentOnlyProductionExactRuntimeInstantiation

/-!
Thin interface for the production exact-runtime parent-only instantiation.

Spec: `specs/DirectParentOnlyProductionExactRuntimeInstantiation.spec.md`
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionExactRuntimeInstantiationInterface

export DirectParentOnlyProductionExactRuntimeInstantiation (
  ExactChecks
  RuntimeAuthoritySoundness
  VerifyPrior
  openingSurface
  runtimeSurface
  certifiedPriorVerifier
  soundPriorVerifier
  verifyPriorOpens
  verifyPriorAsRuntimeSurface
  certifiedPriorVerifierAccepts
  soundPriorVerifierAccepts
  verifyPriorReaches
  verifyPriorRejectsUnreachable
  parentOnlyTerminalSoundness
  privateDecFacts
  privateDecNoSwapAudit
  section71StageAudit
)

end DirectParentOnlyProductionExactRuntimeInstantiationInterface

end DirectCcsFPrime
