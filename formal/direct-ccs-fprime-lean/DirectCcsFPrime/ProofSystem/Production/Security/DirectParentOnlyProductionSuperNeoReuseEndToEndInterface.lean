import DirectCcsFPrime.ProofSystem.Production.Security.DirectParentOnlyProductionSuperNeoReuseEndToEnd
import DirectCcsFPrime.ProofSystem.Production.Impl.Runtime.DirectParentOnlyProductionFPrimePriorVerifierInterface
import DirectCcsFPrime.ProofSystem.Production.Impl.Runtime.DirectParentOnlyProductionExactRuntimeInstantiationInterface
import DirectCcsFPrime.ProofSystem.Terminal.Security.ParentOnlyPrivateChildrenFlowInterface

/-!
Production interface for the parent-only terminal theorem.

Spec: `specs/ProofSystem/Production/Security/DirectParentOnlyProductionSuperNeoReuseEndToEnd.spec.md`

The advertised entry point is the exact-runtime F' prior verifier path. Lower
level opening lemmas remain in implementation modules for proof factoring, but
this interface does not re-export caller-supplied opening-premise surfaces.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuseEndToEndInterface

/-- Production context used by the parent-only terminal theorem. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.ProductionContext

/-- Final certified terminal package for the parent-only path. -/
abbrev CertifiedTerminalEndToEnd :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalEndToEnd

/-- Flattened non-aggregate private DEC and stage facts. -/
abbrev CertifiedTerminalNonAggregatePrivateDecStageFacts :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalNonAggregatePrivateDecStageFacts

/-- Named certificate carried by the non-aggregate private DEC/stage package. -/
abbrev CertifiedTerminalNonAggregatePrivateDecStageCertificate :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalNonAggregatePrivateDecStageCertificate

/-- Extract non-aggregate private DEC/stage facts from the final package. -/
abbrev nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd

/-- Extract the private DEC certificate from the non-aggregate facts. -/
abbrev privateDecCertificate_of_nonAggregatePrivateDecStageFacts :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.privateDecCertificate_of_nonAggregatePrivateDecStageFacts

/-- Extract the child audit trail from the non-aggregate facts. -/
abbrev childAuditTrail_of_nonAggregatePrivateDecStageFacts :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.childAuditTrail_of_nonAggregatePrivateDecStageFacts

/-- Extract pointwise private-child uniqueness from the non-aggregate facts. -/
abbrev uniquePrivateChildren_of_nonAggregatePrivateDecStageFacts :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.uniquePrivateChildren_of_nonAggregatePrivateDecStageFacts

/-- Extract exact child-witness to next-`Pi_CCS` wire identity. -/
abbrev nextPiCCSInputs_eq_childWitnessDigitTable_of_nonAggregatePrivateDecStageFacts :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.nextPiCCSInputs_eq_childWitnessDigitTable_of_nonAggregatePrivateDecStageFacts

/-- Extract the Section 7.1 owner-target audit from the final package. -/
abbrev section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd :=
  @DirectParentOnlyProductionSuperNeoReuseEndToEnd.section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd

/-! ## Exact-runtime F' prior verifier entry point -/

/-- Implementation-shaped exact-runtime prior verifier surface. -/
abbrev RuntimeExactSurface :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.RuntimeExactSurface

/-- Public-IO layout binding for the exact-runtime verifier surface. -/
abbrev RuntimeExactLayout :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.RuntimeExactLayout

/-- Exact-runtime prior verifier acceptance predicate. -/
abbrev RuntimeExactVerify :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.RuntimeExactVerify

/-- Exact-runtime acceptance opens folded F' authority for the same image. -/
abbrev verifyOpensOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.verifyOpensOfRuntimeExact

/-- Opened authority from exact-runtime acceptance satisfies folded F' reachability. -/
abbrev openedAuthorityAcceptsOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.openedAuthorityAcceptsOfRuntimeExact

/-- Exact-runtime acceptance cannot succeed when the fixed opener is empty. -/
abbrev cannotAcceptWithoutOpeningOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.cannotAcceptWithoutOpeningOfRuntimeExact

/-- Exact-runtime acceptance preserves prior public-image invariants. -/
abbrev publicImageInvariantsOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.publicImageInvariantsOfRuntimeExact

/-- Exact-runtime sound verifier accepts the same opaque proof consistently. -/
abbrev soundSameProofOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.soundSameProofOfRuntimeExact

/--
Canonical production terminal theorem.

Exact-runtime prior verifier acceptance plus one latest Construction-2 step
returns the parent-only terminal end-to-end package.
-/
abbrev endToEndOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.endToEndOfRuntimeExact

/-- Canonical production projection for non-aggregate private DEC/stage facts. -/
abbrev privateDecFactsOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.privateDecFactsOfRuntimeExact

/-- Canonical production no-swap audit for alternate pointwise-valid child tables. -/
abbrev privateDecNoSwapAuditOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.privateDecNoSwapAuditOfRuntimeExact

/-- Canonical production Section 7.1 owner-target stage audit. -/
abbrev stageAuditOfRuntimeExact :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.stageAuditOfRuntimeExact

/-! ## Short aliases for normal call sites -/

abbrev terminalEndToEnd :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.endToEndOfRuntimeExact

abbrev privateDecFacts :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.privateDecFactsOfRuntimeExact

abbrev privateDecNoSwapAudit :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.privateDecNoSwapAuditOfRuntimeExact

abbrev section71StageAudit :=
  @DirectParentOnlyProductionFPrimePriorVerifierInterface.stageAuditOfRuntimeExact

/-! ## Production exact-runtime instantiation -/

/-- Production exact verifier checks for the parent-only terminal path. -/
abbrev ProductionExactChecks :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.ExactChecks

/-- Runtime authority-soundness boundary for production exact checks. -/
abbrev ProductionRuntimeAuthoritySoundness :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.RuntimeAuthoritySoundness

/-- Opening surface induced by production exact checks and runtime soundness. -/
abbrev productionOpeningSurface :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.openingSurface

/-- Production exact prior verifier acceptance opens folded F' authority. -/
abbrev productionVerifyPriorOpens :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.verifyPriorOpens

/-- Production exact prior verifier acceptance proves prior reachability. -/
abbrev productionVerifyPriorReaches :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.verifyPriorReaches

/-- Production exact prior verifier acceptance rejects unreachable prior images. -/
abbrev productionVerifyPriorRejectsUnreachable :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.verifyPriorRejectsUnreachable

/-- Production exact checks plus latest-step evidence give terminal soundness. -/
abbrev productionTerminalSoundness :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.parentOnlyTerminalSoundness

/-- Production exact projection for non-aggregate private DEC/stage facts. -/
abbrev productionPrivateDecFacts :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.privateDecFacts

/-- Production exact no-swap audit for alternate pointwise-valid child tables. -/
abbrev productionPrivateDecNoSwapAudit :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.privateDecNoSwapAudit

/-- Production exact Section 7.1 owner-target stage audit. -/
abbrev productionSection71StageAudit :=
  @DirectParentOnlyProductionExactRuntimeInstantiationInterface.section71StageAudit

/-! ## Parent-only private-child flow facts -/

abbrev private_children_flow_of_parent_only_step :=
  @ParentOnlyPrivateChildrenFlowInterface.private_children_flow_of_parent_only_step

abbrev same_private_child_inputs_without_public_child_hashes :=
  @ParentOnlyPrivateChildrenFlowInterface.same_private_child_inputs_without_public_child_hashes

abbrev same_next_parent_source_without_public_child_hashes :=
  @ParentOnlyPrivateChildrenFlowInterface.same_next_parent_source_without_public_child_hashes

end DirectParentOnlyProductionSuperNeoReuseEndToEndInterface

end DirectCcsFPrime
